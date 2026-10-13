"""
metrics.py - Métricas de desempenho e risco sobre séries diárias.

Convenções: retornos simples diários; rf diária vinda do CDI histórico.
Usar Selic constante num período em que ela foi de 2% a 15% distorce o
Sharpe. Anualização por 252.

Correções em relação à versão anterior:
- Sortino: o denominador é o downside deviation sqrt(mean(min(x,0)^2)) sobre
  TODAS as observações. A versão anterior usava o desvio-padrão do
  subconjunto negativo, que mede a dispersão em torno da média das perdas e
  não a distância ao alvo. Pode ser 0 com perdas reais e retornava +inf
  (contraexemplo verificado).
- Calmar: CAGR / |MDD| (definição de Young, 1991), sem subtrair rf.
- Alfa de Jensen: intercepto da regressão dos excessos diários, não a
  diferença de CAGRs.
- Information ratio: média anualizada do retorno ativo dividida pelo
  tracking error.
- Curtose reportada como EXCESSO (normal = 0). O texto antigo dizia "> 3".

PSR/DSR usam Sharpe e momentos POR PERÍODO do excesso sobre o CDI e a curtose
BRUTA (normal = 3), como nos artigos originais. Não misturar com ``sharpe``,
que é anualizado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from core.config import DIAS_UTEIS_ANO

_RAIZ_ANO = np.sqrt(DIAS_UTEIS_ANO)


def retornos_de_valores(valores: pd.Series) -> pd.Series:
    return valores.pct_change().dropna()


def cagr(valores: pd.Series) -> float:
    n_periodos = len(valores) - 1
    if n_periodos <= 0 or valores.iloc[0] <= 0:
        return float("nan")
    return float((valores.iloc[-1] / valores.iloc[0]) ** (DIAS_UTEIS_ANO / n_periodos) - 1.0)


def drawdown(valores: pd.Series) -> tuple[pd.Series, float, int]:
    """(série de drawdown, drawdown máximo <= 0, maior sequência de pregões abaixo do pico)."""
    pico = valores.cummax()
    serie = valores / pico - 1.0
    abaixo = (serie < 0).astype(int)
    # comprimento da sequência corrente: cumsum que reinicia a cada pregão no pico
    sequencias = abaixo.groupby((abaixo == 0).cumsum()).cumsum()
    return serie, float(serie.min()), int(sequencias.max()) if len(sequencias) else 0


def sharpe(excesso_diario: pd.Series) -> float:
    desvio = excesso_diario.std()
    return float(excesso_diario.mean() / desvio * _RAIZ_ANO) if desvio > 0 else float("nan")


def downside_deviation(excesso_diario: pd.Series) -> float:
    """sqrt( (1/T) * sum(min(x_t, 0)^2) ) sobre todas as T observações."""
    return float(np.sqrt(np.mean(np.minimum(excesso_diario.values, 0.0) ** 2)))


def sortino(excesso_diario: pd.Series) -> float:
    dd = downside_deviation(excesso_diario)
    if dd == 0:
        return float("inf") if excesso_diario.mean() > 0 else float("nan")
    return float(excesso_diario.mean() / dd * _RAIZ_ANO)


def omega(excesso_diario: pd.Series) -> float:
    """Omega(rf) = E[(x)+] / E[(x)-], sobre o excesso diário."""
    ganhos = excesso_diario.clip(lower=0).sum()
    perdas = -excesso_diario.clip(upper=0).sum()
    return float(ganhos / perdas) if perdas > 0 else float("inf")


def var_historico(retornos: pd.Series, confianca: float = 0.95) -> float:
    """VaR diário histórico, como perda positiva."""
    return float(-np.percentile(retornos.values, (1 - confianca) * 100))


def var_cornish_fisher(retornos: pd.Series, confianca: float = 0.95) -> float:
    """VaR diário paramétrico com ajuste de assimetria e curtose (Cornish-Fisher)."""
    z = stats.norm.ppf(1 - confianca)
    s = stats.skew(retornos.values)
    k = stats.kurtosis(retornos.values)  # excesso
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(retornos.mean() + z_cf * retornos.std()))


def cvar_historico(retornos: pd.Series, confianca: float = 0.95) -> float:
    """Expected shortfall diário: média das perdas na cauda além do VaR."""
    limite = -var_historico(retornos, confianca)
    cauda = retornos[retornos <= limite]
    return float(-cauda.mean()) if len(cauda) else var_historico(retornos, confianca)


# ------------------------------------------------- Sharpe estatístico (PSR/DSR)

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class MomentosSharpe:
    """Insumos do PSR, todos POR PERÍODO (diários, não anualizados)."""

    sharpe: float          # média / desvio-padrão (ddof=1) do excesso
    n_obs: int
    assimetria: float
    curtose: float         # curtose BRUTA (normal = 3), não o excesso


def momentos_sharpe(excesso_diario: pd.Series) -> MomentosSharpe:
    """Sharpe diário e momentos do EXCESSO sobre o CDI (não dos retornos brutos)."""
    valores = excesso_diario.dropna().values
    # Série constante: np.std devolve ~1e-19 por arredondamento, não 0. Testar a amplitude.
    if len(valores) < 2 or np.ptp(valores) == 0:
        return MomentosSharpe(float("nan"), len(valores), float("nan"), float("nan"))
    return MomentosSharpe(
        sharpe=float(np.mean(valores) / np.std(valores, ddof=1)),
        n_obs=len(valores),
        assimetria=float(stats.skew(valores)),
        curtose=float(stats.kurtosis(valores, fisher=False)),
    )


def sharpe_probabilistico(m: MomentosSharpe, sharpe_referencia: float = 0.0) -> float:
    """
    PSR (Bailey e López de Prado, 2012): P[Sharpe verdadeiro > referência].

        PSR = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )

    SR e SR* por período; g3 = assimetria; g4 = curtose bruta. Supõe retornos
    estacionários e independentes no tempo: a autocorrelação não entra no erro
    padrão. Devolve NaN se o Sharpe não existe ou se T < 2.
    """
    if not np.isfinite(m.sharpe) or m.n_obs < 2:
        return float("nan")
    variancia = 1.0 - m.assimetria * m.sharpe + (m.curtose - 1.0) / 4.0 * m.sharpe**2
    if variancia <= 0:
        return float("nan")
    z = (m.sharpe - sharpe_referencia) * np.sqrt(m.n_obs - 1) / np.sqrt(variancia)
    return float(stats.norm.cdf(z))


def sharpe_referencia_deflacionado(variancia_sharpes: float, n_tentativas: int) -> float:
    """
    SR0 do DSR (Bailey e López de Prado, 2014): Sharpe máximo esperado entre N
    tentativas independentes cujo Sharpe verdadeiro é zero.

        SR0 = sqrt(V) [ (1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)) ],  g = Euler-Mascheroni

    V = variância dos Sharpes POR PERÍODO das tentativas. Com N = 1 não há
    seleção e SR0 = 0 (a fórmula daria Phi^-1(0) = -inf). O DSR é o PSR com
    referência SR0. Tentativas correlacionadas são menos de N independentes,
    então o SR0 fica conservador nesse ponto. Tentativas não contadas (ajustes
    manuais de parâmetros) o deixam otimista.
    """
    if n_tentativas < 1:
        raise ValueError(f"n_tentativas deve ser >= 1: {n_tentativas}")
    if n_tentativas == 1:
        return 0.0
    quantil_n = stats.norm.ppf(1.0 - 1.0 / n_tentativas)
    quantil_ne = stats.norm.ppf(1.0 - 1.0 / (n_tentativas * np.e))
    return float(np.sqrt(variancia_sharpes)
                 * ((1.0 - _EULER_MASCHERONI) * quantil_n + _EULER_MASCHERONI * quantil_ne))


@dataclass(frozen=True)
class ResumoDesempenho:
    retorno_total: float
    cagr: float
    volatilidade: float
    sharpe: float
    sortino: float
    calmar: float
    omega: float
    max_drawdown: float
    duracao_max_drawdown: int
    var_95_diario: float
    cvar_95_diario: float
    assimetria: float
    curtose_excesso: float
    n_pregoes: int


def resumo_desempenho(valores: pd.Series, rf_diaria: pd.Series) -> ResumoDesempenho:
    """Todas as métricas de uma curva de capital diária."""
    retornos = retornos_de_valores(valores)
    excesso = retornos - rf_diaria.reindex(retornos.index).fillna(0.0)
    _, mdd, duracao = drawdown(valores)
    taxa_composta = cagr(valores)
    return ResumoDesempenho(
        retorno_total=float(valores.iloc[-1] / valores.iloc[0] - 1.0),
        cagr=taxa_composta,
        volatilidade=float(retornos.std() * _RAIZ_ANO),
        sharpe=sharpe(excesso),
        sortino=sortino(excesso),
        calmar=float(taxa_composta / abs(mdd)) if mdd < 0 else float("inf"),
        omega=omega(excesso),
        max_drawdown=mdd,
        duracao_max_drawdown=duracao,
        var_95_diario=var_cornish_fisher(retornos),
        cvar_95_diario=cvar_historico(retornos),
        assimetria=float(stats.skew(retornos.values)),
        curtose_excesso=float(stats.kurtosis(retornos.values)),
        n_pregoes=len(retornos),
    )


@dataclass(frozen=True)
class ComparacaoBenchmark:
    alfa_anual: float
    beta: float
    correlacao: float
    tracking_error: float
    information_ratio: float
    t_stat_alfa: float


def comparar_benchmark(valores: pd.Series, valores_benchmark: pd.Series,
                       rf_diaria: pd.Series) -> ComparacaoBenchmark:
    """
    CAPM por MQO nos excessos diários:  (r_p - rf) = a + b (r_b - rf) + e.
    Alfa anualizado = 252·a. A estatística t mostra se o alfa é distinguível de
    zero; com 3 a 5 anos de dados quase nunca é.
    """
    comum = valores.index.intersection(valores_benchmark.index)
    if len(comum) < 60:
        raise ValueError("Menos de 60 pregões em comum com o benchmark.")
    r_p = retornos_de_valores(valores.loc[comum])
    r_b = retornos_de_valores(valores_benchmark.loc[comum])
    rf = rf_diaria.reindex(r_p.index).fillna(0.0)
    regressao = stats.linregress((r_b - rf).values, (r_p - rf).values)
    ativo = r_p - r_b
    tracking = float(ativo.std() * _RAIZ_ANO)
    t_alfa = (regressao.intercept / regressao.intercept_stderr
              if regressao.intercept_stderr > 0 else float("nan"))
    return ComparacaoBenchmark(
        alfa_anual=float(regressao.intercept * DIAS_UTEIS_ANO),
        beta=float(regressao.slope),
        correlacao=float(regressao.rvalue),
        tracking_error=tracking,
        information_ratio=float(ativo.mean() * DIAS_UTEIS_ANO / tracking) if tracking > 0 else float("nan"),
        t_stat_alfa=float(t_alfa),
    )


def contribuicao_risco(pesos: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """RC_i = w_i (Sigma w)_i / (w' Sigma w); soma 1."""
    ativos = pesos[pesos > 0].index
    w = pesos[ativos].values / pesos[ativos].sum()
    sigma_w = cov.loc[ativos, ativos].values @ w
    variancia = float(w @ sigma_w)
    if variancia <= 0:
        return pd.Series(np.nan, index=ativos)
    return pd.Series(w * sigma_w / variancia, index=ativos).sort_values(ascending=False)


def metricas_rolantes(retornos: pd.Series, rf_diaria: pd.Series, janela: int = 63) -> pd.DataFrame:
    excesso = retornos - rf_diaria.reindex(retornos.index).fillna(0.0)
    vol = retornos.rolling(janela).std() * _RAIZ_ANO
    sharpe_rolante = excesso.rolling(janela).mean() / excesso.rolling(janela).std() * _RAIZ_ANO
    return pd.DataFrame({"vol_rolling": vol, "sharpe_rolling": sharpe_rolante}).dropna()
