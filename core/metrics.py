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
