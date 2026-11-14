"""
market_data.py - Aquisição e limpeza de dados de mercado (sem UI).

Fontes:
- Preços diários (carteira de hoje e walk-forward): lago de dados da B3
  (``core.lago``), retorno total a partir do COTAHIST e dos eventos da B3/CVM,
  com deslistadas, cisões e JCP líquido. Pregão sem negócio do papel vem vazio.
- Preços mensais (painel macro, em desenvolvimento): Yahoo Finance via yfinance
  (``auto_adjust=True`` -> retorno total conforme o ajuste do Yahoo).
- CDI diário (SGS 12) e Selic meta (SGS 432): BCB, via ``core.bcb``.
- Cotação do dia (plano de compra): último negócio nas barras de 1 min do
  Yahoo, com atraso de ~15 min; sem ele, o último fechamento diário.

Calendário: a grade de datas é o calendário de pregões da B3, inferido dos
próprios tickers ``.SA``. Datas fora dela (feriados com dado espúrio) saem,
para não criar retornos zero falsos que subestimariam a volatilidade e
quebrariam a anualização por 252.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from core import bcb, lago

logger = logging.getLogger(__name__)

SELIC_FALLBACK = 0.1375          # Selic meta lida na SGS 432 em 2026-10-12; usada só se o BCB falhar
SELIC_FALLBACK_DATA = "2026-10"


class ErroDadosMercado(RuntimeError):
    """Falha ao obter dados de mercado de uma fonte externa."""


@dataclass(frozen=True)
class TaxaSelic:
    valor: float
    fonte: str          # "BCB SGS 432" ou "fallback 2026-03"


@dataclass
class DadosPrecos:
    """Preços limpos e alinhados ao calendário B3, mais o relatório de limpeza."""

    precos: pd.DataFrame
    descartados_sem_dados: list[str] = field(default_factory=list)
    descartados_cobertura: list[str] = field(default_factory=list)


def eh_ticker_b3(ticker: str) -> bool:
    return ticker.upper().endswith(".SA")


def _extrair_fechamentos(bruto: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normaliza a saída do yfinance (MultiIndex ou não) para DataFrame ticker x data."""
    if bruto is None or bruto.empty:
        return pd.DataFrame()
    if isinstance(bruto.columns, pd.MultiIndex):
        fechamentos = bruto["Close"]
    else:
        fechamentos = bruto[["Close"]].rename(columns={"Close": tickers[0]})
    fechamentos = fechamentos.reindex(columns=tickers)
    if fechamentos.index.tz is not None:
        fechamentos.index = fechamentos.index.tz_localize(None)
    return fechamentos.sort_index()


def baixar_fechamentos(tickers: list[str], inicio: date, fim: date) -> pd.DataFrame:
    """Download bruto de fechamentos ajustados. Lança ErroDadosMercado se vier vazio."""
    if not tickers:
        raise ValueError("Lista de tickers vazia.")
    bruto = yf.download(
        tickers,
        start=inicio.isoformat(),
        end=(fim + timedelta(days=1)).isoformat(),  # 'end' do yfinance é exclusivo
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    fechamentos = _extrair_fechamentos(bruto, tickers)
    if fechamentos.dropna(how="all").empty:
        raise ErroDadosMercado(f"Yahoo não retornou preços para {len(tickers)} tickers.")
    return fechamentos


def calendario_b3(fechamentos: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Pregões da B3: datas em que pelo menos metade dos tickers .SA tem preço.

    Sem tickers .SA (só índices, ex.: ^BVSP), usa dias úteis seg-sex.
    """
    colunas_b3 = [c for c in fechamentos.columns if eh_ticker_b3(c)]
    if not colunas_b3:
        return fechamentos.index[fechamentos.index.dayofweek < 5]
    cobertura = fechamentos[colunas_b3].notna().mean(axis=1)
    return fechamentos.index[cobertura >= 0.5]


def limpar_precos(
    fechamentos: pd.DataFrame,
    cobertura_minima: float = 0.95,
    max_dias_ffill: int = 5,
) -> DadosPrecos:
    """
    Alinha ao calendário B3, descarta ativos sem dados ou com cobertura baixa e
    preenche lacunas curtas.

    Por que ``ffill(limit=5)`` e não ffill ilimitado: um ativo suspenso ou deslistado
    teria o último preço repetido por meses, o que gera retornos zero, derruba a
    volatilidade estimada e atrai o otimizador de mínima variância. Lacunas
    maiores ficam NaN, e a amostra comum (``retornos_simples``) as exclui.
    """
    sem_dados = [c for c in fechamentos.columns if fechamentos[c].notna().sum() == 0]
    precos = fechamentos.drop(columns=sem_dados)

    # limite do ffill contado em pregões B3, não em dias corridos
    precos = precos.reindex(calendario_b3(precos)).ffill(limit=max_dias_ffill)

    cobertura = precos.notna().mean()
    baixa_cobertura = cobertura[cobertura < cobertura_minima].index.tolist()
    precos = precos.drop(columns=baixa_cobertura)

    if sem_dados:
        logger.warning("Sem dados no Yahoo (deslistados/renomeados?): %s", sem_dados)
    if baixa_cobertura:
        logger.info("Cobertura < %.0f%%, descartados: %s", cobertura_minima * 100, baixa_cobertura)

    return DadosPrecos(
        precos=precos,
        descartados_sem_dados=sem_dados,
        descartados_cobertura=baixa_cobertura,
    )


def carregar_precos(
    tickers: list[str],
    anos: int,
    hoje: date | None = None,
    cobertura_minima: float = 0.95,
) -> DadosPrecos:
    """Pipeline completo: preços de retorno total do lago da B3 e limpeza no calendário B3."""
    fim = hoje or date.today()
    inicio = fim - timedelta(days=int(anos * 365.25) + 10)
    try:
        fechamentos = lago.ler_fechamentos(list(dict.fromkeys(tickers)), inicio, fim)
    except FileNotFoundError as erro:
        raise ErroDadosMercado(f"Arquivo de preços do lago ausente: {erro.filename}") from erro
    return limpar_precos(fechamentos, cobertura_minima=cobertura_minima)


def carregar_precos_mensais(tickers: list[str], inicio: date, hoje: date) -> pd.DataFrame:
    """
    Último fechamento ajustado de cada mês COMPLETO (meses anteriores ao de ``hoje``).

    Sem filtro de cobertura e sem ffill, ao contrário de ``carregar_precos``: o
    modelo de fatores usa até 180 meses e precisa também das ações com
    histórico curto. Cada regressão usa só os meses em que a ação tem preço.
    Índice: PeriodIndex mensal ``mes``.
    """
    fechamentos = baixar_fechamentos(tickers, inicio, hoje)
    mensais = fechamentos.groupby(fechamentos.index.to_period("M")).last().rename_axis("mes")
    return mensais[mensais.index < pd.Period(hoje, freq="M")]


FONTE_INTRADIARIA = "Yahoo 1 min (atraso ~15 min)"
FONTE_FECHAMENTO = "último fechamento"


def baixar_ultimos_negocios(tickers: list[str]) -> pd.DataFrame:
    """Último negócio do pregão mais recente (barras de 1 min do Yahoo). Colunas: preco, horario."""
    bruto = yf.download(tickers, period="1d", interval="1m", progress=False, auto_adjust=False,
                        threads=True)
    barras = _extrair_fechamentos(bruto, tickers)
    linhas = {}
    for ticker in barras.columns:
        precos = barras[ticker]
        negocios = precos[precos > 0]          # o Yahoo às vezes devolve barra com preço 0; NaN também sai aqui
        if not negocios.empty:
            linhas[ticker] = (float(negocios.iloc[-1]), negocios.index[-1])
    return pd.DataFrame.from_dict(linhas, orient="index", columns=["preco", "horario"])


def cotacoes_atuais(tickers: list[str], precos_diarios: pd.DataFrame) -> pd.DataFrame:
    """
    Cotação de cada ticker para dimensionar a compra: o último negócio do dia no
    Yahoo (atraso de ~15 min) ou, se o ticker não negociou ou o Yahoo falhou, o
    último fechamento válido de ``precos_diarios``. O fechamento ajustado mais
    recente é o fechamento de fato: o ajuste por proventos só reescala as datas
    anteriores a eles.

    Colunas: preco (NaN se nenhuma fonte tem o ticker), horario, fonte.
    """
    try:
        intradiario = baixar_ultimos_negocios(tickers)
    except Exception as erro:  # noqa: BLE001 - qualquer falha do Yahoo cai no fechamento, com aviso
        logger.warning("Cotações do dia indisponíveis (%s); usando o último fechamento.", erro)
        intradiario = pd.DataFrame(columns=["preco", "horario"])
    linhas = {}
    for ticker in tickers:
        if ticker in intradiario.index:
            linhas[ticker] = (intradiario.at[ticker, "preco"], intradiario.at[ticker, "horario"],
                              FONTE_INTRADIARIA)
        else:
            linhas[ticker] = _ultimo_fechamento(precos_diarios, ticker)
    return pd.DataFrame.from_dict(linhas, orient="index", columns=["preco", "horario", "fonte"])


def _ultimo_fechamento(precos_diarios: pd.DataFrame, ticker: str) -> tuple[float, pd.Timestamp, str]:
    fechamentos = (precos_diarios[ticker].dropna() if ticker in precos_diarios.columns
                   else pd.Series(dtype=float))
    if fechamentos.empty:
        return float("nan"), pd.NaT, FONTE_FECHAMENTO
    return float(fechamentos.iloc[-1]), fechamentos.index[-1], FONTE_FECHAMENTO


def baixar_cdi_diario(anos: int, hoje: date | None = None) -> pd.Series:
    """CDI diário (decimal ao dia) via SGS 12. Mais de 10 anos é quebrado em décadas pelo ``core.bcb``."""
    try:
        return bcb.baixar_cdi_diario(anos, hoje or date.today())
    except bcb.ErroBcb as erro:
        raise ErroDadosMercado(str(erro)) from erro


def obter_selic_meta(hoje: date | None = None) -> TaxaSelic:
    """
    Selic meta vigente em ``hoje`` (SGS 432), ou o fallback se o BCB falhar.

    Lazy de propósito: buscar na importação congelaria a taxa pelo tempo de
    vida do processo (num container Cloud Run, dias) e faria rede na importação.
    O cache (6 h na década corrente) fica no ``core.bcb``.
    """
    try:
        return TaxaSelic(bcb.obter_selic_meta(hoje or date.today()), "BCB SGS 432")
    except bcb.ErroBcb as erro:
        logger.warning("Selic indisponível (%s); usando fallback %s.", erro, SELIC_FALLBACK_DATA)
        return TaxaSelic(SELIC_FALLBACK, f"fallback {SELIC_FALLBACK_DATA}")


def serie_rf_diaria(indice: pd.DatetimeIndex, cdi_diario: pd.Series | None,
                    taxa_anual_constante: float) -> pd.Series:
    """
    Taxa livre de risco diária alinhada a ``indice``. Usa o CDI histórico
    quando disponível (forward-fill em feriados). Fora da cobertura, ou sem
    CDI, usa a taxa constante convertida para diária.
    """
    diaria_constante = (1.0 + taxa_anual_constante) ** (1.0 / 252) - 1.0
    if cdi_diario is None or cdi_diario.empty:
        return pd.Series(diaria_constante, index=indice)
    alinhado = cdi_diario.reindex(indice.union(cdi_diario.index)).ffill().reindex(indice)
    return alinhado.fillna(diaria_constante)


def data_hoje_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")
