"""
market_data.py - Aquisição e limpeza de dados de mercado (sem UI).

Fontes:
- Preços: Yahoo Finance via yfinance (``auto_adjust=True`` -> retorno total,
  dividendos/JCP reinvestidos conforme o ajuste do Yahoo).
- CDI diário (SGS 12), Selic meta (SGS 432) e PTAX (SGS 1): BCB, via ``core.bcb``.
- Câmbio: PTAX de venda; se o BCB falhar, USDBRL=X do Yahoo.

Calendário: a grade de datas é o calendário de pregões da B3, inferido dos
próprios tickers ``.SA``. Ativos que negociam em outros dias (cripto 24/7, EUA)
são reamostrados para essa grade. Movimentos de fim de semana caem no retorno
da segunda-feira em vez de criar retornos zero falsos nas ações, que
subestimariam a volatilidade e quebrariam a anualização por 252.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from core import bcb

logger = logging.getLogger(__name__)

SELIC_FALLBACK = 0.1375          # Selic meta lida na SGS 432 em 2026-10-12; usada só se o BCB falhar
SELIC_FALLBACK_DATA = "2026-10"
FONTE_PTAX = "BCB PTAX venda (SGS 1)"
FONTE_YAHOO_CAMBIO = "Yahoo USDBRL=X"
_FOLGA_CAMBIO_DIAS = 15          # PTAX começa antes dos preços: o 1º preço precisa de câmbio


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
    fonte_cambio: str | None = None     # None quando nenhum ativo é cotado em USD


def eh_ticker_b3(ticker: str) -> bool:
    return ticker.upper().endswith(".SA")


def eh_cotado_em_usd(ticker: str) -> bool:
    """Ativos sem sufixo .SA e que não são índices (^) são cotados em USD no Yahoo."""
    t = ticker.upper()
    return not t.endswith(".SA") and not t.startswith("^") and not t.endswith("=X")


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

    Sem tickers .SA (universo só cripto/EUA), usa dias úteis seg-sex.
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

    grade = calendario_b3(precos)
    colunas_b3 = [c for c in precos.columns if eh_ticker_b3(c)]
    colunas_externas = [c for c in precos.columns if c not in colunas_b3]
    partes = []
    if colunas_b3:
        # limite do ffill contado em pregões B3, não em dias corridos
        partes.append(precos[colunas_b3].reindex(grade).ffill(limit=max_dias_ffill))
    if colunas_externas:
        # cripto/EUA: preenche no calendário próprio antes de reamostrar, para
        # que o preço da segunda incorpore o fim de semana
        partes.append(precos[colunas_externas].ffill(limit=max_dias_ffill).reindex(grade))
    precos = pd.concat(partes, axis=1)[[c for c in precos.columns]]

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


def converter_usd_para_brl(precos: pd.DataFrame, cambio_usdbrl: pd.Series) -> pd.DataFrame:
    """
    Converte colunas cotadas em USD para BRL. Lança erro se o câmbio faltar.

    Cada data usa a última cotação conhecida até ela, inclusive cotações em
    datas fora da grade de preços (a PTAX de sexta vale para o BTC de sábado).
    """
    colunas_usd = [c for c in precos.columns if eh_cotado_em_usd(c)]
    if not colunas_usd:
        return precos
    if cambio_usdbrl.empty:
        raise ErroDadosMercado(f"Câmbio USDBRL indisponível; não é possível converter {colunas_usd}.")
    fx = cambio_usdbrl.reindex(cambio_usdbrl.index.union(precos.index)).ffill().reindex(precos.index)
    convertido = precos.copy()
    convertido[colunas_usd] = convertido[colunas_usd].mul(fx, axis=0)
    return convertido


def _ptax_ou_none(inicio: date, fim: date) -> pd.Series | None:
    try:
        return bcb.baixar_ptax(inicio - timedelta(days=_FOLGA_CAMBIO_DIAS), fim)
    except bcb.ErroBcb as erro:
        logger.warning("PTAX indisponível (%s); usando %s.", erro, FONTE_YAHOO_CAMBIO)
        return None


def carregar_precos(
    tickers: list[str],
    anos: int,
    hoje: date | None = None,
    cobertura_minima: float = 0.95,
) -> DadosPrecos:
    """
    Pipeline completo: download, conversão USD->BRL e limpeza no calendário B3.

    O câmbio vem da PTAX (fonte oficial). O Yahoo ``USDBRL=X`` só entra se o
    BCB falhar.
    """
    fim = hoje or date.today()
    inicio = fim - timedelta(days=int(anos * 365.25) + 10)

    precisa_cambio = any(eh_cotado_em_usd(t) for t in tickers)
    cambio = _ptax_ou_none(inicio, fim) if precisa_cambio else None
    fonte_cambio = None
    if precisa_cambio:
        fonte_cambio = FONTE_PTAX if cambio is not None else FONTE_YAHOO_CAMBIO
    usar_yahoo = precisa_cambio and cambio is None
    pedidos = list(dict.fromkeys(tickers + (["USDBRL=X"] if usar_yahoo else [])))
    fechamentos = baixar_fechamentos(pedidos, inicio, fim)

    if precisa_cambio:
        if usar_yahoo:
            cambio = fechamentos.pop("USDBRL=X").dropna()
        fechamentos = converter_usd_para_brl(fechamentos, cambio)

    dados = limpar_precos(fechamentos, cobertura_minima=cobertura_minima)
    return replace(dados, fonte_cambio=fonte_cambio)


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
