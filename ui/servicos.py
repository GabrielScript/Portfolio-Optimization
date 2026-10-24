"""
servicos.py - Fachada com cache entre a UI e o core.

Regra de cache: TODO argumento que muda o resultado entra na chave. A versão
anterior passava os preços como ``_precos``, e o Streamlit ignora argumentos
com prefixo "_" no hash. Mudar o universo não mudava a chave, e o
backtest antigo continuava sendo exibido. Aqui as funções recebem só tipos
hasheáveis (tuplas, números, DataFrames) e montam os objetos do core dentro.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from core import bcb, cdi_lago, focus, lago, market_data, precos_do_app, reamostragem, sentimento
from core.backtest import (ResultadoBacktest, estrategia_equiponderada, estrategia_hrp, estrategia_nucleo_satelite,
                           estrategia_otimizada, estrategia_reamostrada, walk_forward)
from core.config import ConfigBacktest, ConfigOtimizacao, LimiteGrupo, Objetivo, limites_por_setor


@dataclass(frozen=True)
class ParametrosOtimizacao:
    """
    Parâmetros vindos da sidebar, todos hasheáveis. ``setores``: pares (ticker, setor) em tupla
    ordenada, e não ``LimiteGrupo``, para que a chave do cache não dependa da ordem de um frozenset.
    """

    objetivo: str
    peso_maximo: float
    n_ativos_max: int
    vol_maxima: float | None
    aversao_risco: float
    teto_setor: float = 1.0
    setores: tuple[tuple[str, str], ...] = ()
    fracao_satelite: float | None = None      # núcleo Mín. Vol + satélite com este objetivo; None = só o objetivo

    @property
    def grupos(self) -> tuple[LimiteGrupo, ...]:
        return limites_por_setor(dict(self.setores), self.teto_setor)

    def para_config(self, taxa_livre_risco: float) -> ConfigOtimizacao:
        return ConfigOtimizacao(
            objetivo=Objetivo(self.objetivo), peso_maximo=self.peso_maximo,
            n_ativos_max=self.n_ativos_max, vol_maxima=self.vol_maxima,
            aversao_risco=self.aversao_risco, taxa_livre_risco=taxa_livre_risco, grupos=self.grupos,
        )


ARQUIVO_PAPER = Path(__file__).resolve().parents[1] / "paper" / "carteiras.csv"
ARQUIVO_PAPER_V2 = Path(__file__).resolve().parents[1] / "paper-v2" / "carteiras.csv"
PASTA_MARCOS_V2 = Path(__file__).resolve().parents[1] / "paper-v2" / "marcos"
ARQUIVO_LIQUIDEZ = Path(__file__).resolve().parents[1] / "lago" / "liquidez.parquet"
# the market panel (scripts/painel_do_app.py, refreshed by the daily upkeep) and the lake's log, local only
ARQUIVOS_DO_PAINEL = tuple(Path(__file__).resolve().parents[1] / "lago" / nome for nome in (
    "sentimento_diario.parquet", "fatos_relevantes.parquet", "fundamentos.parquet"))
PASTA_DO_LOG = Path("D:/b3-datalake/sentiment")
INDICADORES_DO_FOCUS = ("Selic", "IPCA", "PIB Total", "Câmbio")
# the frozen rerun's daily curves (scripts/curvas_conservadora.py): the Conservadora's results by period and year
ARQUIVO_CURVAS = Path(__file__).resolve().parents[1] / "docs" / "dados" / "conservadora-curvas-2026-10-23.parquet"


def carregar_satelite_registrado(arquivo: Path = ARQUIVO_PAPER) -> tuple[date, pd.Series] | None:
    """O último satélite registrado no paper trading (``paper/carteiras.csv``): data e pesos com ".SA". None sem
    registro (arquivo ausente ou só com o cabeçalho)."""
    return _ultimo_registro(arquivo)


def carregar_conservadora_registrada(arquivo: Path = ARQUIVO_PAPER_V2) -> tuple[date, pd.Series] | None:
    """The last Conservadora registered in paper-v2 (``paper-v2/carteiras.csv``): date and weights with ".SA"."""
    return _ultimo_registro(arquivo)


def _ultimo_registro(arquivo: Path) -> tuple[date, pd.Series] | None:
    if not arquivo.exists():
        return None
    registro = pd.read_csv(arquivo, parse_dates=["data"])
    if registro.empty:
        return None
    ultimo = registro[registro["data"] == registro["data"].max()]
    return ultimo["data"].iloc[0].date(), ultimo.set_index("ticker")["peso"].rename(lambda papel: papel + ".SA")


def setores_do_lago(tickers: tuple[str, ...], pasta: Path = PASTA_MARCOS_V2) -> dict[str, str]:
    """Ticker -> the lake's sector (by issuer, the ticker's first 4 letters) in the latest paper-v2 milestone: the
    sectors the registered decision used. A ticker without a sector is left out (``checar_carteira``: no sector)."""
    marcos = sorted(caminho for caminho in pasta.iterdir() if (caminho / "sectors.parquet").exists())
    setores = lago.ler_setores(marcos[-1] / "sectors.parquet")
    return {ticker: setores[ticker.removesuffix(".SA")[:4]] for ticker in tickers
            if ticker.removesuffix(".SA")[:4] in setores.index}


@st.cache_data(show_spinner=False, max_entries=2)
def carregar_curvas_conservadora(arquivo: Path = ARQUIVO_CURVAS) -> pd.DataFrame | None:
    """The Conservadora's and BOVA11's daily values in the 3 phases of the frozen rerun, or None without the file."""
    return pd.read_parquet(arquivo) if arquivo.exists() else None


def carregar_liquidez(arquivo: Path = ARQUIVO_LIQUIDEZ) -> pd.DataFrame | None:
    """The app's liquidity snapshot (``scripts/liquidez_do_app.py``), or None without the file."""
    return pd.read_parquet(arquivo) if arquivo.exists() else None


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_precos(tickers: tuple[str, ...], anos: int, dia: date, versao_dados: int) -> market_data.DadosPrecos:
    """``dia`` e ``versao_dados`` (``precos_do_app.versao_dos_dados()``) entram na chave: o cache vira com o dia e a
    cada publicação do lago ou dos pregões recentes."""
    return precos_do_app.carregar_precos(list(tickers), anos, hoje=dia)


@st.cache_data(ttl=86400, show_spinner=False)
def carregar_cdi(anos: int, dia: date) -> pd.Series:
    """The lake's CDI copy, topped up from the BCB (``cdi_lago``); without the file (Cloud), the BCB as before."""
    try:
        return cdi_lago.daily_cdi(anos, dia)
    except market_data.ErroDadosMercado:
        return pd.Series(dtype=float)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def carregar_selic(dia: date) -> market_data.TaxaSelic:
    return market_data.obter_selic_meta(hoje=dia)


@st.cache_data(ttl=5 * 60, show_spinner=False, max_entries=16)
def carregar_cotacoes(tickers: tuple[str, ...], precos_recentes: pd.DataFrame) -> pd.DataFrame:
    """Último negócio do dia (Yahoo, ~15 min de atraso); sem ele, o último fechamento de ``precos_recentes``."""
    return market_data.cotacoes_atuais(list(tickers), precos_recentes)


def descartar_cotacoes(tickers: tuple[str, ...], precos_recentes: pd.DataFrame) -> None:
    """
    Apaga só a entrada desta carteira (botão "Atualizar cotações"). O cache é do
    processo, compartilhado por todos os usuários do app: ``clear()`` sem
    argumentos apagaria também as cotações dos outros.
    """
    carregar_cotacoes.clear(tickers, precos_recentes)


@st.cache_data(show_spinner=False, max_entries=8)
def otimizar_reamostrado(retornos: pd.DataFrame, params: ParametrosOtimizacao, taxa_livre_risco: float,
                         shrink_mu: bool) -> reamostragem.ResultadoReamostragem:
    """Carteira de hoje reamostrada."""
    return reamostragem.otimizar_reamostrado(retornos, params.para_config(taxa_livre_risco), shrink_mu=shrink_mu)


@st.cache_data(show_spinner=False, max_entries=8)
def rodar_walk_forward(retornos: pd.DataFrame, params: ParametrosOtimizacao, cfg_bt: ConfigBacktest,
                       cdi_diario: pd.Series, taxa_constante_aritmetica: float,
                       reamostrar: bool) -> dict[str, ResultadoBacktest]:
    """
    Todas as estratégias no MESMO protocolo: mesmo universo,
    mesmas janelas e mesmos custos (``cfg_bt``, o mesmo do benchmark). rf point-in-time pelo CDI.

    A PRIMEIRA estratégia do dicionário é a principal da tela (o app usa
    ``next(iter(...))``): a reamostrada, quando ligada; senão, a otimizada.
    """
    shrink_mu = cfg_bt.shrink_mu
    cfg_opt = params.para_config(taxa_constante_aritmetica)
    rf = cdi_diario if not cdi_diario.empty else None
    estrategias = {}
    if reamostrar:
        estrategias[f"Reamostrada ({params.objetivo})"] = estrategia_reamostrada(cfg_opt, shrink_mu, rf)
    if params.fracao_satelite is None:
        principal = {f"Otimizada ({params.objetivo})": estrategia_otimizada(cfg_opt, shrink_mu, rf)}
    else:
        nucleo = estrategia_otimizada(replace(cfg_opt, objetivo=Objetivo.MIN_VOL), shrink_mu, rf)
        satelite = estrategia_otimizada(cfg_opt, shrink_mu, rf)
        principal = {f"Otimizada (núcleo Mín. Vol + satélite Kelly {params.fracao_satelite:.0%})":
                     estrategia_nucleo_satelite(nucleo, satelite, params.fracao_satelite)}
    estrategias.update(principal)
    estrategias.update({
        "1/N": estrategia_equiponderada(),
        "HRP": estrategia_hrp(arvore=False),
        "HRP-Árvore": estrategia_hrp(arvore=True),
    })
    return {nome: walk_forward(retornos, estrategia, cfg_bt, nome)
            for nome, estrategia in estrategias.items()}


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_painel() -> tuple[pd.DataFrame | None, ...]:
    """The panel files: daily news tone, recent CVM filings and per-stock fundamentals (None when one is missing)."""
    return tuple(pd.read_parquet(arquivo) if arquivo.exists() else None for arquivo in ARQUIVOS_DO_PAINEL)


@st.cache_data(ttl=600, show_spinner=False)
def carregar_manchetes() -> pd.DataFrame | None:
    """The last 3 days' relevant headlines from the lake's log on D: (title and link only); None where the lake's
    disk is not reachable (the deployed app)."""
    if not PASTA_DO_LOG.exists():
        return None
    itens, notas = sentimento.ler_log(PASTA_DO_LOG, date.today() - timedelta(days=3))
    return sentimento.manchetes(itens, notas, quantas=60)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def carregar_focus(dia: date) -> dict[str, pd.DataFrame] | None:
    """The Focus survey's annual indicators, live from the BCB (``core.focus``); None when the API fails."""
    try:
        return {nome: focus.baixar_focus_anual(nome, dia) for nome in INDICADORES_DO_FOCUS}
    except focus.ErroFocus:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def carregar_indicadores_bcb(dia: date) -> dict[str, tuple[pd.Timestamp, float]] | None:
    """The last IPCA 12 months (SGS 13522, %) and PTAX (SGS 1, R$/US$), live from the BCB; None when it fails."""
    try:
        ipca = bcb.baixar_serie(bcb.SERIE_IPCA_12M, dia - timedelta(days=400), dia).dropna()
        ptax = bcb.baixar_ptax(dia - timedelta(days=30), dia).dropna()
    except bcb.ErroBcb:
        return None
    return {"IPCA 12 meses (%)": (ipca.index[-1], float(ipca.iloc[-1])),
            "Dólar PTAX (R$)": (ptax.index[-1], float(ptax.iloc[-1]))}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def carregar_precos_da_tabela(tickers: tuple[str, ...], dia: date, versao_dados: int) -> pd.DataFrame:
    """3 years of daily closes of ``tickers``, raw (no coverage filter: the table shows every stock)."""
    return precos_do_app.ler_fechamentos(list(tickers), dia - timedelta(days=3 * 365 + 10), dia)
