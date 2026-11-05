"""
servicos.py - Fachada com cache entre a UI e o core.

Regra de cache: TODO argumento que muda o resultado entra na chave. A versão
anterior passava os preços como ``_precos``, e o Streamlit ignora argumentos
com prefixo "_" no hash. Mudar o universo não mudava a chave, e o
backtest antigo continuava sendo exibido. Aqui as funções recebem só tipos
hasheáveis (tuplas, números, DataFrames) e montam os objetos do core dentro.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

import ativos_tela
from core import (assessor, bcb, cdi_lago, cotacao_b3, focus, indicadores, lago, macro_bcb, manchetes_online,
                  market_data, precos_do_app, reamostragem, sentimento, simulador)
from core.backtest import (ResultadoBacktest, estrategia_equiponderada, estrategia_hrp, estrategia_nucleo_satelite,
                           estrategia_otimizada, estrategia_reamostrada, walk_forward)
from core.config import ConfigBacktest, ConfigOtimizacao, LimiteGrupo, Objetivo, limites_por_setor
from core.periodos import anos_em_reais
from ui.eua import ResultadoEUA


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
ARQUIVO_PAPER_V3 = Path(__file__).resolve().parents[1] / "paper-v3" / "composto.csv"
PASTA_MARCOS_V3 = Path(__file__).resolve().parents[1] / "paper-v3" / "marcos"
ARQUIVO_LIQUIDEZ = Path(__file__).resolve().parents[1] / "lago" / "liquidez.parquet"
# the market panel (scripts/painel_do_app.py, refreshed by the daily upkeep) and the lake's log, local only
ARQUIVOS_DO_PAINEL = tuple(Path(__file__).resolve().parents[1] / "lago" / nome for nome in (
    "sentimento_diario.parquet", "fatos_relevantes.parquet", "fundamentos.parquet"))
PASTA_DO_LOG = Path("D:/b3-datalake/sentiment")
INDICADORES_DO_FOCUS = ("Selic", "IPCA", "PIB Total", "Câmbio")
# the live rerun's daily curves (scripts/curvas_ao_vivo.py, DECISOES 188): results by period and year
ARQUIVO_CURVAS = Path(__file__).resolve().parents[1] / "docs" / "dados" / "curvas-ao-vivo.parquet"
# the CDI and the Selic next to the curves (scripts/juros_diarios.py)
ARQUIVO_JUROS = ARQUIVO_CURVAS.with_name("juros-diarios.parquet")
# the fictitious account's ledger (core/simulador.py), written by the daily upkeep
ARQUIVO_SIMULADOR = Path(__file__).resolve().parents[1] / "simulador" / "ordens.csv"
# the derived results of the US test (scripts/us_avaliar.py)
PASTA_EUA = Path(__file__).resolve().parents[1] / "docs" / "dados"
# the DeepSeek portfolio's opinions (scripts/carteira_deepseek.py, DECISOES 240), one JSON per opinion
PASTA_PARECERES_DEEPSEEK = Path(__file__).resolve().parents[1] / "deepseek" / "pareceres"


def carregar_parecer_deepseek(pasta: Path = PASTA_PARECERES_DEEPSEEK) -> dict | None:
    """The newest DeepSeek opinion on the Composite; None before the first one or when the file cannot be read."""
    arquivos = sorted(pasta.glob("*.json")) if pasta.exists() else []
    if not arquivos:
        return None
    try:
        return json.loads(arquivos[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def carregar_satelite_registrado(arquivo: Path = ARQUIVO_PAPER) -> tuple[date, pd.Series] | None:
    """O último satélite registrado no paper trading (``paper/carteiras.csv``): data e pesos com ".SA". None sem
    registro (arquivo ausente ou só com o cabeçalho)."""
    return _ultimo_registro(arquivo)


def carregar_conservadora_registrada(arquivo: Path = ARQUIVO_PAPER_V2) -> tuple[date, pd.Series] | None:
    """The last Conservadora registered in paper-v2 (``paper-v2/carteiras.csv``): date and weights with ".SA"."""
    return _ultimo_registro(arquivo)


def carregar_plano_registrado(arquivo: Path = ARQUIVO_PAPER_V3) -> tuple[date, pd.Series] | None:
    """The real-money plan: the last Composite registered in paper-v3 (``paper-v3/composto.csv``), date and weights
    with ".SA" (the user's decision of 2026-11-03, DECISOES 242; the Conservadora of paper-v2 before)."""
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


def carregar_eua(pasta: Path = PASTA_EUA) -> ResultadoEUA | None:
    """The derived results of the US test, or None before it has run."""
    if not (pasta / "us-cobertura.json").exists():
        return None
    curvas = pd.read_parquet(pasta / "us-curvas.parquet")
    curvas = curvas.set_index(pd.DatetimeIndex(pd.to_datetime(curvas.pop("date"))))
    def opcional(nome: str) -> pd.DataFrame | None:
        return pd.read_parquet(pasta / nome) if (pasta / nome).exists() else None
    return ResultadoEUA(curvas, pd.read_parquet(pasta / "us-resumo.parquet"), pd.read_parquet(pasta / "us-portao.parquet"),
                        pd.read_parquet(pasta / "us-carteiras.parquet"),
                        json.loads((pasta / "us-cobertura.json").read_text(encoding="utf-8")),
                        opcional("us-indicadores.parquet"), opcional("us-macro.parquet"),
                        _anos_dos_eua(curvas, opcional(ARQUIVO_CURVAS.name), opcional(ARQUIVO_JUROS.name)),
                        opcional("us-hrp-desde-2026.parquet"))


def _anos_dos_eua(curvas: pd.DataFrame, b3: pd.DataFrame | None, juros: pd.DataFrame | None) -> pd.DataFrame | None:
    """The US strategies by year next to BOVA11 (the live curves' first phase, the one the US curves use), the CDI and
    the Selic, or None without those two files."""
    if b3 is None or juros is None:
        return None
    fase = b3[b3["fase"] == b3["fase"].min()].set_index("date")
    estrategias = curvas[[nome for nome in ("Composto", "Conservadora") if nome in curvas.columns]].rename(
        columns=lambda nome: f"{nome} EUA")
    return anos_em_reais(estrategias, fase["BOVA11"], juros.set_index("date"))


@st.cache_data(show_spinner=False, max_entries=2)
def carregar_curvas_conservadora(arquivo: Path = ARQUIVO_CURVAS) -> pd.DataFrame | None:
    """The live rerun's daily values (3 phases: the Conservadora, the composite, BOVA11, BITH11, QBTC11 and the
    CDI), or None without the file."""
    return pd.read_parquet(arquivo) if arquivo.exists() else None


@st.cache_data(show_spinner=False, max_entries=1)
def carregar_juros(arquivo: Path = ARQUIVO_JUROS) -> pd.DataFrame | None:
    """The CDI and Selic of each published day, decimal per day and indexed by date, or None without the file."""
    return pd.read_parquet(arquivo).set_index("date") if arquivo.exists() else None


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


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_simulador(dia: date, versao_dados: int, arquivo: Path = ARQUIVO_SIMULADOR) -> simulador.Painel | None:
    """The fictitious account on the app's price files; None before its first buy."""
    ordens = simulador.ler_ordens(arquivo)
    if ordens.empty:
        return None
    papeis = [*ordens["ticker"].unique(), *simulador.REFERENCIAS]
    precos = precos_do_app.ler_fechamentos([f"{papel}.SA" for papel in papeis], ordens["data"].min().date(), dia)
    return simulador.painel(ordens, precos.rename(columns=lambda papel: papel.removesuffix(".SA")),
                            carregar_selic_diaria(dia.year - simulador.INICIO.year + 1, dia))


@st.cache_data(ttl=86400, show_spinner=False)
def carregar_selic_diaria(anos: int, dia: date) -> pd.Series:
    """The daily Selic from the BCB (``cdi_lago.daily_selic``): the simulator's cash rate; empty when it fails."""
    try:
        return cdi_lago.daily_selic(anos, dia)
    except bcb.ErroBcb:
        return pd.Series(dtype=float)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def carregar_selic(dia: date) -> market_data.TaxaSelic:
    return market_data.obter_selic_meta(hoje=dia)


@st.cache_data(ttl=5 * 60, show_spinner=False, max_entries=16)
def carregar_cotacoes(tickers: tuple[str, ...], precos_recentes: pd.DataFrame) -> pd.DataFrame:
    """The session's last trade from B3's own quote service (``core.cotacao_b3``, ~15 min late; Yahoo is no source
    since 2026-11-04); without it, or once the lake has that session's COTAHIST close, the last close of
    ``precos_recentes``."""
    return precos_do_app.com_fechamento_oficial(cotacao_b3.cotacoes_b3(tickers, precos_recentes), precos_recentes)


def descartar_cotacoes(tickers: tuple[str, ...], precos_recentes: pd.DataFrame) -> None:
    """
    Apaga só a entrada desta carteira (botão "Atualizar cotações"). O cache é do
    processo, compartilhado por todos os usuários do app: ``clear()`` sem
    argumentos apagaria também as cotações dos outros.
    """
    carregar_cotacoes.clear(tickers, precos_recentes)


@st.cache_data(show_spinner=False, max_entries=4)
def calcular_fronteira(mu: pd.Series, cov: pd.DataFrame, params: ParametrosOtimizacao,
                       taxa_livre_risco: float) -> pd.DataFrame:
    """The Markowitz frontier of the page's last section (``core.optimization.fronteira_eficiente``), cached: a rerun
    of any widget does not solve the 30 points again."""
    from core.optimization import fronteira_eficiente
    return fronteira_eficiente(mu, cov, params.para_config(taxa_livre_risco))


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


def carregar_manchetes() -> tuple[pd.DataFrame | None, datetime | None]:
    """The relevant headlines (title, link, topic and tone) and when they were published: live from the lake's log
    on D: (publish time None) where the disk is reachable; on the deployed app, the private store's file read with the
    token of Streamlit secrets (``core.manchetes_online``, DECISOES 134); (None, None) without either."""
    if PASTA_DO_LOG.exists():
        return _manchetes_do_lago(), None
    token = _token_das_manchetes()
    return _manchetes_online(token) if token else (None, None)


@st.cache_data(ttl=600, show_spinner=False)
def _manchetes_do_lago() -> pd.DataFrame:
    itens, notas = sentimento.ler_log(PASTA_DO_LOG, date.today() - timedelta(days=3))
    return sentimento.manchetes(itens, notas, quantas=manchetes_online.QUANTAS_MANCHETES)


def _token_das_manchetes() -> str | None:
    try:
        return st.secrets["manchetes"]["token"]
    except Exception:       # noqa: BLE001 - no secrets file (a local run, the tests) or no [manchetes] entry
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _manchetes_online(token: str) -> tuple[pd.DataFrame | None, datetime | None]:
    texto = manchetes_online.baixar(token)
    if texto is None:
        return None, None
    try:
        return manchetes_online.de_json(texto)
    except (ValueError, KeyError, TypeError):       # a file not made by para_json
        return None, None


def chave_da_api(secao: str, variavel: str) -> str | None:
    """An API key from Streamlit secrets (``[secao] api_key``, the deployed app) or the environment variable
    ``variavel`` (a local run); None without either. The key is only ever sent in a request header."""
    try:
        return st.secrets[secao]["api_key"]
    except Exception:       # noqa: BLE001 - no secrets file (a local run, the tests) or no such entry
        return os.environ.get(variavel) or None


PARECERES_POR_DIA = 8      # paid opinions per day per app process: the public app spends the owner's key


@st.cache_resource
def contador_de_pareceres() -> dict[date, int]:
    """Paid opinions per day, shared by every visitor of this process."""
    return {}


_TRAVA_DO_CONTADOR = threading.Lock()


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=4)
def gerar_parecer(texto_do_pacote: str, universo: tuple[str, ...], dia: date, _pacote: tuple[assessor.Evidencia, ...],
                  _fontes: assessor.Fontes, _chave: str, esforco: str = assessor.ESFORCO,
                  partes: int = 1) -> assessor.Resultado:
    """The adviser's opinion (``core.assessor.pedir_parecer``), once per pack, plan and day: a second click on the
    same day and depth (``esforco``, the thinking effort, and ``partes``, the parallel parts) costs nothing, and past ``PARECERES_POR_DIA`` paid ones a day the answer is the limit message. The pack
    enters the cache key as its text; the tools' sources and the key stay out of it (the key is never hashed)."""
    with _TRAVA_DO_CONTADOR:
        contador = contador_de_pareceres()
        if contador.get(dia, 0) >= PARECERES_POR_DIA:
            return assessor.Resultado(None, f"limite diário de {PARECERES_POR_DIA} pareceres atingido; tente amanhã",
                                      assessor.Uso(), [])
        contador[dia] = contador.get(dia, 0) + 1
    return assessor.pedir_parecer(list(_pacote), list(universo), assessor.Ferramentas(_fontes, {}), _chave,
                                  datetime.now(timezone.utc), carteira=assessor.CARTEIRA_COMPOSTA, esforco=esforco,
                                  partes=partes)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def carregar_focus(dia: date) -> dict[str, pd.DataFrame] | None:
    """The Focus survey's annual indicators, live from the BCB (``core.focus``); None when the API fails."""
    try:
        return {nome: focus.baixar_focus_anual(nome, dia) for nome in INDICADORES_DO_FOCUS}
    except focus.ErroFocus:
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def carregar_indicadores_bcb(dia: date) -> dict[str, tuple[pd.Timestamp, float]] | None:
    """The last IPCA 12 months (SGS 13522, %) and PTAX (SGS 1, R$/US$), live from the BCB, plus the activity, direct
    investment and credit indicators of ``core.macro_bcb`` (the user, 2026-11-04; one that fails is left out); None
    when the IPCA or the PTAX fails."""
    try:
        ipca = bcb.baixar_serie(bcb.SERIE_IPCA_12M, dia - timedelta(days=400), dia).dropna()
        ptax = bcb.baixar_ptax(dia - timedelta(days=30), dia).dropna()
    except bcb.ErroBcb:
        return None
    return {"IPCA 12 meses (%)": (ipca.index[-1], float(ipca.iloc[-1])),
            "Dólar PTAX (R$)": (ptax.index[-1], float(ptax.iloc[-1])),
            **macro_bcb.indicadores(bcb.baixar_serie, dia)}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def carregar_ata_copom(dia: date) -> macro_bcb.Ata | None:
    """The last Copom minutes on the BCB site (``dia`` keys the cache); None when the site fails."""
    try:
        return macro_bcb.ata_do_copom()
    except (requests.RequestException, ValueError, KeyError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_ativos_da_tela() -> pd.DataFrame:
    """Every stock, BDR and ETF of the app's daily prices, with B3's name, type and sector (``ativos_tela``)."""
    return ativos_tela.ler_nomes()


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=2)
def carregar_desempenho(tickers: tuple[str, ...], dia: date, versao_dados: int, referencia: str) -> pd.DataFrame:
    """CAGR, Sharpe, Sortino, Calmar, CVaR and the CAPM of every name of the market table over its 3 years, against
    ``referencia`` and over the daily Selic (``core.indicadores.desempenho_por_acao``): ~15 s, so once per day and price
    version for every visitor of the process."""
    precos = carregar_precos_da_tabela(tickers, dia, versao_dados)
    precos = precos.loc[:, precos.notna().any()]
    return indicadores.desempenho_por_acao(precos, referencia, carregar_selic_diaria(4, dia))


@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def carregar_precos_da_tabela(tickers: tuple[str, ...], dia: date, versao_dados: int) -> pd.DataFrame:
    """3 years of daily closes of ``tickers``, raw (no coverage filter: the table shows every stock, BDR and ETF)."""
    return precos_do_app.ler_fechamentos(list(tickers), dia - timedelta(days=3 * 365 + 10), dia)
