"""
servicos.py - Fachada com cache entre a UI e o core.

Regra de cache: TODO argumento que muda o resultado entra na chave. A versão
anterior passava os preços como ``_precos``, e o Streamlit ignora argumentos
com prefixo "_" no hash. Mudar o filtro growth alterava o universo, mas o
backtest antigo continuava sendo exibido. Aqui as funções recebem só tipos
hasheáveis (tuplas, números, DataFrames) e montam os objetos do core dentro.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from core import market_data
from core.backtest import (ResultadoBacktest, estrategia_equiponderada, estrategia_hrp,
                           estrategia_otimizada, walk_forward)
from core.config import ConfigBacktest, ConfigOtimizacao, LimiteGrupo, Objetivo
from core.growth import calcular_growth_scores


@dataclass(frozen=True)
class ParametrosOtimizacao:
    """Parâmetros vindos da sidebar, todos hasheáveis."""

    objetivo: str
    peso_maximo: float
    n_ativos_max: int
    vol_maxima: float | None
    aversao_risco: float
    tickers_cripto: tuple[str, ...]
    teto_cripto: float

    def para_config(self, taxa_livre_risco: float) -> ConfigOtimizacao:
        grupos = ()
        if self.tickers_cripto:
            grupos = (LimiteGrupo("cripto", frozenset(self.tickers_cripto), self.teto_cripto),)
        return ConfigOtimizacao(
            objetivo=Objetivo(self.objetivo), peso_maximo=self.peso_maximo,
            n_ativos_max=self.n_ativos_max, vol_maxima=self.vol_maxima,
            aversao_risco=self.aversao_risco, taxa_livre_risco=taxa_livre_risco, grupos=grupos,
        )


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_precos(tickers: tuple[str, ...], anos: int, dia: date) -> market_data.DadosPrecos:
    """``dia`` entra na chave para invalidar o cache na virada do dia."""
    return market_data.carregar_precos(list(tickers), anos, hoje=dia)


@st.cache_data(ttl=86400, show_spinner=False)
def carregar_cdi(anos: int, dia: date) -> pd.Series:
    try:
        return market_data.baixar_cdi_diario(anos, hoje=dia)
    except market_data.ErroDadosMercado:
        return pd.Series(dtype=float)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def carregar_selic(dia: date) -> market_data.TaxaSelic:
    return market_data.obter_selic_meta(hoje=dia)


@st.cache_data(ttl=86400, show_spinner=False)
def carregar_growth(tickers: tuple[str, ...], precos: pd.DataFrame, setores: tuple[tuple[str, str], ...],
                    usar_cvm: bool, dia: date) -> pd.DataFrame:
    return calcular_growth_scores(list(tickers), precos, dict(setores), usar_cvm=usar_cvm,
                                  data_referencia=dia)


@st.cache_data(show_spinner=False, max_entries=8)
def rodar_walk_forward(retornos: pd.DataFrame, params: ParametrosOtimizacao, shrink_mu: bool,
                       custo_bps: float, capital: float, cdi_diario: pd.Series,
                       taxa_constante_aritmetica: float) -> dict[str, ResultadoBacktest]:
    """
    Todas as estratégias no MESMO protocolo: mesmo universo (pré-filtro growth),
    mesmas janelas e mesmos custos. rf point-in-time pelo CDI.
    """
    cfg_bt = ConfigBacktest(capital_inicial=capital, custo_bps=custo_bps, shrink_mu=shrink_mu)
    cfg_opt = params.para_config(taxa_constante_aritmetica)
    rf = cdi_diario if not cdi_diario.empty else None
    estrategias = {
        f"Otimizada ({params.objetivo})": estrategia_otimizada(cfg_opt, shrink_mu, rf),
        "1/N": estrategia_equiponderada(),
        "HRP": estrategia_hrp(arvore=False),
        "HRP-Árvore": estrategia_hrp(arvore=True),
    }
    return {nome: walk_forward(retornos, estrategia, cfg_bt, nome)
            for nome, estrategia in estrategias.items()}
