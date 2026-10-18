"""
risco.py - Regras de risco fixas, conferidas num ponto só.

Toda carteira que vira ordem (carteira de hoje, plano de compra, paper trading, dinheiro real) passa por aqui.
O módulo não otimiza nem muda pesos: aprova ou recusa. A tela pode apertar os limites, nunca afrouxar; mudar um
valor de ``LIMITES`` é pré-registro novo (``PLAN.md``, "Regras de risco fixas").
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from core.config import ConfigOtimizacao

TOLERANCIA = 1e-6          # a mesma do ``core.ordens`` (TOLERANCIA_SOMA_PESOS)


@dataclass(frozen=True)
class Limites:
    alavancagem_maxima: float = 1.0          # w >= 0, soma <= 1, caixa >= 0
    peso_maximo_acao: float = 0.20
    peso_maximo_emissor: float = 0.20        # somando as classes (ITUB3 + ITUB4)
    teto_setor: float = 0.30
    vol_maxima: float = 0.40                 # ex-ante, anual, teto obrigatório
    fracao_maxima_satelite: float = 0.50
    fracao_padrao_satelite: float = 0.30
    queda_relativa_stop: float = 0.25        # queda da razão satélite/BOVA11 desde o pico
    perda_diaria_maxima: float = 0.05        # perda do satélite num pregão que congela as ordens
    pregoes_congelados: int = 1


LIMITES = Limites()


class Regra(str, Enum):
    ALAVANCAGEM = "alavancagem"
    PESO_NEGATIVO = "peso negativo"
    ACAO = "peso por ação"
    EMISSOR = "peso por emissor"
    SETOR = "teto de setor"
    SEM_SETOR = "papel sem setor"
    VOL = "vol ex-ante"


@dataclass(frozen=True)
class Violacao:
    regra: Regra
    alvo: str          # ticker, emissor, setor ou "carteira"
    valor: float
    limite: float


def emissor(ticker: str) -> str:
    """Código do emissor na B3: as 4 primeiras letras do ticker (ITUB3 e ITUB4.SA -> ITUB)."""
    return ticker[:4]


def checar_carteira(pesos: pd.Series, setores: Mapping[str, str], cov: pd.DataFrame | None = None,
                    limites: Limites = LIMITES) -> list[Violacao]:
    """
    Todas as violações das regras fixas na carteira ``pesos`` (fração do patrimônio; o resto é caixa).
    ``cov`` anual, a mesma do ``otimizar``; sem ela a vol ex-ante não é conferida. Tolerância de 1e-6.
    """
    violacoes = [Violacao(Regra.PESO_NEGATIVO, ticker, float(peso), 0.0)
                 for ticker, peso in pesos.items() if peso < -TOLERANCIA]
    # exposição comprada: a soma líquida esconderia 120% comprado com 20% vendido (que já viola PESO_NEGATIVO)
    comprado = float(pesos.clip(lower=0.0).sum())
    if comprado > limites.alavancagem_maxima + TOLERANCIA:
        violacoes.append(Violacao(Regra.ALAVANCAGEM, "carteira", comprado, limites.alavancagem_maxima))
    com_peso = pesos[pesos > TOLERANCIA]
    violacoes += [Violacao(Regra.ACAO, ticker, float(peso), limites.peso_maximo_acao)
                  for ticker, peso in com_peso.items() if peso > limites.peso_maximo_acao + TOLERANCIA]
    classes = com_peso.groupby(com_peso.index.map(emissor))
    # emissor de uma classe só já é conferido pela regra da ação
    violacoes += [Violacao(Regra.EMISSOR, nome, float(grupo.sum()), limites.peso_maximo_emissor)
                  for nome, grupo in classes
                  if len(grupo) > 1 and grupo.sum() > limites.peso_maximo_emissor + TOLERANCIA]
    violacoes += [Violacao(Regra.SEM_SETOR, ticker, float(peso), 0.0)
                  for ticker, peso in com_peso.items() if not setores.get(ticker)]
    por_setor = com_peso.groupby(com_peso.index.map(lambda t: setores.get(t) or None)).sum()   # sem setor fica fora
    violacoes += [Violacao(Regra.SETOR, setor, float(peso), limites.teto_setor)
                  for setor, peso in por_setor.items() if peso > limites.teto_setor + TOLERANCIA]
    if cov is not None:
        w = com_peso.values
        vol = float(np.sqrt(w @ cov.loc[com_peso.index, com_peso.index].values @ w))   # KeyError se a cov não cobre
        if vol > limites.vol_maxima + TOLERANCIA:
            violacoes.append(Violacao(Regra.VOL, "carteira", vol, limites.vol_maxima))
    return violacoes


def validar_config(cfg: ConfigOtimizacao, limites: Limites = LIMITES) -> list[Violacao]:
    """
    Recusa a configuração que afrouxa uma regra fixa: peso por ação acima do limite, sem teto de vol (ou acima
    dele), sem teto de setor (``grupos=()`` = 100% num setor) ou grupo com teto acima do de setor.
    """
    violacoes = []
    if cfg.peso_maximo > limites.peso_maximo_acao + TOLERANCIA:
        violacoes.append(Violacao(Regra.ACAO, "configuração", cfg.peso_maximo, limites.peso_maximo_acao))
    if cfg.vol_maxima is None or cfg.vol_maxima > limites.vol_maxima + TOLERANCIA:
        vol = float("inf") if cfg.vol_maxima is None else cfg.vol_maxima
        violacoes.append(Violacao(Regra.VOL, "configuração", vol, limites.vol_maxima))
    if not cfg.grupos:
        violacoes.append(Violacao(Regra.SETOR, "configuração", 1.0, limites.teto_setor))
    violacoes += [Violacao(Regra.SETOR, grupo.nome, grupo.teto, limites.teto_setor)
                  for grupo in cfg.grupos if grupo.teto > limites.teto_setor + TOLERANCIA]
    return violacoes


def stop_relativo(valores_satelite: pd.Series, valores_referencia: pd.Series, limites: Limites = LIMITES) -> bool:
    """
    True se a razão satélite/referência (BOVA11) caiu ``queda_relativa_stop`` ou mais desde o pico, nas datas em
    comum. Mede o risco ativo: queda igual à do mercado não dispara. Conferido em cada rebalanceio.
    """
    razao = (valores_satelite / valores_referencia).dropna()
    if razao.empty:
        return False
    queda = 1.0 - razao.iloc[-1] / razao.cummax().iloc[-1]
    return bool(queda >= limites.queda_relativa_stop - TOLERANCIA)


def pode_negociar(retornos_diarios: pd.Series, limites: Limites = LIMITES) -> bool:
    """
    False nos ``pregoes_congelados`` pregões seguintes a uma perda do satélite de ``perda_diaria_maxima`` ou mais
    (fechamento a fechamento das posições registradas). Só congela ordens: nunca manda vender.
    """
    recentes = retornos_diarios.dropna().iloc[-limites.pregoes_congelados:]
    return not bool((recentes <= -limites.perda_diaria_maxima + TOLERANCIA).any())
