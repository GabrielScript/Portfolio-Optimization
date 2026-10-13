"""
config.py - Constantes e objetos de configuração do motor quantitativo.

Convenções usadas em todo o pacote:
- Retornos SIMPLES diários (P_t / P_{t-1} - 1). Markowitz/Sharpe são definidos
  sobre retornos simples; log-retornos só aparecem onde há composição.
- Anualização aritmética: mu_anual = 252 * mu_diario, Sigma_anual = 252 * Sigma_diario.
- Taxas anuais "de mercado" (Selic/CDI, capitalização composta) são convertidas
  para a mesma convenção aritmética antes de entrar em Sharpe/otimização
  (ver ``taxa_anual_para_aritmetica``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

DIAS_UTEIS_ANO = 252


class Objetivo(str, Enum):
    """Funções-objetivo suportadas pelo otimizador."""

    KELLY = "kelly"                # max mu'w - (lambda/2) w'Sigma w  (crescimento)
    MAX_SHARPE = "max_sharpe"      # Charnes-Cooper: QP convexo exato
    MAX_RETORNO = "max_retorno"    # max mu'w s.a. teto de vol (legado; solução de canto)
    MIN_VOL = "min_vol"            # min w'Sigma w
    HRP = "hrp"                    # Hierarchical Risk Parity (ignora mu)


@dataclass(frozen=True)
class LimiteGrupo:
    """Teto de exposição agregada para um grupo de ativos (ex.: cripto <= 10%)."""

    nome: str
    tickers: frozenset[str]
    teto: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.teto <= 1.0:
            raise ValueError(f"Teto do grupo '{self.nome}' fora de [0, 1]: {self.teto}")


@dataclass(frozen=True)
class ConfigOtimizacao:
    """Parâmetros de uma otimização. Valores anuais na convenção aritmética."""

    objetivo: Objetivo = Objetivo.KELLY
    peso_maximo: float = 0.20
    n_ativos_max: int | None = 10
    vol_maxima: float | None = 0.40
    aversao_risco: float = 2.0
    taxa_livre_risco: float = 0.0
    grupos: tuple[LimiteGrupo, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 < self.peso_maximo <= 1.0:
            raise ValueError(f"peso_maximo deve estar em (0, 1]: {self.peso_maximo}")
        if self.n_ativos_max is not None and self.n_ativos_max < 1:
            raise ValueError(f"n_ativos_max deve ser >= 1: {self.n_ativos_max}")
        if self.n_ativos_max is not None and self.n_ativos_max * self.peso_maximo < 1.0 - 1e-9:
            raise ValueError(
                f"Inviável: {self.n_ativos_max} ativos x teto {self.peso_maximo:.0%} "
                f"= {self.n_ativos_max * self.peso_maximo:.0%} < 100%."
            )
        if self.vol_maxima is not None and self.vol_maxima <= 0:
            raise ValueError(f"vol_maxima deve ser > 0: {self.vol_maxima}")
        if self.aversao_risco <= 0:
            raise ValueError(f"aversao_risco deve ser > 0: {self.aversao_risco}")


@dataclass(frozen=True)
class ConfigBacktest:
    """Parâmetros do walk-forward."""

    janela_treino: int = 2 * DIAS_UTEIS_ANO
    janela_teste: int = 63
    capital_inicial: float = 100_000.0
    custo_bps: float = 10.0          # custo por lado, em bps do valor negociado
    shrink_mu: bool = True

    def __post_init__(self) -> None:
        if self.janela_treino < 60:
            raise ValueError("janela_treino precisa de >= 60 pregões para estimar covariância.")
        if self.janela_teste < 1:
            raise ValueError("janela_teste deve ser >= 1.")
        if self.capital_inicial <= 0:
            raise ValueError("capital_inicial deve ser > 0.")
        if self.custo_bps < 0:
            raise ValueError("custo_bps não pode ser negativo.")


def taxa_anual_para_diaria(taxa_anual: float) -> float:
    """Taxa composta anual (ex.: Selic 14,75% a.a.) -> taxa diária equivalente."""
    return (1.0 + taxa_anual) ** (1.0 / DIAS_UTEIS_ANO) - 1.0


def taxa_anual_para_aritmetica(taxa_anual: float) -> float:
    """
    Converte taxa composta anual para a convenção aritmética do pacote (252 x diária).

    Necessário porque mu é anualizado como 252 * média diária. Comparar esse mu
    com a Selic "cheia" (14,75%) superestima a barreira em ~1 p.p.:
    252 * ((1.1475)^(1/252) - 1) = 13,77%.
    """
    return DIAS_UTEIS_ANO * taxa_anual_para_diaria(taxa_anual)
