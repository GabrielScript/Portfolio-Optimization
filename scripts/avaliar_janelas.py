"""
Avaliação PRÉ-REGISTRADA da janela de estimação (rolling 2, 3 e 5 anos e Sigma EWMA).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assets import BENCHMARK_ACOES  # noqa: E402
from core.backtest import (Estrategia, ResultadoBacktest, com_janela, estrategia_buy_and_hold_ativo,  # noqa: E402
                           estrategia_equiponderada, estrategia_otimizada, walk_forward)
from core import market_data  # noqa: E402
from core.config import (DIAS_UTEIS_ANO, ConfigBacktest, ConfigOtimizacao, Objetivo,  # noqa: E402
                         limites_por_setor, taxa_anual_para_aritmetica)
from core.metrics import (momentos_sharpe, resumo_desempenho, retornos_de_valores,  # noqa: E402
                          sharpe_probabilistico, sharpe_referencia_deflacionado)

CONTROLE = "R5"
MEIA_VIDA_EWMA = 126                        # pregões (6 meses)
PREGOES_DE_TREINO = {"R5": 5 * DIAS_UTEIS_ANO, "R3": 3 * DIAS_UTEIS_ANO, "R2": 2 * DIAS_UTEIS_ANO,
                     "EWMA": 5 * DIAS_UTEIS_ANO}
# Um cfg só para candidatas e referências: treino da maior janela, mesmas datas, mesmos custos.
CFG_BT = ConfigBacktest(janela_treino=max(PREGOES_DE_TREINO.values()), janela_teste=63,
                        capital_inicial=10_000.0, custo_bps=10.0, shrink_mu=True)
CFG_KELLY = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=10, vol_maxima=0.40,
                             aversao_risco=2.0)
TETO_SETOR = 0.30
N_DESAFIANTES = 3
LIMIAR_P = 1.0 - 0.05 / N_DESAFIANTES       # Bonferroni: 0,95 dividido entre os 3 desafiantes
PIORA_MAXIMA_DRAWDOWN = 0.05                # em pontos do drawdown máximo
GIRO_MAXIMO_RELATIVO = 2.0                  # giro anual da desafiante <= 2x o do controle


@dataclass(frozen=True)
class Dados:
    retornos: pd.DataFrame       # amostra comum das ações e do BOVA11
    acoes: list[str]
    setores: dict[str, str]      # ticker -> setor
    cdi: pd.Series               # vazia se o BCB falhar
    selic: float                 # Selic meta de hoje (fallback da taxa livre de risco)


def estrategias(cfg: ConfigOtimizacao, cdi_diario: pd.Series | None) -> dict[str, Estrategia]:
    """As 4 candidatas, cada uma no próprio recorte do treino (o walk-forward entrega o treino da maior)."""
    return {
        nome: com_janela(estrategia_otimizada(cfg, True, cdi_diario,
                                              meia_vida_cov=MEIA_VIDA_EWMA if nome == "EWMA" else None),
                         pregoes)
        for nome, pregoes in PREGOES_DE_TREINO.items()
    }


def rodar(retornos: pd.DataFrame, acoes: list[str], cfg: ConfigOtimizacao,
          cdi_diario: pd.Series | None) -> dict[str, ResultadoBacktest]:
    """Candidatas, 1/N e BOVA11 no mesmo walk-forward. ``retornos``: amostra comum das ações e do BOVA11."""
    resultados = {nome: walk_forward(retornos[acoes], estrategia, CFG_BT, nome)
                  for nome, estrategia in estrategias(cfg, cdi_diario).items()}
    resultados["1/N"] = walk_forward(retornos[acoes], estrategia_equiponderada(), CFG_BT, "1/N")
    resultados[BENCHMARK_ACOES] = walk_forward(retornos, estrategia_buy_and_hold_ativo(BENCHMARK_ACOES),
                                               CFG_BT, BENCHMARK_ACOES)
    return resultados


@dataclass(frozen=True)
class Medidas:
    """O que a regra de decisão usa de cada estratégia no walk-forward."""

    sharpe: float
    max_drawdown: float          # <= 0
    giro_anual: float
    p_supera_controle: float     # PSR da diferença diária contra o controle (NaN no próprio controle)


def medir(resultados: Mapping[str, ResultadoBacktest], rf_diaria: pd.Series,
          controle: str = CONTROLE) -> dict[str, Medidas]:
    """
    Medidas de cada estratégia. O P vem da série diferença diária (estratégia - controle):
    a taxa livre de risco se cancela nela, então a falta do CDI não muda a decisão.
    """
    retornos_controle = retornos_de_valores(resultados[controle].valores)
    medidas = {}
    for nome, resultado in resultados.items():
        resumo = resumo_desempenho(resultado.valores, rf_diaria)
        diferenca = retornos_de_valores(resultado.valores) - retornos_controle
        p = float("nan") if nome == controle else sharpe_probabilistico(momentos_sharpe(diferenca))
        medidas[nome] = Medidas(resumo.sharpe, resumo.max_drawdown, resultado.turnover_anual_medio, p)
    return medidas


@dataclass(frozen=True)
class Veredito:
    escolhida: str
    criterios: dict[str, dict[str, bool]]     # desafiante -> {critério: passou}


def criterios(desafiante: Medidas, controle: Medidas) -> dict[str, bool]:
    """Os três critérios pré-registrados. NaN reprova: toda comparação com NaN é falsa."""
    return {
        f"C1 P[supera {CONTROLE}] >= {LIMIAR_P:.4f}": desafiante.p_supera_controle >= LIMIAR_P,
        f"C2 drawdown no máximo {PIORA_MAXIMA_DRAWDOWN:.0%} pior":
            desafiante.max_drawdown >= controle.max_drawdown - PIORA_MAXIMA_DRAWDOWN,
        f"C3 giro <= {GIRO_MAXIMO_RELATIVO:g}x": desafiante.giro_anual <= GIRO_MAXIMO_RELATIVO * controle.giro_anual,
    }


def decidir(medidas: Mapping[str, Medidas], controle: str = CONTROLE) -> Veredito:
    """A desafiante aprovada de maior Sharpe; sem aprovada, o controle."""
    avaliacoes = {nome: criterios(m, medidas[controle]) for nome, m in medidas.items() if nome != controle}
    aprovadas = [nome for nome, resultado in avaliacoes.items() if all(resultado.values())]
    return Veredito(max(aprovadas, key=lambda nome: medidas[nome].sharpe, default=controle), avaliacoes)


def montar_tabela(resultados: Mapping[str, ResultadoBacktest], medidas: Mapping[str, Medidas],
                  rf_diaria: pd.Series) -> pd.DataFrame:
    """Uma linha por estratégia. DSR só nas candidatas (N = 4): as referências não foram escolhidas."""
    linhas, momentos = {}, {}
    for nome, resultado in resultados.items():
        resumo = resumo_desempenho(resultado.valores, rf_diaria)
        retornos = retornos_de_valores(resultado.valores)
        momentos[nome] = momentos_sharpe(retornos - rf_diaria.reindex(retornos.index).fillna(0.0))
        linhas[nome] = {"Retorno total": resumo.retorno_total, "CAGR": resumo.cagr, "Vol": resumo.volatilidade,
                        "Sharpe": resumo.sharpe, "Sortino": resumo.sortino, "MDD": resumo.max_drawdown,
                        "CVaR95 diário": resumo.cvar_95_diario, "Giro anual": resultado.turnover_anual_medio,
                        "PSR": sharpe_probabilistico(momentos[nome]), "DSR": float("nan"),
                        f"P[> {CONTROLE}]": medidas[nome].p_supera_controle, "Falhas": len(resultado.falhas)}
    sharpes = [momentos[nome].sharpe for nome in PREGOES_DE_TREINO]
    referencia = sharpe_referencia_deflacionado(float(np.var(sharpes, ddof=1)), len(sharpes))
    for nome in PREGOES_DE_TREINO:
        linhas[nome]["DSR"] = sharpe_probabilistico(momentos[nome], referencia)
    return pd.DataFrame(linhas).T


@dataclass(frozen=True)
class Avaliacao:
    resultados: dict[str, ResultadoBacktest]
    medidas: dict[str, Medidas]
    veredito: Veredito
    tabela: pd.DataFrame


def avaliar(dados: Dados) -> Avaliacao:
    """O teste inteiro sobre dados já carregados (sem rede)."""
    cfg = replace(CFG_KELLY, taxa_livre_risco=taxa_anual_para_aritmetica(dados.selic),
                  grupos=limites_por_setor(dados.setores, TETO_SETOR))
    resultados = rodar(dados.retornos, dados.acoes, cfg, dados.cdi if not dados.cdi.empty else None)
    rf = market_data.serie_rf_diaria(resultados[CONTROLE].valores.index, dados.cdi, dados.selic)
    medidas = medir(resultados, rf)
    veredito = decidir({nome: medidas[nome] for nome in PREGOES_DE_TREINO})
    return Avaliacao(resultados, medidas, veredito, montar_tabela(resultados, medidas, rf))
