"""
backtest.py - Walk-forward fora da amostra, com custos de transação.

Protocolo (idêntico para TODAS as estratégias, para que a comparação seja justa):
1. No rebalanceamento t, a estratégia recebe SÓ os retornos de [t - treino, t).
   Os pesos decididos no fechamento de t-1 valem a partir do retorno de t.
2. Custo = custo_bps x sum_i |w_alvo_i - w_derivado_i| x valor. Cobra os dois
   lados (compra e venda) e a montagem inicial a partir do caixa.
3. Entre rebalanceamentos a carteira é buy-and-hold: os pesos derivam com os
   preços.
4. Se a estratégia falha (solver, amostra curta), a carteira é mantida e a
   falha é registrada. A versão anterior trocava por 1/N em silêncio.

O que este backtest NÃO corrige (documentado na UI):
- Viés de sobrevivência: o universo é a lista ATUAL de tickers.
- Filtro/tilt growth: usa fundamentos de hoje, não pode entrar no
  walk-forward sem dados point-in-time. A curva OOS mede só a ALOCAÇÃO.
- Imposto de renda (15% sobre ganho; isenção de R$ 20 mil/mês em vendas de
  ações) não é modelado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from core.config import DIAS_UTEIS_ANO, ConfigBacktest, ConfigOtimizacao
from core.estimation import covariancia_ledoit_wolf, estimar_momentos
from core.hierarchical import otimizar_hrp, otimizar_hrp_arvore
from core.optimization import ErroOtimizacao, combinar_nucleo_satelite, otimizar
from core.reamostragem import ConfigReamostragem, otimizar_reamostrado

logger = logging.getLogger(__name__)

Estrategia = Callable[[pd.DataFrame], pd.Series]
_FALHAS_ESPERADAS = (ErroOtimizacao, ValueError, np.linalg.LinAlgError)


@dataclass
class ResultadoBacktest:
    nome: str
    metodologia: str                        # "walk-forward OOS" | "in-sample"
    valores: pd.Series                      # curva de capital (R$), inclui a data inicial
    pesos_alvo: pd.DataFrame                # pesos decididos em cada rebalanceamento
    turnover: pd.Series                     # sum |delta w| por rebalanceamento
    custos_totais: float
    falhas: list[str] = field(default_factory=list)

    @property
    def turnover_anual_medio(self) -> float:
        anos = max((len(self.valores) - 1) / DIAS_UTEIS_ANO, 1e-9)
        return float(self.turnover.sum() / anos)


def _validar_pesos(pesos: pd.Series, colunas: pd.Index) -> pd.Series:
    alinhados = pesos.reindex(colunas).fillna(0.0)
    if (alinhados < -1e-8).any():
        raise ValueError("Estratégia devolveu peso negativo (carteira é long-only).")
    total = float(alinhados.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Estratégia devolveu pesos nulos ou não finitos.")
    return alinhados.clip(lower=0.0) / total


def _simular(retornos: pd.DataFrame, decidir_pesos: Callable[[int], pd.Series],
             inicio: int, passo: int, capital: float, custo_bps: float,
             nome: str, metodologia: str) -> ResultadoBacktest:
    """Motor comum: rebalanceia em inicio, inicio+passo, ...; buy-and-hold entre eles."""
    colunas = retornos.columns
    valor = capital
    pesos_derivados = pd.Series(0.0, index=colunas)       # 0 = 100% caixa antes da montagem
    data_inicial = retornos.index[inicio - 1] if inicio > 0 else retornos.index[0] - pd.Timedelta(days=1)
    datas, valores = [data_inicial], [valor]
    registros_pesos, registros_turnover, falhas = {}, {}, []
    custos = 0.0

    for posicao in range(inicio, len(retornos), passo):
        data = retornos.index[posicao]
        try:
            alvo = _validar_pesos(decidir_pesos(posicao), colunas)
        except _FALHAS_ESPERADAS as erro:
            falhas.append(f"{data:%Y-%m-%d}: {erro}")
            logger.warning("%s falhou em %s (%s); mantendo carteira.", nome, data.date(), erro)
            alvo = (pesos_derivados if pesos_derivados.sum() > 0
                    else pd.Series(1.0 / len(colunas), index=colunas))

        giro = float((alvo - pesos_derivados).abs().sum())
        custo = valor * giro * custo_bps / 1e4
        valor -= custo
        custos += custo
        registros_pesos[data], registros_turnover[data] = alvo, giro

        bloco = retornos.iloc[posicao:posicao + passo]
        crescimento = (1.0 + bloco).cumprod()
        curva = valor * (crescimento.values @ alvo.values)
        datas.extend(bloco.index)
        valores.extend(curva)

        final = alvo.values * crescimento.values[-1]
        pesos_derivados = pd.Series(final / final.sum(), index=colunas)
        valor = float(curva[-1])

    return ResultadoBacktest(
        nome=nome, metodologia=metodologia,
        valores=pd.Series(valores, index=pd.DatetimeIndex(datas)),
        pesos_alvo=pd.DataFrame(registros_pesos).T,
        turnover=pd.Series(registros_turnover),
        custos_totais=custos, falhas=falhas,
    )


def walk_forward(retornos: pd.DataFrame, estrategia: Estrategia, cfg: ConfigBacktest,
                 nome: str) -> ResultadoBacktest:
    """Walk-forward OOS. ``retornos``: simples, diários, sem NaN (amostra comum)."""
    if retornos.isna().any().any():
        raise ValueError("retornos contém NaN; use estimation.retornos_simples().")
    if len(retornos) <= cfg.janela_treino + 1:
        raise ValueError(
            f"Histórico insuficiente: {len(retornos)} pregões para treino de {cfg.janela_treino}."
        )

    def decidir(posicao: int) -> pd.Series:
        return estrategia(retornos.iloc[posicao - cfg.janela_treino:posicao])

    return _simular(retornos, decidir, cfg.janela_treino, cfg.janela_teste,
                    cfg.capital_inicial, cfg.custo_bps, nome, "walk-forward OOS")


def backtest_pesos_fixos(retornos: pd.DataFrame, pesos: pd.Series, cfg: ConfigBacktest,
                         nome: str) -> ResultadoBacktest:
    """
    Aplica pesos fixos a todo o histórico, rebalanceando a cada janela_teste.
    Se os pesos foram estimados nesse mesmo histórico, o resultado é
    IN-SAMPLE e otimista por construção. Serve só para ilustrar.
    """
    return _simular(retornos, lambda _: pesos, 0, cfg.janela_teste,
                    cfg.capital_inicial, cfg.custo_bps, nome, "in-sample")


# ---------------------------------------------------------------- estratégias

def _cfg_com_rf_da_data(cfg: ConfigOtimizacao, rf_diaria: pd.Series | None,
                        data: pd.Timestamp) -> ConfigOtimizacao:
    """Taxa livre de risco vigente NA DATA (point-in-time), na convenção aritmética 252 x diária."""
    if rf_diaria is None or rf_diaria.empty:
        return cfg
    rf_vigente = rf_diaria.asof(data)
    if pd.isna(rf_vigente):
        return cfg
    return replace(cfg, taxa_livre_risco=float(rf_vigente) * DIAS_UTEIS_ANO)


def estrategia_otimizada(cfg_otimizacao: ConfigOtimizacao, shrink_mu: bool,
                         rf_diaria: pd.Series | None = None,
                         meia_vida_cov: float | None = None) -> Estrategia:
    """
    Estima momentos no treino e otimiza. Se ``rf_diaria`` (CDI) for dada, a taxa
    livre de risco é a vigente na data do rebalanceamento. Sem ela, usa a do cfg.
    Com ``meia_vida_cov``, Sigma é a covariância EWMA (ver ``estimar_momentos``).
    """
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        momentos = estimar_momentos(treino, shrink_mu=shrink_mu, meia_vida_cov=meia_vida_cov)
        cfg = _cfg_com_rf_da_data(cfg_otimizacao, rf_diaria, treino.index[-1])
        return otimizar(momentos.mu, momentos.cov, cfg).pesos
    return _estrategia


def estrategia_reamostrada(cfg_otimizacao: ConfigOtimizacao, shrink_mu: bool,
                           rf_diaria: pd.Series | None = None,
                           config: ConfigReamostragem = ConfigReamostragem()) -> Estrategia:
    """Como ``estrategia_otimizada``, mas com a carteira reamostrada (core.reamostragem) do treino."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        cfg = _cfg_com_rf_da_data(cfg_otimizacao, rf_diaria, treino.index[-1])
        return otimizar_reamostrado(treino, cfg, shrink_mu=shrink_mu, config=config).resultado.pesos
    return _estrategia


def com_janela(estrategia: Estrategia, pregoes: int) -> Estrategia:
    """
    A mesma estratégia, estimada só nos últimos ``pregoes`` do treino. Com o walk-forward
    configurado para a maior janela, estratégias de janelas diferentes decidem nas mesmas
    datas e no mesmo universo.
    """
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        if len(treino) < pregoes:
            raise ValueError(f"Treino de {len(treino)} pregões para uma janela de {pregoes}.")
        return estrategia(treino.iloc[-pregoes:])
    return _estrategia


def estrategia_nucleo_satelite(nucleo: Estrategia, satelite: Estrategia, fracao_satelite: float) -> Estrategia:
    """
    Núcleo + satélite decididos no mesmo treino, pesos combinados (``combinar_nucleo_satelite``). Igual, nos
    rebalanceios, às duas curvas levadas de volta ao alvo; o custo sai um pouco menor (compras e vendas se compensam).
    """
    return lambda treino: combinar_nucleo_satelite(nucleo(treino), satelite(treino), fracao_satelite)


def estrategia_equiponderada() -> Estrategia:
    return lambda treino: pd.Series(1.0 / treino.shape[1], index=treino.columns)


def estrategia_hrp(arvore: bool = False) -> Estrategia:
    alocador = otimizar_hrp_arvore if arvore else otimizar_hrp
    return lambda treino: alocador(covariancia_ledoit_wolf(treino))


def estrategia_buy_and_hold_ativo(ticker: str) -> Estrategia:
    """Benchmark de um único ativo (ex.: BOVA11) dentro do mesmo motor e custos."""
    return lambda treino: pd.Series({ticker: 1.0}).reindex(treino.columns).fillna(0.0)
