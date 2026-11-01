"""
estrategias_todas_us.py - The recorded strategies on the US's monthly grid with thousands of names on 36 rows
(docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md, sections 2 and 4). Each has the optimum of the dense code it stands in
for, on the low-rank covariance. Every function takes returns at their real monthly scale, under ``no_grid_mensal_real``: the engine's
annualization by 252 would inflate the mean, whatever the scale of the rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.baixo_posto import CovarianciaBaixoPosto, densa, ledoit_wolf_baixo_posto, otimizar_baixo_posto
from core.carteiras_livros import media_por_periodo
from core.config import ConfigOtimizacao, limites_por_setor
from core.estimation import estimar_momentos
from core.optimization import ErroOtimizacao, otimizar, projetar_pesos
from core.reamostragem import ConfigReamostragem, indices_bootstrap_circular
from core.risco import LIMITES, Limites, checar_carteira
from core.us_teste import MESES_POR_ANO
from core.valor import NotaDoMes
from core.views import DELTA, TAU, Sinal, mu_com_views

DIAS_DO_MOTOR = 252
PARA_A_ESCALA_DO_MOTOR = np.sqrt(MESES_POR_ANO / DIAS_DO_MOTOR)     # as ``no_grid_mensal``: 252 x this squared = 12 on a covariance
DO_MOTOR_PARA_O_MES = MESES_POR_ANO / DIAS_DO_MOTOR                 # ``estimar_momentos`` annualized real monthly rows by 252

Selecao = Callable[[pd.Timestamp, pd.DataFrame, pd.Series], list[str]]


def _com_rf(cfg: ConfigOtimizacao, rf_mensal: pd.Series | None, data: pd.Timestamp) -> ConfigOtimizacao:
    """The T-bill of the last complete month, annualized by 12 as the engine's rate by 252. ``data`` is the decision's month."""
    if rf_mensal is None or rf_mensal.empty:
        return cfg
    vigente = rf_mensal.asof(data - pd.offsets.MonthEnd(1))
    return cfg if pd.isna(vigente) else replace(cfg, taxa_livre_risco=float(vigente) * MESES_POR_ANO)


def _elegiveis(treino: pd.DataFrame, setores: Mapping[str, str] | None) -> pd.DataFrame:
    return treino if setores is None else treino[[papel for papel in treino.columns if setores.get(papel)]]


def _nas_regras(cfg: ConfigOtimizacao, retornos: pd.DataFrame, setores: Mapping[str, str], limites: Limites) -> ConfigOtimizacao:
    return replace(cfg, grupos=limites_por_setor({papel: setores[papel] for papel in retornos.columns}, limites.teto_setor))


def _conferir(pesos: pd.Series, sigma: CovarianciaBaixoPosto, setores: Mapping[str, str], limites: Limites) -> None:
    violacoes = checar_carteira(pesos, setores, densa(sigma, np.flatnonzero(pesos.to_numpy() > 0.0)), limites)
    if violacoes:
        raise CarteiraRecusada(violacoes)


def estrategia_grande(cfg: ConfigOtimizacao, rf_mensal: pd.Series | None = None, setores: Mapping[str, str] | None = None,
                      limites: Limites = LIMITES) -> Estrategia:
    """``estrategia_otimizada``, or with ``setores`` ``estrategia_otimizada_nas_regras``, with Jorion's mean where it applies."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        retornos = _elegiveis(treino, setores)
        sigma = ledoit_wolf_baixo_posto(retornos, MESES_POR_ANO)
        cfg_da_data = _com_rf(cfg, rf_mensal, treino.index[-1])
        if setores is not None:
            cfg_da_data = _nas_regras(cfg_da_data, retornos, setores, limites)
        pesos = otimizar_baixo_posto(media_por_periodo(retornos, sigma, MESES_POR_ANO) * MESES_POR_ANO, sigma, cfg_da_data)
        if setores is not None:
            _conferir(pesos, sigma, setores, limites)
        return pesos
    return _estrategia


def estrategia_reamostrada_grande(cfg: ConfigOtimizacao, config: ConfigReamostragem, periodos: int = MESES_POR_ANO) -> Estrategia:
    """``otimizar_reamostrado``: the average of the resamples' portfolios, projected on the caps. A resample that fails is dropped."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        linhas = len(treino)
        gerador = np.random.default_rng(config.semente)
        sorteios = [indices_bootstrap_circular(linhas, config.tamanho_bloco, gerador) for _ in range(config.n_reamostras)]
        cfg_reamostra = replace(cfg, n_ativos_max=None)
        soma, validas = pd.Series(0.0, index=treino.columns), 0
        for indices in sorteios:
            amostra = pd.DataFrame(treino.to_numpy()[indices], columns=treino.columns)
            try:
                sigma = ledoit_wolf_baixo_posto(amostra, periodos)
                soma += otimizar_baixo_posto(media_por_periodo(amostra, sigma, periodos) * periodos, sigma, cfg_reamostra)
            except (ErroOtimizacao, ValueError, np.linalg.LinAlgError):
                continue
            validas += 1
        if validas == 0:
            raise ErroOtimizacao(f"All {config.n_reamostras} resamples failed.")
        return projetar_pesos(soma / validas, densa(ledoit_wolf_baixo_posto(treino, periodos)), cfg).pesos
    return _estrategia


def hrp_grande(alocador: Callable[[pd.DataFrame], pd.Series]) -> Estrategia:
    """An HRP on the Ledoit-Wolf covariance, without the dense estimator's cost. HRP does not read the mean or the scale."""
    return lambda treino: alocador(densa(ledoit_wolf_baixo_posto(treino, MESES_POR_ANO)))


def estrategia_views_grande(cfg: ConfigOtimizacao, setores: Mapping[str, str], sinais: Sequence[Sinal], valor_mercado: pd.DataFrame,
                            limites: Limites = LIMITES) -> Estrategia:
    """``estrategia_com_views``: the views' mean comes from the covariance alone, the historical mean is never read."""
    meses = pd.to_datetime(valor_mercado["month"])

    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        retornos = _elegiveis(treino, setores)
        mes = treino.index[-1].to_period("M").to_timestamp()
        sigma = ledoit_wolf_baixo_posto(retornos, MESES_POR_ANO)
        do_mes = valor_mercado[meses == mes].set_index("issuer")["market_cap"]
        mu = mu_com_views(densa(sigma), list(retornos.columns), mes, do_mes, sinais, DELTA, TAU)
        pesos = otimizar_baixo_posto(mu, sigma, _nas_regras(cfg, retornos, setores, limites))
        _conferir(pesos, sigma, setores, limites)
        return pesos
    return _estrategia


def selecao_e_max_sharpe_mensal(selecionar: Selecao, nota: NotaDoMes, setores: Mapping[str, str], cfg: ConfigOtimizacao,
                                rf_mensal: pd.Series | None = None, limites: Limites = LIMITES) -> Estrategia:
    """``selection_then_max_sharpe``. The selection reads only the covariance, so it takes rows on the engine's scale; the weights, on
    the few names chosen, read the mean: the engine's moments on the real rows, brought from 252 to 12 periods a year."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        data = treino.index[-1]
        score = nota(data.to_period("M").to_timestamp(), list(treino.columns))
        nomes = selecionar(data, treino * PARA_A_ESCALA_DO_MOTOR, score)
        if not nomes:
            raise ValueError(f"{data:%Y-%m-%d}: no name selected")
        retornos = _elegiveis(treino[nomes], setores)
        momentos = estimar_momentos(retornos, shrink_mu=True)
        cov = momentos.cov * DO_MOTOR_PARA_O_MES
        cfg_da_data = _nas_regras(_com_rf(cfg, rf_mensal, data), retornos, setores, limites)
        pesos = otimizar(momentos.mu * DO_MOTOR_PARA_O_MES, cov, cfg_da_data).pesos
        violacoes = checar_carteira(pesos, setores, cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia
