"""
views.py - Black-Litterman com views dos sinais aprovados no S2 (PLAN §5, S3): prior de equilíbrio pelos pesos de
mercado do lago, uma view absoluta por sinal com a confiança tirada do IC histórico ponto no tempo, e o Kelly com as
regras sobre a média posterior. Fica fora do caminho do satélite (``core/estimation.py`` e ``core/optimization.py``
são usados, não mudados).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.backtest import CarteiraRecusada, Estrategia, _cfg_com_rf_da_data
from core.config import ConfigOtimizacao, limites_por_setor
from core.estimation import estimar_momentos
from core.optimization import otimizar
from core.risco import LIMITES, Limites, checar_carteira, emissor
from core.valor import NotaDoMes

TAU = 0.05                  # incerteza do prior; com Ω proporcional a τ, ele se cancela na média posterior
MESES_MINIMOS_DE_IC = 36    # histórico de IC para um sinal entrar nas views
DELTA = 2.0                 # aversão do prior de equilíbrio = a do Kelly λ=2 (sem views, o Kelly volta ao mercado)


@dataclass(frozen=True)
class View:
    q: pd.Series        # retorno anual em excesso que a view diz para cada papel
    omega: pd.Series    # variância da view (diagonal)


def view_do_sinal(pi: pd.Series, cov: pd.DataFrame, z: pd.Series, ic: float, tau: float = TAU) -> View:
    """
    A view absoluta de um sinal (P = I): Q = Π + σ z, um desvio-padrão por unidade de escore normal, sem contração; a
    contração vem de Ω = τ σ² (1 - IC)/IC, que dá à view o peso IC (com Σ diagonal, posterior - Π = IC σ z, o alfa de
    Grinold). Papel sem escore fica com z = 0: a view diz "sem alfa" com a mesma confiança.
    """
    variancias = pd.Series(np.diag(cov.to_numpy()), index=cov.index).reindex(pi.index)
    return View(pi + np.sqrt(variancias) * z.reindex(pi.index).fillna(0.0), tau * variancias * (1.0 - ic) / ic)


def posterior(pi: pd.Series, cov: pd.DataFrame, views: Sequence[View], tau: float = TAU) -> pd.Series:
    """Média posterior de Black-Litterman com views absolutas: [(τΣ)⁻¹ + Σ Ω⁻¹]⁻¹ [(τΣ)⁻¹ Π + Σ Ω⁻¹ Q]."""
    precisao_prior = np.linalg.inv(tau * cov.loc[pi.index, pi.index].to_numpy())
    matriz, vetor = precisao_prior.copy(), precisao_prior @ pi.to_numpy()
    for view in views:
        precisao = 1.0 / view.omega.reindex(pi.index).to_numpy()
        matriz += np.diag(precisao)
        vetor += precisao * view.q.reindex(pi.index).to_numpy()
    return pd.Series(np.linalg.solve(matriz, vetor), index=pi.index)


def escore_normal(nota: pd.Series) -> pd.Series:
    """Φ⁻¹((posto - 0,5)/n) entre os papéis com nota (empate com o posto médio): a nota de cada sinal vira um escore
    normal padrão, robusto a extremos e comparável entre sinais; sem nota fica vazio."""
    postos = nota.rank()
    return pd.Series(norm.ppf((postos - 0.5) / postos.count()), index=nota.index)


def retorno_equilibrio(cov: pd.DataFrame, pesos_mercado: pd.Series, delta: float) -> pd.Series:
    """Π = δ Σ w_mercado: o retorno em excesso que torna os pesos de mercado ótimos para a aversão ``delta``."""
    pesos = pesos_mercado.reindex(cov.index).fillna(0.0)
    return pd.Series(delta * cov.to_numpy() @ pesos.to_numpy(), index=cov.index)


def ic_ponto_no_tempo(ics: pd.Series, mes: pd.Timestamp, minimo: int = MESES_MINIMOS_DE_IC) -> float | None:
    """IC médio dos meses antes de ``mes`` (o de m-1 usa o retorno de m-1, já fechado na decisão de m); None com menos
    de ``minimo`` meses ou IC médio <= 0: o sinal fica fora das views naquele mês."""
    passado = ics[ics.index < mes].dropna()
    if len(passado) < minimo:
        return None
    media = float(passado.mean())
    return media if media > 0.0 else None


def pesos_de_mercado(valor_mercado: pd.Series, papeis: Sequence[str]) -> pd.Series:
    """Pesos de mercado nos ``papeis`` pelo valor de mercado do emissor (``valor_mercado``: emissor -> valor; o universo
    do lago tem uma classe por emissor), somando 1 nos que têm valor; papel sem valor de mercado fica com 0 (o Π dele
    sai só das covariâncias com os outros)."""
    valores = pd.Series([valor_mercado.get(emissor(papel), np.nan) for papel in papeis], index=list(papeis))
    return (valores / valores.sum()).fillna(0.0)


@dataclass(frozen=True)
class Sinal:
    nota_do_mes: NotaDoMes      # (mês, candidatos) -> nota por papel, só com dado anterior ao mês
    ics: pd.Series              # IC de Spearman mensal do sinal (índice: 1º dia do mês)


def mu_com_views(cov: pd.DataFrame, papeis: Sequence[str], mes: pd.Timestamp, valor_mercado: pd.Series,
                 sinais: Sequence[Sinal], delta: float, tau: float = TAU) -> pd.Series:
    """
    μ anual em excesso dos ``papeis`` na decisão de ``mes``: o prior Π = δ Σ w de mercado (``valor_mercado``: emissor
    -> valor na decisão) e uma view por sinal com ``ic_ponto_no_tempo`` (o que não tem histórico ou tem IC médio <= 0
    fica fora), com o escore normal da nota do mês nos ``papeis``.
    """
    pi = retorno_equilibrio(cov, pesos_de_mercado(valor_mercado, papeis), delta)
    views = []
    for sinal in sinais:
        ic = ic_ponto_no_tempo(sinal.ics, mes)
        if ic is not None:
            views.append(view_do_sinal(pi, cov, escore_normal(sinal.nota_do_mes(mes, list(papeis))), ic, tau))
    return posterior(pi, cov, views, tau)


def nota_media_dos_escores(*notas_do_mes: NotaDoMes) -> NotaDoMes:
    """Um sinal só a partir de sinais correlacionados (a Fórmula Mágica e o Trending Value: duas views contariam o
    mesmo alfa duas vezes): a média dos escores normais de cada um nos candidatos, com os que houver."""
    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        escores = pd.DataFrame({posicao: escore_normal(nota_do_mes(mes, candidatos))
                                for posicao, nota_do_mes in enumerate(notas_do_mes)})
        return escores.reindex(list(candidatos)).mean(axis=1).dropna()
    return _nota


def estrategia_com_views(cfg_otimizacao: ConfigOtimizacao, setores: Mapping[str, str], sinais: Sequence[Sinal],
                         valor_mercado: pd.DataFrame, delta: float = DELTA, tau: float = TAU,
                         rf_diaria: pd.Series | None = None, limites: Limites = LIMITES) -> Estrategia:
    """
    A ``estrategia_otimizada_nas_regras`` com o μ trocado: na decisão (último pregão do treino, mês m), só papéis com
    setor, a cov de Ledoit-Wolf do treino e o ``mu_com_views`` do mês m (``valor_mercado``: ``issuer``, ``month``,
    ``market_cap`` do lago), teto de setor no otimizador e a carteira conferida por ``checar_carteira``; violação =
    ``CarteiraRecusada`` (o motor mantém a atual).
    """
    meses = pd.to_datetime(valor_mercado["month"])

    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        treino = treino[[papel for papel in treino.columns if setores.get(papel)]]
        mes = treino.index[-1].to_period("M").to_timestamp()
        momentos = estimar_momentos(treino, shrink_mu=False)
        do_mes = valor_mercado[meses == mes].set_index("issuer")["market_cap"]
        mu = mu_com_views(momentos.cov, list(treino.columns), mes, do_mes, sinais, delta, tau)
        grupos = limites_por_setor({papel: setores[papel] for papel in treino.columns}, limites.teto_setor)
        cfg = replace(_cfg_com_rf_da_data(cfg_otimizacao, rf_diaria, treino.index[-1]), grupos=grupos)
        pesos = otimizar(mu, momentos.cov, cfg).pesos
        violacoes = checar_carteira(pesos, setores, momentos.cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia
