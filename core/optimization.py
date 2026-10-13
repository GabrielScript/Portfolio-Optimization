"""
optimization.py - Otimização convexa de carteiras long-only (cvxpy).

Conjunto viável comum a todos os objetivos baseados em mu:
    W = { w : 1'w = 1, 0 <= w_i <= u, sum_{i in G} w_i <= c_G, ||L'w|| <= vol_max }
com Sigma = L L' (Cholesky). O teto de vol entra como cone de segunda ordem,
melhor condicionado que w'Sigma w <= vol^2.

Objetivos:
- KELLY: max mu'w - (lambda/2) w'Sigma w. Com lambda = 1, aproxima por
  segunda ordem a carteira de máximo crescimento geométrico (Kelly) dentro de
  W. A aproximação vale na escala diária, e o argmax é invariante à
  anualização. Sem ativo livre de risco em W, lambda > 1 NÃO é "fração de
  Kelly" no sentido clássico (não há caixa para diluir): é só aversão a risco
  maior. É o objetivo racional do investidor agressivo, porque ir além de
  Kelly (o que MAX_RETORNO faz) aumenta o risco E reduz o crescimento
  esperado.
- MAX_SHARPE: exato via transformação de Charnes-Cooper/Cornuejols-Tütüncü,
  um único QP convexo (o código anterior varria 50 pontos da fronteira). Só é
  definido se alguma carteira viável tem mu'w > rf. Com Selic de dois
  dígitos isso falha com frequência na B3, e então caímos para MIN_VOL com
  aviso explícito.
- MAX_RETORNO: max mu'w sujeito ao teto de vol. Programa quase linear:
  solução de canto que concentra no teto de peso dos ativos de maior mu, ou
  seja, maximiza o erro de estimação. Mantido só para comparação.
- MIN_VOL: min w'Sigma w.

Cardinalidade (no máximo k ativos) é NP-difícil (MIQP). Heurística em duas
etapas: resolve o relaxado, mantém os k maiores pesos e resolve de novo
restrito a eles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from core.config import ConfigOtimizacao, Objetivo
from core.hierarchical import otimizar_hrp

logger = logging.getLogger(__name__)

LIMIAR_PESO = 1e-4               # pesos abaixo disso são ruído numérico do solver
_STATUS_OK = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
_SOLVERS = ("CLARABEL", "SCS")   # CLARABEL: QP + SOCP, ponto interior; SCS: último recurso


class ErroOtimizacao(RuntimeError):
    """O solver não encontrou solução."""


class ErroInviavel(ErroOtimizacao):
    """As restrições não admitem nenhuma carteira."""


@dataclass(frozen=True)
class ResultadoOtimizacao:
    pesos: pd.Series
    retorno_esperado: float
    volatilidade: float
    sharpe: float
    objetivo: Objetivo
    status: str          # "otimo" | "impreciso" | "fallback"
    mensagem: str

    @property
    def tem_aviso(self) -> bool:
        return self.status != "otimo"


@dataclass(frozen=True)
class _Solucao:
    pesos: np.ndarray
    status: str
    mensagem: str


def preparar_covariancia(cov: pd.DataFrame) -> np.ndarray:
    """Simetriza e aplica ridge mínimo para garantir Cholesky (Sigma > 0)."""
    matriz = np.asarray(cov.values, dtype=float)
    matriz = (matriz + matriz.T) / 2.0
    ridge = 1e-10 * max(float(np.trace(matriz)) / len(matriz), 1e-12)
    return matriz + ridge * np.eye(len(matriz))


def _resolver(problema: cp.Problem, variavel: cp.Variable) -> tuple[np.ndarray, str]:
    ultimo_erro = "nenhum solver executado"
    for solver in _SOLVERS:
        try:
            problema.solve(solver=solver)
        except cp.error.SolverError as erro:
            ultimo_erro = f"{solver}: {erro}"
            continue
        if problema.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
            raise ErroInviavel(f"Restrições inviáveis ({solver}).")
        if problema.status in _STATUS_OK and variavel.value is not None:
            status = "otimo" if problema.status == cp.OPTIMAL else "impreciso"
            return np.asarray(variavel.value, dtype=float).ravel(), status
        ultimo_erro = f"{solver}: status {problema.status}"
    raise ErroOtimizacao(ultimo_erro)


def _limpar_pesos(pesos: np.ndarray) -> np.ndarray:
    limpos = np.where(pesos < LIMIAR_PESO, 0.0, pesos)
    total = limpos.sum()
    if total <= 0:
        raise ErroOtimizacao("Solver devolveu vetor de pesos nulo.")
    return limpos / total


def _restricoes(w: cp.Variable, tickers: list[str], cfg: ConfigOtimizacao,
                escala: cp.Variable | float = 1.0) -> list:
    """Restrições lineares homogêneas em (w, escala). Para Charnes-Cooper, escala = kappa."""
    restricoes = [cp.sum(w) == escala, w >= 0, w <= cfg.peso_maximo * escala]
    for grupo in cfg.grupos:
        indices = [i for i, t in enumerate(tickers) if t in grupo.tickers]
        if indices:
            restricoes.append(cp.sum(w[indices]) <= grupo.teto * escala)
    return restricoes


def _checar_capacidade(n_ativos: int, cfg: ConfigOtimizacao) -> None:
    if n_ativos * cfg.peso_maximo < 1.0 - 1e-9:
        raise ErroInviavel(
            f"{n_ativos} ativos x teto {cfg.peso_maximo:.0%} não somam 100%."
        )


def _min_vol(mu: np.ndarray, cov: np.ndarray, tickers: list[str],
             cfg: ConfigOtimizacao) -> _Solucao:
    w = cp.Variable(len(mu))
    problema = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))),
                          _restricoes(w, tickers, cfg))
    pesos, status = _resolver(problema, w)
    return _Solucao(pesos, status, "Mínima variância.")


def _com_teto_vol(objetivo_fn: Callable[[cp.Variable], cp.Expression], mu: np.ndarray,
                  cov: np.ndarray, tickers: list[str], cfg: ConfigOtimizacao,
                  rotulo: str) -> _Solucao:
    """Maximiza objetivo_fn(w) em W com teto de vol; se o teto for inviável, cai para MIN_VOL."""
    w = cp.Variable(len(mu))
    restricoes = _restricoes(w, tickers, cfg)
    if cfg.vol_maxima is not None:
        restricoes.append(cp.norm(np.linalg.cholesky(cov).T @ w, 2) <= cfg.vol_maxima)
    try:
        pesos, status = _resolver(cp.Problem(cp.Maximize(objetivo_fn(w)), restricoes), w)
        return _Solucao(pesos, status, rotulo)
    except ErroInviavel:
        if cfg.vol_maxima is None:
            raise
        fallback = _min_vol(mu, cov, tickers, cfg)
        vol_min = float(np.sqrt(fallback.pesos @ cov @ fallback.pesos))
        return _Solucao(fallback.pesos, "fallback",
                        f"Teto de vol {cfg.vol_maxima:.0%} inviável (mínima possível "
                        f"{vol_min:.1%}); usada a carteira de mínima variância.")


def _kelly(mu: np.ndarray, cov: np.ndarray, tickers: list[str],
           cfg: ConfigOtimizacao) -> _Solucao:
    risco = cp.psd_wrap(cov)
    return _com_teto_vol(
        lambda w: mu @ w - (cfg.aversao_risco / 2.0) * cp.quad_form(w, risco),
        mu, cov, tickers, cfg, f"Crescimento (Kelly, lambda={cfg.aversao_risco:g}).")


def _max_retorno(mu: np.ndarray, cov: np.ndarray, tickers: list[str],
                 cfg: ConfigOtimizacao) -> _Solucao:
    return _com_teto_vol(lambda w: mu @ w, mu, cov, tickers, cfg, "Máximo retorno.")


def _max_sharpe(mu: np.ndarray, cov: np.ndarray, tickers: list[str],
                cfg: ConfigOtimizacao) -> _Solucao:
    """
    Charnes-Cooper: min y'Sigma y  s.a. (mu - rf)'y = 1, 1'y = kappa, 0 <= y <= u*kappa,
    grupos <= c*kappa, ||L'y|| <= vol_max*kappa, kappa >= 0;  w* = y*/kappa*.
    """
    rf = cfg.taxa_livre_risco
    w_teste = cp.Variable(len(mu))
    _, _ = _resolver(cp.Problem(cp.Maximize(mu @ w_teste), _restricoes(w_teste, tickers, cfg)), w_teste)
    retorno_max = float(mu @ w_teste.value)
    if retorno_max <= rf + 1e-10:
        fallback = _min_vol(mu, cov, tickers, cfg)
        return _Solucao(fallback.pesos, "fallback",
                        f"Nenhuma carteira viável supera a taxa livre de risco "
                        f"({retorno_max:.1%} <= {rf:.1%} a.a.): Sharpe máximo indefinido. "
                        f"Usada a mínima variância; a renda fixa domina no histórico.")

    y = cp.Variable(len(mu))
    kappa = cp.Variable(nonneg=True)
    restricoes = _restricoes(y, tickers, cfg, escala=kappa) + [(mu - rf) @ y == 1]
    if cfg.vol_maxima is not None:
        restricoes.append(cp.norm(np.linalg.cholesky(cov).T @ y, 2) <= cfg.vol_maxima * kappa)
    problema = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov))), restricoes)
    try:
        y_otimo, status = _resolver(problema, y)
    except ErroInviavel:
        fallback = _min_vol(mu, cov, tickers, cfg)
        return _Solucao(fallback.pesos, "fallback",
                        "Nenhuma carteira com Sharpe positivo respeita o teto de vol; "
                        "usada a mínima variância.")
    return _Solucao(y_otimo / float(kappa.value), status, "Máximo Sharpe (Charnes-Cooper, exato).")


_SOLUCIONADORES = {
    Objetivo.KELLY: _kelly,
    Objetivo.MAX_SHARPE: _max_sharpe,
    Objetivo.MAX_RETORNO: _max_retorno,
    Objetivo.MIN_VOL: _min_vol,
}


def _resolver_com_cardinalidade(mu: pd.Series, cov: pd.DataFrame,
                                cfg: ConfigOtimizacao) -> _Solucao:
    solucionador = _SOLUCIONADORES[cfg.objetivo]
    tickers = mu.index.tolist()
    cov_np = preparar_covariancia(cov)
    _checar_capacidade(len(tickers), cfg)

    etapa1 = solucionador(mu.values, cov_np, tickers, cfg)
    pesos = _limpar_pesos(etapa1.pesos)
    k = cfg.n_ativos_max
    if k is None or int((pesos > 0).sum()) <= k:
        return _Solucao(pesos, etapa1.status, etapa1.mensagem)

    selecionados = np.argsort(pesos)[::-1][:k]
    sub_tickers = [tickers[i] for i in selecionados]
    try:
        etapa2 = solucionador(mu.values[selecionados], cov_np[np.ix_(selecionados, selecionados)],
                              sub_tickers, cfg)
    except ErroOtimizacao as erro:
        return _Solucao(pesos, "fallback",
                        f"{etapa1.mensagem} Limite de {k} ativos não aplicado: {erro}")
    pesos_finais = np.zeros(len(tickers))
    pesos_finais[selecionados] = _limpar_pesos(etapa2.pesos)
    return _Solucao(pesos_finais, etapa2.status,
                    f"{etapa2.mensagem} Cardinalidade <= {k} (heurística 2 etapas).")


def metricas_carteira(pesos: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                      taxa_livre_risco: float) -> tuple[float, float, float]:
    """(retorno esperado, volatilidade, Sharpe) anualizados."""
    retorno = float(pesos @ mu)
    vol = float(np.sqrt(max(pesos @ cov @ pesos, 0.0)))
    sharpe = (retorno - taxa_livre_risco) / vol if vol > 0 else 0.0
    return retorno, vol, sharpe


def otimizar(mu: pd.Series, cov: pd.DataFrame, cfg: ConfigOtimizacao) -> ResultadoOtimizacao:
    """
    Ponto de entrada. ``mu``/``cov`` anualizados (aritméticos) e
    ``cfg.taxa_livre_risco`` na mesma convenção (ver config.taxa_anual_para_aritmetica).

    Lança ErroInviavel/ErroOtimizacao. Não devolve pesos zerados em silêncio.
    """
    if not mu.index.equals(cov.index) or not cov.index.equals(cov.columns):
        raise ValueError("mu e cov precisam ter os mesmos tickers na mesma ordem.")
    if mu.isna().any() or cov.isna().any().any():
        raise ValueError("mu/cov contêm NaN.")

    if cfg.objetivo is Objetivo.HRP:
        pesos = otimizar_hrp(cov).reindex(mu.index).fillna(0.0).values
        solucao = _Solucao(pesos, "otimo",
                           "HRP (López de Prado): ignora mu, teto de peso e cardinalidade.")
    else:
        solucao = _resolver_com_cardinalidade(mu, cov, cfg)

    cov_np = preparar_covariancia(cov)
    retorno, vol, sharpe = metricas_carteira(solucao.pesos, mu.values, cov_np, cfg.taxa_livre_risco)
    return ResultadoOtimizacao(
        pesos=pd.Series(solucao.pesos, index=mu.index),
        retorno_esperado=retorno, volatilidade=vol, sharpe=sharpe,
        objetivo=cfg.objetivo, status=solucao.status, mensagem=solucao.mensagem,
    )


def fronteira_eficiente(mu: pd.Series, cov: pd.DataFrame, cfg: ConfigOtimizacao,
                        n_pontos: int = 30) -> pd.DataFrame:
    """
    Fronteira média-variância com os mesmos tetos de peso e de grupo, mas SEM
    cardinalidade e sem teto de vol. Ela é o limite superior teórico: a
    carteira com cardinalidade pode ficar abaixo dela.
    """
    tickers = mu.index.tolist()
    cov_np = preparar_covariancia(cov)
    _checar_capacidade(len(tickers), cfg)
    base = _min_vol(mu.values, cov_np, tickers, cfg)
    w = cp.Variable(len(mu))
    _resolver(cp.Problem(cp.Maximize(mu.values @ w), _restricoes(w, tickers, cfg)), w)
    retorno_min, retorno_max = float(base.pesos @ mu.values), float(mu.values @ w.value)

    alvo = cp.Parameter()
    problema = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_np))),
                          _restricoes(w, tickers, cfg) + [mu.values @ w >= alvo])
    pontos = []
    for retorno_alvo in np.linspace(retorno_min, retorno_max, n_pontos):
        alvo.value = retorno_alvo
        try:
            pesos, _ = _resolver(problema, w)
        except ErroOtimizacao:
            continue
        retorno, vol, sharpe = metricas_carteira(pesos, mu.values, cov_np, cfg.taxa_livre_risco)
        pontos.append({"retorno": retorno, "volatilidade": vol, "sharpe": sharpe})
    return pd.DataFrame(pontos)
