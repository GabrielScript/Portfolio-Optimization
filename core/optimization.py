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
from dataclasses import dataclass, replace
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from core.config import ConfigOtimizacao, LimiteGrupo, Objetivo
from core.hierarchical import otimizar_hrp

logger = logging.getLogger(__name__)

LIMIAR_PESO = 1e-4               # pesos abaixo disso são ruído numérico do solver
_FOLGA_MINIMA = 1e-12            # folga (ou falta) menor que isso já é zero ao encher até os tetos
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


def encher_ate_os_tetos(pesos: pd.Series, total: float, peso_maximo: float,
                        grupos: tuple[LimiteGrupo, ...]) -> pd.Series:
    """
    Leva ``pesos`` até somar ``total``, dando o que falta a cada ativo na proporção do peso dele, só até o teto por
    ativo e o teto de cada grupo. O que não couber fica de fora (soma < ``total``). Um peso que já passava de um teto
    não é cortado, só deixa de receber. Cada rodada enche pelo menos um teto, então termina.
    """
    alvo = pesos.astype(float).copy()
    membros = np.array([alvo.index.isin(grupo.tickers) for grupo in grupos],
                       dtype=bool).reshape(len(grupos), len(alvo))
    tetos = np.array([grupo.teto for grupo in grupos])
    for _ in range(len(alvo) + len(grupos) + 1):
        falta = total - float(alvo.sum())
        folga_ativo = np.clip(peso_maximo - alvo.to_numpy(), 0.0, None)
        folga_grupo = np.clip(tetos - membros @ alvo.to_numpy(), 0.0, None)
        livre = np.minimum(folga_ativo, np.where(membros, folga_grupo[:, None], np.inf).min(axis=0, initial=np.inf))
        abertos = (livre > _FOLGA_MINIMA) & (alvo.to_numpy() > 0)
        if falta <= _FOLGA_MINIMA or not abertos.any():
            break
        proposta = np.where(abertos, alvo.to_numpy(), 0.0)
        proposta = falta * proposta / proposta.sum()
        # a maior fração da proposta que não fura nenhum teto (por ativo ou pela soma do grupo)
        por_grupo = membros @ proposta
        fator = min(1.0, float(np.min(livre[abertos] / proposta[abertos])),
                    float(np.min(folga_grupo[por_grupo > 0] / por_grupo[por_grupo > 0], initial=np.inf)))
        alvo += fator * proposta
    return alvo


def ajustar_aos_tetos(pesos: pd.Series, cfg: ConfigOtimizacao) -> pd.Series:
    """
    Tira o ruído do solver acima dos tetos (zerar pesos < ``LIMIAR_PESO`` e renormalizar empurra quem está no teto
    para cima em ~1e-6): corta no teto por ativo e no de cada grupo e devolve o corte a quem tem folga.
    """
    cortados = pesos.astype(float).clip(upper=cfg.peso_maximo)
    for grupo in cfg.grupos:
        membros = cortados.index.isin(grupo.tickers)
        soma = float(cortados[membros].sum())
        if soma > grupo.teto:
            cortados[membros] *= grupo.teto / soma
    return encher_ate_os_tetos(cortados, float(pesos.sum()), cfg.peso_maximo, cfg.grupos)


def combinar_nucleo_satelite(nucleo: pd.Series, satelite: pd.Series, fracao_satelite: float) -> pd.Series:
    """Pesos de núcleo + satélite: (1 − f)·núcleo + f·satélite. Tetos lineares e de vol valem na mistura de viáveis."""
    if not 0.0 <= fracao_satelite <= 1.0:
        raise ValueError(f"fracao_satelite fora de [0, 1]: {fracao_satelite}")
    tickers = nucleo.index.union(satelite.index, sort=False)
    return ((1.0 - fracao_satelite) * nucleo.reindex(tickers, fill_value=0.0)
            + fracao_satelite * satelite.reindex(tickers, fill_value=0.0))


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


def _checar_capacidade_grupos(tickers: list[str], cfg: ConfigOtimizacao) -> None:
    """
    O máximo que os ativos somam com os tetos: sum_G min(c_G, n_G u) + n_fora u. Exato para
    grupos disjuntos (setores). Com sobreposição é só um limite superior, e o solver decide.
    """
    agrupados: set[str] = set()
    capacidade = 0.0
    for grupo in cfg.grupos:
        membros = grupo.tickers.intersection(tickers)
        agrupados |= membros
        capacidade += min(grupo.teto, len(membros) * cfg.peso_maximo)
    capacidade += len(set(tickers) - agrupados) * cfg.peso_maximo
    if capacidade < 1.0 - 1e-9:
        raise ErroInviavel(
            f"Com teto de {cfg.peso_maximo:.0%} por ativo e os tetos de grupo, os {len(tickers)} "
            f"ativos somam no máximo {capacidade:.0%}."
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
    _checar_capacidade_grupos(tickers, cfg)

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


@dataclass(frozen=True)
class PesosProjetados:
    pesos: pd.Series
    status: str          # "otimo" | "impreciso" | "fallback"
    mensagem: str


def projetar_pesos(alvo: pd.Series, cov: pd.DataFrame, cfg: ConfigOtimizacao) -> PesosProjetados:
    """
    Carteira viável mais próxima de ``alvo`` na norma euclidiana:

        min ||w - alvo_S||^2  s.a.  w em W restrito a S,  ||L'w|| <= vol_max

    S são os k maiores pesos de ``alvo`` (k = cfg.n_ativos_max; sem k, todos).
    Uso: a média de carteiras viáveis (reamostragem) já satisfaz as restrições
    lineares de W, que formam um conjunto convexo, mas pode ter mais de k
    ativos e romper o teto de vol sob ``cov``, porque cada reamostra o
    respeitou sob a própria covariância. Status "fallback" em dois casos:
    - teto de vol inviável em S: projeta só nas restrições lineares;
    - restrições lineares inviáveis em S (ex.: um teto de grupo com membros
      entre os k maiores): projeta em todos os ativos, sem o limite de k.
      Isso sempre tem solução quando o alvo é média de carteiras viáveis.
    """
    if not alvo.index.equals(cov.index) or not cov.index.equals(cov.columns):
        raise ValueError("alvo e cov precisam ter os mesmos tickers na mesma ordem.")
    k = len(alvo) if cfg.n_ativos_max is None else min(cfg.n_ativos_max, len(alvo))
    _checar_capacidade(k, cfg)
    try:
        return _projetar_no_suporte(alvo, cov, cfg, k)
    except ErroInviavel:
        if k == len(alvo):
            raise
        projecao = _projetar_no_suporte(alvo, cov, cfg, len(alvo))
        mensagem = (f"Limite de {k} ativos não aplicado: os {k} maiores pesos não cabem nos tetos "
                    f"por ativo e de grupo. {projecao.mensagem}").strip()
        return PesosProjetados(projecao.pesos, "fallback", mensagem)


def _projetar_no_suporte(alvo: pd.Series, cov: pd.DataFrame, cfg: ConfigOtimizacao,
                         k: int) -> PesosProjetados:
    suporte = np.argsort(-alvo.values, kind="stable")[:k]
    tickers = [alvo.index[i] for i in suporte]
    w = cp.Variable(k)
    objetivo = cp.Minimize(cp.sum_squares(w - alvo.values[suporte]))
    lineares = _restricoes(w, tickers, cfg)

    status, mensagem = "otimo", ""
    if cfg.vol_maxima is None:
        pesos, status = _resolver(cp.Problem(objetivo, lineares), w)
    else:
        cholesky = np.linalg.cholesky(preparar_covariancia(cov.iloc[suporte, suporte]))
        try:
            pesos, status = _resolver(
                cp.Problem(objetivo, lineares + [cp.norm(cholesky.T @ w, 2) <= cfg.vol_maxima]), w)
        except ErroInviavel:
            pesos, _ = _resolver(cp.Problem(objetivo, lineares), w)
            status = "fallback"
            mensagem = (f"Teto de vol {cfg.vol_maxima:.0%} inviável com os {k} ativos escolhidos; "
                        "projeção feita sem ele.")
    finais = np.zeros(len(alvo))
    finais[suporte] = _limpar_pesos(pesos)
    return PesosProjetados(ajustar_aos_tetos(pd.Series(finais, index=alvo.index), cfg), status, mensagem)


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
        solucao = replace(solucao, pesos=ajustar_aos_tetos(pd.Series(solucao.pesos, index=mu.index), cfg).to_numpy())

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
