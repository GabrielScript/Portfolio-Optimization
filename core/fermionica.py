"""
fermionica.py - Fermionic (determinantal) selection on a score of the month
(docs/superpowers/specs/2026-11-11-fermionic-portfolio-design.md). Research only: the app, the paper logs and the
screen do not read this.

Names are free fermions: the probability of a set is a determinant (a k-DPP with kernel
L = diag(e^{theta z}) C diag(e^{theta z})), so similar names repel. The picks maximize

    F(S) = 2 theta sum_{i in S} z_i + log det C_S          (|S| = k, group caps)

z the score's cross-sectional z-score, C the return correlation. log det C_S <= 0 is minus twice the Gaussian total
correlation of the set: redundancy of every order, not only of pairs. theta -> infinity is the ranking by score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.estimation import estimar_momentos
from core.qpo import candidates
from core.qubo_universo import FULL_UNIVERSE, GroupCaps
from core.risco import LIMITES, Limites, checar_carteira
from core.sinais import NOMES, POR_SETOR
from core.valor import NotaDoMes

THETA = 1.0            # pre-registered: 0.2 of z-score is traded for removing a correlation of ~0.57 with the set
MIN_GAIN = 1e-12       # a swap must raise F by more than this


def objective(z: np.ndarray, corr: np.ndarray, picks: Sequence[int], theta: float) -> float:
    """F(S) = 2 theta sum z_S + log det C_S; -inf when C_S is singular."""
    indices = list(picks)
    sinal, log_det = np.linalg.slogdet(corr[np.ix_(indices, indices)])
    return float(2.0 * theta * z[indices].sum() + log_det) if sinal > 0 else -np.inf


def _admissible(grupos: np.ndarray | None, teto: int, picks: list[int], n: int,
                saindo: int | None = None) -> np.ndarray:
    """Names outside ``picks`` whose group still has room once ``saindo`` leaves."""
    livres = np.ones(n, dtype=bool)
    livres[picks] = False
    if grupos is not None:
        ficam = [i for i in picks if i != saindo]
        nomes, quantos = np.unique(grupos[ficam], return_counts=True)
        livres &= ~np.isin(grupos, nomes[quantos >= teto])
    return np.flatnonzero(livres)


def _gains(z: np.ndarray, corr: np.ndarray, base: list[int], entram: np.ndarray, theta: float) -> np.ndarray:
    """F(base + j) - F(base) for each j: 2 theta z_j + log of j's variance unexplained by ``base`` (Schur)."""
    if not base:
        return 2.0 * theta * z[entram]
    cruzada = corr[np.ix_(base, entram)]
    resto = 1.0 - np.einsum("ij,ij->j", cruzada, np.linalg.solve(corr[np.ix_(base, base)], cruzada))
    return 2.0 * theta * z[entram] + np.log(np.maximum(resto, 1e-300))


def _greedy(z: np.ndarray, corr: np.ndarray, k: int, grupos: np.ndarray | None, teto: int, theta: float,
            por_ganho: bool) -> list[int]:
    """k names one at a time under the caps: by marginal gain of F, or by score alone (the plain ranking)."""
    picks: list[int] = []
    while len(picks) < k:
        livres = _admissible(grupos, teto, picks, len(z))
        if not len(livres):
            raise ValueError(f"the group caps admit fewer than k = {k} of the {len(z)} names")
        valores = _gains(z, corr, picks, livres, theta) if por_ganho else z[livres]
        picks.append(int(livres[np.argmax(valores)]))
    return picks


def _swap_ascent(z: np.ndarray, corr: np.ndarray, picks: list[int], grupos: np.ndarray | None, teto: int,
                 theta: float) -> list[int]:
    """Steepest single swap (one out, one in, caps kept) until none raises F by more than ``MIN_GAIN``."""
    atual = objective(z, corr, picks, theta)
    while True:
        melhor, troca = atual + MIN_GAIN, None
        for sai in picks:
            base = [i for i in picks if i != sai]
            entram = _admissible(grupos, teto, picks, len(z), saindo=sai)
            if not len(entram):
                continue
            valores = objective(z, corr, base, theta) + _gains(z, corr, base, entram, theta)
            if valores.max() > melhor:
                melhor, troca = float(valores.max()), (sai, int(entram[np.argmax(valores)]))
        if troca is None:
            return picks
        picks = [i for i in picks if i != troca[0]] + [troca[1]]
        atual = objective(z, corr, picks, theta)


def fermionic_picks(z: np.ndarray, corr: np.ndarray, k: int, caps: GroupCaps | None, theta: float) -> tuple[int, ...]:
    """argmax F over |S| = k under the caps by swap ascent from two starts (the ranking by score, the greedy by
    gain); the higher F wins, ties to the ranking's. Deterministic."""
    z = np.asarray(z, dtype=float)
    grupos = np.asarray(caps.groups) if caps is not None else None
    teto = caps.max_per_group if caps is not None else len(z)
    finais = [_swap_ascent(z, corr, _greedy(z, corr, k, grupos, teto, theta, por_ganho), grupos, teto, theta)
              for por_ganho in (False, True)]
    return tuple(sorted(max(finais, key=lambda picks: objective(z, corr, picks, theta))))


def estrategia_fermionica(nota: NotaDoMes, setores: Mapping[str, str], theta: float = THETA, nomes: int = NOMES,
                          por_setor: int = POR_SETOR, limites: Limites = LIMITES,
                          registro: dict | None = None) -> Estrategia:
    """Equal weights on the fermionic picks among every scored candidate (one class per issuer, with a sector);
    the house rules as in ``core.composto.estrategia_por_nota``. ``registro`` keeps (tickers, picks, z, corr) per
    decision date."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        score = nota(mes, list(treino.columns)).dropna()
        candidatos = candidates(score, setores, FULL_UNIVERSE)
        if not candidatos:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no eligible name with a score")
        escolhidos = candidatos
        if len(candidatos) > nomes:
            cov = estimar_momentos(treino[candidatos], shrink_mu=False).cov.to_numpy()
            vol = np.sqrt(np.diag(cov))
            corr = cov / np.outer(vol, vol)
            valores = score[candidatos].to_numpy(dtype=float)
            desvio = valores.std()
            z = (valores - valores.mean()) / desvio if desvio > 0 else np.zeros(len(valores))
            picks = fermionic_picks(z, corr, nomes, GroupCaps(tuple(setores[p] for p in candidatos), por_setor), theta)
            escolhidos = [candidatos[i] for i in picks]
            if registro is not None:
                registro[treino.index[-1]] = (tuple(candidatos), picks, z, corr)
        pesos = pd.Series(1.0 / len(escolhidos), index=escolhidos)
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[escolhidos], shrink_mu=False).cov, limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia
