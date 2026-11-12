"""
fermionica.py - Determinantal (fermionic) and entropic selection on a score of the month
(docs/superpowers/specs/2026-11-11-fermionic-portfolio-design.md and 2026-11-11-fermionic-v2-design.md). Research,
plus the rule of the "Fermiônica" fictitious account (``core.fermionica_fantasma``): it must import on the app's
requirements, which have no PySCIPOpt.

The picks maximize, over |S| = k and the group caps,

    F(S) = 2 theta sum_{i in S} z_i + D(C_S)

z the score's cross-sectional z-score, C the return correlation, D a functional of ``core.diversidade``. With
D = log det, F is the log-probability of S under the k-DPP of free fermions with kernel
L = diag(e^{theta z}) C diag(e^{theta z}): similar names repel. theta -> infinity is the ranking by score.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.backtest import CarteiraRecusada, Estrategia
from core.diversidade import Diversity, eic_weights, log_det, shrunk_correlation
from core.estimation import estimar_momentos
from core.qpo import candidates
from core.risco import LIMITES, Limites, checar_carteira
from core.sinais import NOMES, POR_SETOR, carteira_da_nota
from core.valor import NotaDoMes

THETA = 1.0            # first reading: 0.2 of z-score is traded for removing a correlation of ~0.57 with the set
SWAP_TOL = 1e-9        # a swap must raise F by more than this, relative to max(1, |F|)
FULL_UNIVERSE = 10**6  # more names than the B3 lists: ``core.qpo.candidates`` keeps every eligible name
SIZES = (5, 8, 10, 12, 15, 20, 25, 30, 40, 50)      # the grand-canonical selection's sizes (5: the 20% name cap)

Correlation = Callable[[pd.DataFrame], np.ndarray]      # training-window returns of the candidates -> C


@dataclass(frozen=True)
class GroupCaps:
    """At most ``max_per_group`` picks share a group (the sector); ``groups`` is aligned with the candidates. Same
    fields as ``core.qubo_universo.GroupCaps``, which cannot be imported here (that module needs PySCIPOpt)."""
    groups: Sequence[str]
    max_per_group: int


def covariance_correlation(treino: pd.DataFrame) -> np.ndarray:
    """The first reading's C: the correlation of the Ledoit-Wolf covariance (deflates low-volatility names more)."""
    cov = estimar_momentos(treino, shrink_mu=False).cov.to_numpy()
    vol = np.sqrt(np.diag(cov))
    return cov / np.outer(vol, vol)


def _scores(z: np.ndarray, corr: np.ndarray, conjuntos: np.ndarray, theta: float, diversity: Diversity) -> np.ndarray:
    """F of each row of ``conjuntos`` (sets of equal size, as indices)."""
    pilha = corr[conjuntos[:, :, None], conjuntos[:, None, :]]
    return 2.0 * theta * z[conjuntos].sum(axis=1) + diversity(pilha)


def objective(z: np.ndarray, corr: np.ndarray, picks: Sequence[int], theta: float,
              diversity: Diversity = log_det) -> float:
    """F(S) = 2 theta sum z_S + D(C_S)."""
    return float(_scores(np.asarray(z, dtype=float), corr, np.asarray([list(picks)]), theta, diversity)[0])


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


def _greedy(z: np.ndarray, corr: np.ndarray, k: int, grupos: np.ndarray | None, teto: int, theta: float,
            diversity: Diversity | None) -> list[int]:
    """k names one at a time under the caps: by F of the enlarged set, or by score alone when ``diversity`` is None
    (the plain ranking; a tie goes to the earlier name, as in ``carteira_da_nota``)."""
    picks: list[int] = []
    while len(picks) < k:
        livres = _admissible(grupos, teto, picks, len(z))
        if not len(livres):
            raise ValueError(f"the group caps admit fewer than k = {k} of the {len(z)} names")
        if diversity is None:
            valores = z[livres]
        else:
            conjuntos = np.column_stack([np.tile(picks, (len(livres), 1)), livres]).astype(int)
            valores = _scores(z, corr, conjuntos, theta, diversity)
        picks.append(int(livres[np.argmax(valores)]))
    return picks


def _swap_ascent(z: np.ndarray, corr: np.ndarray, picks: list[int], grupos: np.ndarray | None, teto: int,
                 theta: float, diversity: Diversity) -> list[int]:
    """Steepest single swap (one out, one in, caps kept) until none raises F by more than the tolerance. Every F
    comes from ``_scores``, so a swap is never accepted on the rounding of two different formulas."""
    picks = list(picks)
    atual = objective(z, corr, picks, theta, diversity)
    while True:
        melhor, troca = atual + SWAP_TOL * max(1.0, abs(atual)), None
        for posicao, sai in enumerate(picks):
            entram = _admissible(grupos, teto, picks, len(z), saindo=sai)
            if not len(entram):
                continue
            conjuntos = np.tile(picks, (len(entram), 1))
            conjuntos[:, posicao] = entram
            valores = _scores(z, corr, conjuntos, theta, diversity)
            if valores.max() > melhor:
                melhor, troca = float(valores.max()), (posicao, int(entram[np.argmax(valores)]))
        if troca is None:
            return picks
        picks[troca[0]] = troca[1]
        atual = melhor


def fermionic_picks(z: np.ndarray, corr: np.ndarray, k: int, caps: GroupCaps | None, theta: float,
                    diversity: Diversity = log_det) -> tuple[int, ...]:
    """argmax F over |S| = k under the caps by swap ascent from two starts (the ranking by score, the greedy by F);
    the higher F wins, ties to the ranking's. Deterministic. ValueError when the caps admit fewer than k."""
    z = np.asarray(z, dtype=float)
    grupos = np.asarray(caps.groups) if caps is not None else None
    teto = caps.max_per_group if caps is not None else len(z)
    finais = [_swap_ascent(z, corr, _greedy(z, corr, k, grupos, teto, theta, inicio), grupos, teto, theta, diversity)
              for inicio in (None, diversity)]
    return tuple(sorted(max(finais, key=lambda picks: objective(z, corr, picks, theta, diversity))))


def _room_under_caps(candidatos: Sequence[str], setores: Mapping[str, str], por_setor: int) -> int:
    """The most names the caps admit among ``candidatos``."""
    return int(pd.Series([setores[papel] for papel in candidatos]).value_counts().clip(upper=por_setor).sum())


def estrategia_fermionica(nota: NotaDoMes, setores: Mapping[str, str], theta: float = THETA, nomes: int = NOMES,
                          por_setor: int = POR_SETOR, limites: Limites = LIMITES, registro: dict | None = None,
                          diversity: Diversity = log_det, correlation: Correlation = shrunk_correlation) -> Estrategia:
    """Equal weights on the picks among every scored candidate (one class per issuer, with a sector); the house
    rules as in ``core.composto.estrategia_por_nota``. When the candidates or the caps leave no choice (no more than
    ``nomes`` admissible) the portfolio is the Composite's ``carteira_da_nota``. ``registro`` keeps
    (tickers, picks, z, corr) per decision date with a choice."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        score = nota(mes, list(treino.columns)).dropna()
        candidatos = candidates(score, setores, FULL_UNIVERSE)
        if not candidatos:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no eligible name with a score")
        if _room_under_caps(candidatos, setores, por_setor) <= nomes:
            pesos = carteira_da_nota(score, setores, nomes, por_setor)
        else:
            corr = correlation(treino[candidatos])
            valores = score[candidatos].to_numpy(dtype=float)
            desvio = valores.std()
            z = (valores - valores.mean()) / desvio if desvio > 0 else np.zeros(len(valores))
            caps = GroupCaps(tuple(setores[papel] for papel in candidatos), por_setor)
            picks = fermionic_picks(z, corr, nomes, caps, theta, diversity)
            if registro is not None:
                registro[treino.index[-1]] = (tuple(candidatos), picks, z, corr)
            pesos = pd.Series(1.0 / nomes, index=[candidatos[i] for i in picks])
        violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[list(pesos.index)], shrink_mu=False).cov,
                                    limites)
        if violacoes:
            raise CarteiraRecusada(violacoes)
        return pesos
    return _estrategia


def names_per_sector(nomes: int) -> int:
    """The house's 30% sector rule in names of weight 1 / ``nomes`` (4 at the Composite's 15)."""
    return 3 * nomes // 10


def grand_canonical_picks(z: np.ndarray, corr: np.ndarray, grupos: Sequence[str] | None, theta: float,
                          sizes: Sequence[int] = SIZES) -> tuple[int, ...]:
    """The set of largest F = 2 theta sum z + log det C_S among the fixed-size picks of each size in ``sizes``
    (each under ``names_per_sector`` of its size): the mode of the L-ensemble, the size free, on that grid. Ties go
    to the smaller size; sizes the names or the caps cannot fill are skipped. ValueError when none can be filled."""
    melhor: tuple[float, tuple[int, ...]] | None = None
    for k in sorted(sizes):
        caps = GroupCaps(tuple(grupos), names_per_sector(k)) if grupos is not None else None
        cabem = (len(z) if caps is None
                 else int(pd.Series(list(caps.groups)).value_counts().clip(upper=caps.max_per_group).sum()))
        if k > cabem:
            continue
        picks = fermionic_picks(z, corr, k, caps, theta)
        valor = objective(z, corr, picks, theta)
        if melhor is None or valor > melhor[0] + SWAP_TOL * max(1.0, abs(melhor[0])):
            melhor = (valor, picks)
    if melhor is None:
        raise ValueError(f"no size of {tuple(sizes)} can be filled with {len(z)} names under the caps")
    return melhor[1]


def _checked(pesos: pd.Series, setores: Mapping[str, str], treino: pd.DataFrame, limites: Limites) -> pd.Series:
    """``pesos`` when the house rules accept it (``CarteiraRecusada`` otherwise), as ``estrategia_por_nota``."""
    violacoes = checar_carteira(pesos, setores, estimar_momentos(treino[list(pesos.index)], shrink_mu=False).cov,
                                limites)
    if violacoes:
        raise CarteiraRecusada(violacoes)
    return pesos


def estrategia_grande_canonica(nota: NotaDoMes, setores: Mapping[str, str], theta: float,
                               tamanhos: Sequence[int] = SIZES, limites: Limites = LIMITES,
                               registro: dict | None = None,
                               correlation: Correlation = shrunk_correlation) -> Estrategia:
    """Equal weights on ``grand_canonical_picks`` among every scored candidate; the house rules. With too few
    candidates for the smallest size the portfolio is the Composite's ``carteira_da_nota``. ``registro`` keeps the
    number of names per decision date."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        score = nota(mes, list(treino.columns)).dropna()
        candidatos = candidates(score, setores, FULL_UNIVERSE)
        if not candidatos:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no eligible name with a score")
        menor = min(tamanhos)
        if _room_under_caps(candidatos, setores, names_per_sector(menor)) < menor:
            pesos = carteira_da_nota(score, setores, NOMES, POR_SETOR)
        else:
            valores = score[candidatos].to_numpy(dtype=float)
            desvio = valores.std()
            z = (valores - valores.mean()) / desvio if desvio > 0 else np.zeros(len(valores))
            picks = grand_canonical_picks(z, correlation(treino[candidatos]), [setores[p] for p in candidatos],
                                          theta, tamanhos)
            pesos = pd.Series(1.0 / len(picks), index=[candidatos[i] for i in picks])
        if registro is not None:
            registro[treino.index[-1]] = len(pesos)
        return _checked(pesos, setores, treino, limites)
    return _estrategia


def com_pesos_eic(estrategia: Estrategia, setores: Mapping[str, str], limites: Limites = LIMITES,
                  correlation: Correlation = shrunk_correlation) -> Estrategia:
    """``estrategia``'s names at the equal-information-contribution weights (``core.diversidade.eic_weights``) of
    their training-window correlation, inside the house's name and sector caps; the house rules."""
    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        nomes = list(estrategia(treino).index)
        por_setor = pd.Series(range(len(nomes))).groupby([setores[papel] for papel in nomes]).apply(list)
        pesos = eic_weights(correlation(treino[nomes]), limites.peso_maximo_acao,
                            [(posicoes, limites.teto_setor) for posicoes in por_setor])
        return _checked(pd.Series(pesos, index=nomes), setores, treino, limites)
    return _estrategia
