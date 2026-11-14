"""
qubo_universo.py - the QPO's QUBO over the full universe (docs/superpowers/specs/2026-10-27-qubo-universo-design.md).

At n of 100-230 names the enumeration of ``core.qpo.solve_exact`` (n <= 28) and any QAOA simulation are out of reach
(C(178, 15) = 2.4e21), so the same QUBO is solved as a convex MIQP by SCIP: binary x, sum x = k, and the epigraph
t >= x'Px with the objective t + c'x (``convex_split``). On binary x this equals x'Qx exactly; starting from the
natural split of Q (a generic eigenvalue shift of Q would loosen it), the identity share of Sigma moves into c.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyscipopt as scip

from core.qpo import RISK_AVERSION, QuboInstance, build_qubo, energy
from core.views import escore_normal

FULL_UNIVERSE = 10**6          # more names than the B3 lists: ``core.qpo.candidates`` keeps every eligible name
TIME_LIMIT = 600.0             # seconds per MIQP (fixed by the timing spike before the pre-registration commit)
SECTOR_CAP = 4                 # picks per sector, as the Conservadora's ``carteira_da_nota(nota, setores, 15, 4)``


@dataclass(frozen=True)
class GroupCaps:
    """At most ``max_per_group`` picks share a group (the sector); ``groups`` is aligned with the instance's tickers."""
    groups: Sequence[str]
    max_per_group: int


@dataclass(frozen=True)
class MiqpSolution:
    picks: tuple[int, ...]   # indices of the chosen names, ascending
    energy: float            # x'Qx of the picks, recomputed in numpy (not SCIP's objective)
    optimal: bool            # SCIP proved optimality (status "optimal")
    gap: float               # SCIP's relative gap at the end
    seconds: float


def convex_split(alpha: np.ndarray, cov: np.ndarray, k: int, risk_aversion: float) -> tuple[np.ndarray, np.ndarray]:
    """(P, c) with x'Px + c'x = x'Qx on binary x and P = S - delta I PSD, S = (lambda / (2 k^2)) Sigma, delta =
    lambda_min(S): x_i^2 = x_i moves delta's share of the diagonal into the linear term, so the continuous relaxation
    pays it in full on fractional x and its bound tightens (diagonal perturbation, as in QCR)."""
    s = risk_aversion / (2.0 * k * k) * (cov + cov.T) / 2.0
    delta = max(float(np.linalg.eigvalsh(s).min()), 0.0)
    return s - delta * np.eye(len(s)), -np.asarray(alpha, dtype=float) / k + delta


def solve_miqp(instance: QuboInstance, risk_aversion: float, caps: GroupCaps | None = None,
               time_limit: float = TIME_LIMIT) -> MiqpSolution:
    """min x'Qx over |x| = k (and the group caps), exactly, as a convex MIQP. SCIP starts from the first k names in
    candidate (score) order that fit the caps, so a solve stopped by ``time_limit`` still returns a feasible pick."""
    alpha, cov, k = instance.alpha, instance.cov, instance.k
    if not np.allclose(build_qubo(alpha, cov, k, risk_aversion), instance.q, rtol=0.0, atol=1e-12):
        raise ValueError(f"risk aversion {risk_aversion} does not rebuild the instance's Q from its alpha and Sigma")
    n = len(alpha)
    modelo = scip.Model()
    modelo.hideOutput()
    x = [modelo.addVar(vtype="B") for _ in range(n)]
    risco = modelo.addVar(lb=0.0)
    modelo.addCons(scip.quicksum(x) == k)
    if caps is not None:
        for grupo in sorted(set(caps.groups)):
            modelo.addCons(scip.quicksum(x[i] for i in range(n) if caps.groups[i] == grupo) <= caps.max_per_group)
    p, c = convex_split(alpha, cov, k, risk_aversion)
    modelo.addCons(scip.quicksum(p[i, j] * x[i] * x[j] for i in range(n) for j in range(n)) <= risco)
    modelo.setObjective(risco + scip.quicksum(c[i] * x[i] for i in range(n)), "minimize")
    modelo.setParam("limits/time", time_limit)
    inicial = _feasible_start(np.arange(n), k, _Grupos(caps, n))
    solucao = modelo.createSol()
    for i in inicial:
        modelo.setSolVal(solucao, x[i], 1.0)
    modelo.setSolVal(solucao, risco, float(p[np.ix_(inicial, inicial)].sum()))
    modelo.addSol(solucao)
    modelo.optimize()
    picks = tuple(i for i in range(n) if modelo.getVal(x[i]) > 0.5)
    escolha = np.zeros(n)
    escolha[list(picks)] = 1.0
    return MiqpSolution(picks, float(energy(instance.q, escolha)), modelo.getStatus() == "optimal",
                        float(modelo.getGap()), float(modelo.getSolvingTime()))


def random_subset_energy(q: np.ndarray, k: int) -> float:
    """E[x'Qx] over a uniform k-subset: P(i in) = k/n, P(i and j in) = k(k-1)/(n(n-1)) for i != j."""
    n = len(q)
    s = (q + q.T) / 2.0
    traco = float(np.trace(s))
    return k / n * traco + k * (k - 1) / (n * (n - 1)) * (float(s.sum()) - traco)


# --- simulated annealing: the quantum-inspired solver of the same QUBO (swap moves keep |x| = k and the caps) -------

ANNEAL_RESTARTS = 8
ANNEAL_STEPS = 20_000          # per restart
ANNEAL_COOLING = 1e-3          # final temperature / initial temperature (geometric schedule)


class _Grupos:
    """Group ids and per-group counts of the current picks; no caps = one group with room for everyone."""

    def __init__(self, caps: GroupCaps | None, n: int) -> None:
        nomes = list(caps.groups) if caps is not None else [""] * n
        indice = {nome: i for i, nome in enumerate(sorted(set(nomes)))}
        self.ids = np.array([indice[nome] for nome in nomes])
        self.teto = caps.max_per_group if caps is not None else n
        self.contagem = np.zeros(len(indice), dtype=int)

    def cabe(self, entra: int, sai: int) -> bool:
        return self.ids[entra] == self.ids[sai] or self.contagem[self.ids[entra]] < self.teto


def _feasible_start(ordem: np.ndarray, k: int, grupos: _Grupos) -> np.ndarray:
    """The first k names in ``ordem`` whose group still has room; ValueError when the caps leave fewer than k."""
    grupos.contagem[:] = 0
    escolhidos = []
    for i in ordem:
        if grupos.contagem[grupos.ids[i]] < grupos.teto:
            escolhidos.append(i)
            grupos.contagem[grupos.ids[i]] += 1
            if len(escolhidos) == k:
                return np.array(escolhidos)
    raise ValueError(f"the group caps admit fewer than k = {k} of the {len(ordem)} names")


def _initial_temperature(s: np.ndarray, dentro: np.ndarray, fora: np.ndarray, campo: np.ndarray,
                         rng: np.random.Generator, amostras: int = 256) -> float:
    """Mean |dE| of random swaps from the start: a move of typical size is accepted with probability ~1/e at first."""
    i, j = rng.choice(dentro, size=amostras), rng.choice(fora, size=amostras)
    variacoes = np.abs(s[i, i] + s[j, j] - 2.0 * s[i, j] - 2.0 * campo[i] + 2.0 * campo[j])
    return max(float(np.mean(variacoes)), 1e-12)


def _anneal_once(s: np.ndarray, k: int, grupos: _Grupos, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """One restart: Metropolis on swaps (one pick out, one name in), geometric cooling; the best state seen."""
    n = len(s)
    dentro = _feasible_start(rng.permutation(n), k, grupos)
    fora = np.setdiff1d(np.arange(n), dentro)
    x = np.zeros(n)
    x[dentro] = 1.0
    campo = s @ x                                   # (Sx)_i, kept up to date
    atual = float(x @ campo)
    melhor, melhores = atual, dentro.copy()
    saidas, entradas = rng.integers(k, size=ANNEAL_STEPS), rng.integers(n - k, size=ANNEAL_STEPS)
    sorteios = rng.random(ANNEAL_STEPS)
    temperatura = _initial_temperature(s, dentro, fora, campo, rng)
    resfriamento = ANNEAL_COOLING ** (1.0 / ANNEAL_STEPS)
    for passo in range(ANNEAL_STEPS):
        i, j = dentro[saidas[passo]], fora[entradas[passo]]
        temperatura *= resfriamento
        if not grupos.cabe(j, i):
            continue
        delta = s[i, i] + s[j, j] - 2.0 * s[i, j] - 2.0 * campo[i] + 2.0 * campo[j]
        if delta > 0.0 and sorteios[passo] >= np.exp(-delta / temperatura):
            continue
        campo += s[:, j] - s[:, i]
        dentro[saidas[passo]], fora[entradas[passo]] = j, i
        grupos.contagem[grupos.ids[i]] -= 1
        grupos.contagem[grupos.ids[j]] += 1
        atual += delta
        if atual < melhor:
            melhor, melhores = atual, dentro.copy()
    return melhor, melhores


def anneal(q: np.ndarray, k: int, caps: GroupCaps | None = None, seed: int = 0) -> tuple[int, ...]:
    """Best of ``ANNEAL_RESTARTS`` annealing runs (exact energies re-scored at the end); reproducible per seed."""
    s = (q + q.T) / 2.0
    rng = np.random.default_rng(seed)
    grupos = _Grupos(caps, len(s))
    finais = [_anneal_once(s, k, grupos, rng)[1] for _ in range(ANNEAL_RESTARTS)]
    melhor = min(finais, key=lambda picks: float(energy(s, np.isin(np.arange(len(s)), picks).astype(float))))
    return tuple(sorted(int(i) for i in melhor))


def miqp_solver(sectors: Mapping[str, str], max_per_sector: int = SECTOR_CAP,
                log: dict | None = None) -> Callable[[QuboInstance], tuple[int, ...]]:
    """The ``solve`` of ``core.qpo.qubo_selector``: the capped MIQP optimum; ``log`` keeps each date's solution."""
    def _solve(instance: QuboInstance) -> tuple[int, ...]:
        caps = GroupCaps(tuple(sectors[papel] for papel in instance.tickers), max_per_sector)
        solucao = solve_miqp(instance, RISK_AVERSION, caps)
        if log is not None:
            log[instance.date] = solucao
        return solucao.picks
    return _solve


@dataclass(frozen=True)
class AnnealRecord:
    picks: tuple[int, ...]   # the lower-energy pick of the two runs
    energy: float
    agree: bool              # both independent runs ended on the same energy


def anneal_solver(sectors: Mapping[str, str], max_per_sector: int = SECTOR_CAP, log: dict | None = None,
                  annealer: Callable[..., tuple[int, ...]] = anneal) -> Callable[[QuboInstance], tuple[int, ...]]:
    """The full universe's solver of record: two independent annealing runs (seeds = the date's ordinal and the next
    integer) on the capped QUBO; the lower energy wins and ``log`` records whether they agreed."""
    def _solve(instance: QuboInstance) -> tuple[int, ...]:
        caps = GroupCaps(tuple(sectors[papel] for papel in instance.tickers), max_per_sector)
        semente = instance.date.toordinal()
        n = len(instance.tickers)
        corridas = [annealer(instance.q, instance.k, caps, seed=semente + i) for i in (0, 1)]
        energias = [float(energy(instance.q, np.isin(np.arange(n), picks).astype(float))) for picks in corridas]
        melhor = int(np.argmin(energias))
        if log is not None:
            log[instance.date] = AnnealRecord(corridas[melhor], energias[melhor],
                                              abs(energias[0] - energias[1]) <= 1e-12)
        return corridas[melhor]
    return _solve


def composite_score(conservadora_of_month: Callable[[pd.Timestamp, Sequence[str]], pd.Series],
                    *others_of_month: Callable[[pd.Timestamp, Sequence[str]], pd.Series]
                    ) -> Callable[[pd.Timestamp, Sequence[str]], pd.Series]:
    """Attempts 48-49's score (docs/superpowers/specs/2026-10-27-qubo-universo-valor-design.md, value; and
    2026-10-28-qubo-universo-crescimento-design.md, + profit growth): over the names with a Conservadora score, the
    mean of its normal score and each other signal's normal score among those names, with the ones a name has (the
    Conservadora's always). Equal weights: nothing is fitted."""
    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        base = conservadora_of_month(mes, candidatos).dropna()
        escores = pd.DataFrame({0: escore_normal(base), **{posicao: escore_normal(outra(mes, list(base.index)).dropna())
                                                          for posicao, outra in enumerate(others_of_month, start=1)}})
        return escores.reindex(base.index).mean(axis=1)
    return _nota
