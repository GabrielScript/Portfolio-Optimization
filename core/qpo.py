"""
qpo.py - Quantum Portfolio Optimization, classical side (docs/superpowers/specs/2026-10-27-qpo-design.md).

The QUBO picks k of the n best Conservadora names: E(x) = x'Qx = -alpha'x / k + (lambda / 2) x'Sigma x / k^2 with
|x| = k (the negative mean-variance utility of the equal-weight portfolio x / k). The exact solver here is the
ceiling a QAOA can at best reach on the same QUBO; the QAOA itself lives in ``qpo_kaggle/qpo_qaoa.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.backtest import Estrategia, estrategia_otimizada_nas_regras
from core.config import ConfigOtimizacao, Objetivo
from core.estimation import estimar_momentos
from core.risco import emissor
from core.sinais import conservadora

IC0 = 0.059          # the Conservadora's 1998-2007 IC (DECISOES 76): before every OOS date


@dataclass(frozen=True)
class ExactSolution:
    picks: tuple[int, ...]   # indices of the chosen names, ascending
    energy: float            # E at the optimum
    worst: float             # E at the worst feasible subset (for the approximation ratio)


def energy(q: np.ndarray, x: np.ndarray) -> float | np.ndarray:
    """x'Qx for one bitstring (n,) or a batch (m, n)."""
    valores = np.einsum("...i,ij,...j->...", x, q, x)
    return float(valores) if np.ndim(valores) == 0 else valores


def build_qubo(alpha: np.ndarray, cov: np.ndarray, k: int, risk_aversion: float) -> np.ndarray:
    """Q = (lambda / (2 k^2)) Sigma - diag(alpha) / k, so x'Qx = -alpha'w + (lambda / 2) w'Sigma w at w = x / k."""
    alpha, cov = np.asarray(alpha, dtype=float), np.asarray(cov, dtype=float)
    if cov.shape != (len(alpha), len(alpha)) or k < 1:
        raise ValueError(f"alpha {alpha.shape}, Sigma {cov.shape}, k = {k}: need an (n, n) Sigma and k >= 1")
    q = risk_aversion / (2.0 * k * k) * (cov + cov.T) / 2.0
    return q - np.diag(alpha) / k


def qubo_to_ising(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    (h, J, const) with x'Qx = h'z + sum_{i<j} J_ij z_i z_j + const for x = (1 - z) / 2, so z = +1 is x = 0 (|0>).
    J is symmetric with a zero diagonal; each pair counts once (the upper triangle).
    """
    s = (q + q.T) / 2.0
    diagonal = np.diag(s)
    fora = s - np.diag(diagonal)
    h = -diagonal / 2.0 - fora.sum(axis=1) / 2.0
    return h, fora / 2.0, float(diagonal.sum() / 2.0 + fora.sum() / 4.0)


def _subsets(m: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    """All r-subsets of range(m): their indices (C(m, r), r) and their 0/1 rows (C(m, r), m), lexicographic."""
    combinacoes = list(itertools.combinations(range(m), r))
    indices = np.array(combinacoes, dtype=np.int64).reshape(len(combinacoes), r)   # r = 0: one empty subset
    linhas = np.zeros((len(indices), m))
    linhas[np.arange(len(indices))[:, None], indices] = 1.0
    return indices, linhas


def solve_exact(q: np.ndarray, k: int) -> ExactSolution:
    """
    Minimum and maximum of x'Qx over every |x| = k, exactly: the names split in halves A and B, and for each split
    j + (k - j) the C(n_A, j) x C(n_B, k - j) energies E_A + E_B + 2 x_A'Q_AB x_B come from one matrix product.
    The largest block at n = 28, k = 15 is 3,432 x 3,003 (about 80 MB in float64).
    """
    n = len(q)
    if not 1 <= k <= n:
        raise ValueError(f"k = {k} is outside 1..{n}")
    s = (q + q.T) / 2.0
    n_a = n // 2
    s_aa, s_bb, s_ab = s[:n_a, :n_a], s[n_a:, n_a:], s[:n_a, n_a:]
    melhor: tuple[float, tuple[int, ...]] | None = None
    pior = -np.inf
    for j in range(max(0, k - (n - n_a)), min(k, n_a) + 1):
        idx_a, x_a = _subsets(n_a, j)
        idx_b, x_b = _subsets(n - n_a, k - j)
        bloco = (energy(s_aa, x_a)[:, None] + energy(s_bb, x_b)[None, :] + 2.0 * (x_a @ s_ab) @ x_b.T)
        i, m = np.unravel_index(int(np.argmin(bloco)), bloco.shape)
        if melhor is None or bloco[i, m] < melhor[0]:
            melhor = (float(bloco[i, m]), tuple(int(v) for v in (*idx_a[i], *(idx_b[m] + n_a))))
        pior = max(pior, float(bloco.max()))
    assert melhor is not None
    return ExactSolution(melhor[1], melhor[0], pior)


def approximation_ratio(e: float, e_best: float, e_worst: float) -> float:
    """(E_worst - E) / (E_worst - E_best) over the feasible set: 1 at the optimum, 0 at the worst subset."""
    if e_worst == e_best:
        return 1.0
    return (e_worst - e) / (e_worst - e_best)


def candidates(score: pd.Series, sectors: Mapping[str, str], n: int) -> list[str]:
    """
    The QUBO's names: the Conservadora score in descending order, skipping a name without sector (the fixed rule
    SEM_SETOR) and the 2nd class of an issuer already in (the higher score stays), as ``carteira_da_nota``, but with
    no per-sector count: the 30% sector cap is enforced on the weights.
    """
    escolhidos: list[str] = []
    emissores: set[str] = set()
    for papel in score.dropna().sort_values(ascending=False, kind="stable").index:
        if not sectors.get(papel) or emissor(papel) in emissores:
            continue
        escolhidos.append(papel)
        emissores.add(emissor(papel))
        if len(escolhidos) == n:
            break
    return escolhidos


def qubo_alphas(score: pd.Series, cov: pd.DataFrame, ic: float = IC0) -> pd.Series:
    """Grinold-Kahn alpha = IC x vol x z, z = the score's cross-sectional z-score (ddof 0); all ties -> zeros."""
    desvio = score.std(ddof=0)
    z = (score - score.mean()) / desvio if desvio > 0 else score * 0.0
    vol = np.sqrt(np.diag(cov.loc[score.index, score.index].to_numpy()))
    return ic * vol * z


# --- strategy layer --------------------------------------------------------------------------------------------------

K = 15                  # the Conservadora's size
RISK_AVERSION = 1.0     # growth-optimal (Kelly) lambda: the "extremely aggressive" choice, unlevered
# the user's fixed objective inside the chosen set: max Sharpe (Charnes-Cooper, rf = CDI of the date), the app's caps
CFG_MAX_SHARPE = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.20, n_ativos_max=None, vol_maxima=0.40)

Selector = Callable[[pd.Timestamp, pd.DataFrame, pd.Series], list[str]]   # (decision date, treino, score) -> names


@dataclass(frozen=True)
class QuboInstance:
    date: pd.Timestamp
    tickers: tuple[str, ...]
    q: np.ndarray
    alpha: np.ndarray
    cov: np.ndarray
    k: int


def qubo_instance(date: pd.Timestamp, treino: pd.DataFrame, score: pd.Series, sectors: Mapping[str, str], n: int,
                  k: int = K, risk_aversion: float = RISK_AVERSION) -> QuboInstance | None:
    """The QUBO of the decision: top-n candidates, Ledoit-Wolf Sigma of the training window. None when there are no
    more than k candidates (nothing to choose: they all go in)."""
    nomes = candidates(score, sectors, n)
    if len(nomes) <= k:
        return None
    cov = estimar_momentos(treino[nomes], shrink_mu=False).cov
    alpha = qubo_alphas(score[nomes], cov).to_numpy()
    return QuboInstance(date, tuple(nomes), build_qubo(alpha, cov.to_numpy(), k, risk_aversion), alpha,
                        cov.to_numpy(), k)


def exact_picks(instance: QuboInstance) -> tuple[int, ...]:
    return solve_exact(instance.q, instance.k).picks


def qubo_selector(sectors: Mapping[str, str], n: int, k: int, solve: Callable[[QuboInstance], tuple[int, ...]],
                  record: dict | None = None) -> Selector:
    """Selection by a QUBO solver; ``record`` (decision date -> (instance, picks)) keeps every QUBO for export."""
    def _select(date: pd.Timestamp, treino: pd.DataFrame, score: pd.Series) -> list[str]:
        instancia = qubo_instance(date, treino, score, sectors, n, k)
        if instancia is None:
            return candidates(score, sectors, n)
        picks = solve(instancia)
        if record is not None:
            record[date] = (instancia, picks)
        return [instancia.tickers[i] for i in picks]
    return _select


def recorded_selector(picks_by_date: Mapping[pd.Timestamp, tuple[str, ...]]) -> Selector:
    """Replay of picks solved elsewhere (the QAOA on Kaggle); a decision date without a pick stops the run."""
    def _select(date: pd.Timestamp, treino: pd.DataFrame, score: pd.Series) -> list[str]:
        if date not in picks_by_date:
            raise KeyError(f"{date:%Y-%m-%d}: no recorded pick")
        return list(picks_by_date[date])
    return _select


def random_selector(sectors: Mapping[str, str], n: int, k: int, seed: int) -> Selector:
    """k uniform draws of the top-n candidates; the stream depends only on (seed, date): reproducible run to run."""
    def _select(date: pd.Timestamp, treino: pd.DataFrame, score: pd.Series) -> list[str]:
        nomes = candidates(score, sectors, n)
        if len(nomes) <= k:
            return nomes
        gerador = np.random.default_rng([seed, date.toordinal()])
        return [nomes[i] for i in sorted(gerador.choice(len(nomes), size=k, replace=False))]
    return _select


def selection_then_max_sharpe(select: Selector, monthly_returns: pd.DataFrame, dividends: pd.DataFrame,
                              sectors: Mapping[str, str], cfg: ConfigOtimizacao,
                              rf_daily: pd.Series | None) -> Estrategia:
    """
    Every QPO arm in the engine: the Conservadora score of the decision month (public data before it only), a
    selection, then ``estrategia_otimizada_nas_regras`` on the chosen names (sector cap, ``checar_carteira``,
    ``CarteiraRecusada``), Jorion-shrunk mu.
    """
    pesos_nas_regras = estrategia_otimizada_nas_regras(cfg, True, sectors, rf_daily)

    def _strategy(treino: pd.DataFrame) -> pd.Series:
        mes = treino.index[-1].to_period("M").to_timestamp()
        score = conservadora(monthly_returns, dividends, mes, treino.columns)
        nomes = select(treino.index[-1], treino, score)
        if not nomes:
            raise ValueError(f"{treino.index[-1]:%Y-%m-%d}: no name selected")
        return pesos_nas_regras(treino[nomes])
    return _strategy
