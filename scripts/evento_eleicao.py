"""
Election-window event study of B3 volatility (user's request of 2026-10-22/23, DECISOES 101 and 123). Pre-registered:
this docstring was committed before the first run, and the script runs once.

Question: is the Brazilian stock market more volatile in the week after a presidential election round than in the
weeks before the campaign's final stretch? Only volatility is tested; with ten rounds, the direction cannot be
established (it is printed, never decided on).

Data: the frozen lake exports of `docs/dados/2026-10-22/` (`universe_prices.parquet`, the daily total-return index,
from 03/01/2005; `universe_monthly.parquet`, the point-in-time universe), each checked against that folder's
SHA256SUMS. The market is the equal-weighted mean daily return of the month's universe members that have a return on
that day and the day before (no clipping: the lake's corrections are the data cleaning).

Events: the ten rounds of 2006-2022 (first round and runoff; the vote is on a Sunday); day 0 is the first session
after the vote. 2002 is out: the frozen export starts in 2005. The 2026 first round (04/10/2026) is printed as a
description, outside the test; the 2026 runoff (25/10/2026) is after the data.

Measure (per event): the log of the ratio between the realized volatility of sessions 0 to +4 and that of sessions
-60 to -11 (realized volatility = root mean square of the daily returns). A runoff's sessions -60 to -11 contain its
first round's week: that biases the runoffs against finding an effect, kept so rounds and placebo days are measured the
same way. Secondary, printed only: the same ratio for
sessions -5 to -1 (anticipation) and |return of day 0| / mean |return| of sessions -60 to -11.

Inference: a placebo test. The placebo days are the first session of each week (the B3's first session after a
Sunday) at least 60 sessions from every round of 2006-2026 and with sessions -60 to +4 inside the data. 20,000 draws of
ten placebo days (without replacement, numpy seed 20261023; draws are independent, the rounds of one year are not:
a known limit) give the null distribution of the mean log ratio;
p = (1 + number of draws whose mean is >= the observed mean) / (1 + 20,000).

Decision (one-sided): "election weeks are more volatile" holds if p <= 0.05; otherwise "not established". Nothing in
the strategy, the paper or the app changes either way: a rule built on this (e.g., no rebalance in an election week)
would be a new strategy test, pre-registered on its own, and would raise N. This study is not a strategy
configuration, so `docs/tentativas.csv` (the DSR's N) does not change.

Usage: .venv/Scripts/python -X utf8 scripts/evento_eleicao.py > docs/eleicao-2026-10-23.txt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE = (-60, -10)  # sessions -60 to -11 (end exclusive)
JANELA = (0, 5)  # sessions 0 to +4
ANTES = (-5, 0)  # sessions -5 to -1


@dataclass(frozen=True)
class Medidas:
    principal: float  # log(RV[0, +4] / RV[-60, -11]): the tested measure
    antecipacao: float  # log(RV[-5, -1] / RV[-60, -11]): printed only
    dia0: float  # |r(0)| / mean |r| of [-60, -11]: printed only


def _rv(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def medidas_do_evento(retornos: np.ndarray, i0: int) -> Medidas:
    """The measures of the event whose day 0 is position ``i0`` of the market's daily ``retornos``."""
    if i0 + BASE[0] < 0 or i0 + JANELA[1] > len(retornos):
        raise ValueError(f"window of day {i0} outside the data ({len(retornos)} sessions)")
    base = retornos[i0 + BASE[0]:i0 + BASE[1]]
    return Medidas(principal=float(np.log(_rv(retornos[i0 + JANELA[0]:i0 + JANELA[1]]) / _rv(base))),
                   antecipacao=float(np.log(_rv(retornos[i0 + ANTES[0]:i0 + ANTES[1]]) / _rv(base))),
                   dia0=float(abs(retornos[i0]) / np.mean(np.abs(base))))


DISTANCIA_MINIMA = 60  # sessions between a placebo day and every round


def dias_placebo(sessoes: pd.DatetimeIndex, eleicoes: list[int]) -> list[int]:
    """Positions of the first session of each week (its previous session is in an earlier ISO week) at least
    `DISTANCIA_MINIMA` sessions from every round in ``eleicoes`` and with sessions -60 to +4 inside the data."""
    semanas = sessoes.isocalendar()
    chave = (semanas["year"] * 100 + semanas["week"]).to_numpy()
    abre = np.flatnonzero(np.r_[False, chave[1:] != chave[:-1]])
    cabe = (abre + BASE[0] >= 0) & (abre + JANELA[1] <= len(sessoes))
    longe = np.all(np.abs(abre[:, None] - np.asarray(eleicoes)[None, :]) >= DISTANCIA_MINIMA, axis=1)
    return [int(p) for p in abre[cabe & longe]]


SORTEIOS, SEMENTE, ALFA = 20_000, 20261023, 0.05


def p_placebo(observado: float, placebo: np.ndarray, n_eventos: int, sorteios: int = SORTEIOS,
              semente: int = SEMENTE) -> float:
    """(1 + draws whose mean of ``n_eventos`` placebo values, without replacement, is >= ``observado``) / (1 + draws)."""
    rng = np.random.default_rng(semente)
    medias = np.array([rng.choice(placebo, size=n_eventos, replace=False).mean() for _ in range(sorteios)])
    return float((1 + np.sum(medias >= observado - 1e-12)) / (1 + sorteios))


def retorno_de_mercado(precos: pd.DataFrame, membros: pd.DataFrame) -> pd.Series:
    """Equal-weighted mean daily return of the month's members (``membros``: month, security) with a price on the day
    and on the session before; the first session (no previous price) is dropped."""
    retornos = (precos / precos.shift(1) - 1).iloc[1:]
    mes = retornos.index.to_period("M").to_timestamp()
    pertence = pd.DataFrame(False, index=retornos.index, columns=retornos.columns)
    for mes_membro, papeis in membros.groupby("month")["security"]:
        dias = mes == pd.Timestamp(mes_membro)
        colunas = retornos.columns.intersection(pd.Index(papeis))
        pertence.loc[dias, colunas] = True
    return retornos.where(pertence).mean(axis=1, skipna=True).rename("mercado")


def dia_zero(sessoes: pd.DatetimeIndex, domingo: pd.Timestamp) -> int:
    """Position in ``sessoes`` of the first session after the vote's Sunday."""
    return int(sessoes.searchsorted(domingo, side="right"))


RODADAS = {  # the Sundays of the presidential votes (TSE calendar)
    "2006 1st": "2006-10-01", "2006 runoff": "2006-10-29", "2010 1st": "2010-10-03", "2010 runoff": "2010-10-31",
    "2014 1st": "2014-10-05", "2014 runoff": "2014-10-26", "2018 1st": "2018-10-07", "2018 runoff": "2018-10-28",
    "2022 1st": "2022-10-02", "2022 runoff": "2022-10-30",
}
FORA_DO_TESTE = {"2026 1st": "2026-10-04", "2026 runoff": "2026-10-25"}  # described / after the data


def decidir(p: float) -> str:
    return "election weeks are more volatile" if p <= ALFA else "not established"


def carregar():
    """The frozen market series: (daily equal-weighted returns, the files' sha256)."""
    from pathlib import Path

    from scripts.avaliar_robustez import conferir_hashes

    pasta = Path(__file__).resolve().parent.parent / "docs" / "dados" / "2026-10-22"
    sha = conferir_hashes(pasta, ["universe_prices.parquet", "universe_monthly.parquet"])
    precos = pd.read_parquet(pasta / "universe_prices.parquet")
    precos = precos.set_index(pd.to_datetime(precos.pop("date"))).sort_index().astype(float)
    membros = pd.read_parquet(pasta / "universe_monthly.parquet")
    membros = membros.assign(month=pd.to_datetime(membros["month"]))
    return retorno_de_mercado(precos, membros).dropna(), sha


def main() -> int:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    mercado, sha = carregar()
    sessoes, r = mercado.index, mercado.to_numpy()
    print(f"Files (sha256): {sha}")
    print(f"Market: {len(r)} sessions, {sessoes[0]:%d/%m/%Y} to {sessoes[-1]:%d/%m/%Y}; mean members' daily return")
    linhas, todas = [], {**RODADAS, **FORA_DO_TESTE}
    posicoes = {nome: dia_zero(sessoes, pd.Timestamp(domingo)) for nome, domingo in todas.items()}
    for nome, i0 in posicoes.items():
        if i0 + JANELA[1] > len(r):
            print(f"{nome}: after the data (day 0 would be session {i0} of {len(r)})")
            continue
        m = medidas_do_evento(r, i0)
        linhas.append({"round": nome, "day 0": f"{sessoes[i0]:%d/%m/%Y}", "log RV ratio [0,+4]": m.principal,
                       "RV ratio": np.exp(m.principal), "log ratio [-5,-1]": m.antecipacao,
                       "|r0| / mean|r|": m.dia0, "r day 0": r[i0], "r [0,+4]": float(np.prod(1 + r[i0:i0 + 5]) - 1),
                       "in test": nome in RODADAS})
    tabela = pd.DataFrame(linhas).set_index("round")
    print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
    teste = tabela[tabela["in test"]]
    observado = float(teste["log RV ratio [0,+4]"].mean())
    placebo_pos = dias_placebo(sessoes, eleicoes=list(posicoes.values()))
    placebo = np.array([medidas_do_evento(r, p).principal for p in placebo_pos])
    p = p_placebo(observado, placebo, n_eventos=len(teste))
    print(f"\nPlacebo days: {len(placebo)} week openings >= {DISTANCIA_MINIMA} sessions from every round; their mean log "
          f"ratio {placebo.mean():.4f} (median {np.median(placebo):.4f})")
    print(f"Observed mean log ratio over the {len(teste)} rounds: {observado:.4f} (RV x{np.exp(observado):.2f}); "
          f"rounds above the placebo median: {int((teste['log RV ratio [0,+4]'] > np.median(placebo)).sum())} of "
          f"{len(teste)}")
    print(f"p = {p:.4f} ({SORTEIOS} draws, seed {SEMENTE}); decision (p <= {ALFA}): {decidir(p)}")
    print(f"Direction, printed only: mean [0,+4] return {teste['r [0,+4]'].mean():+.4f}; "
          f"{int((teste['r [0,+4]'] > 0).sum())} of {len(teste)} rounds up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
