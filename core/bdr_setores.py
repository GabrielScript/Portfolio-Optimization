"""
bdr_setores.py - Fifteen BDRs, one per sector, in the order of the sectors' chance by 2030
(spec docs/superpowers/specs/2026-11-14-bdr-setores-design.md).

In each sector the pick is the candidate of best growth over the window among the liquid ones (traded on at least 90%
of the sessions, median daily volume of at least R$ 50 thousand); a sector with none takes the R$ 10 thousand tier,
flagged "thin"; a sector with neither is skipped. Pure functions: ``scripts/bdr_setores_avaliar.py`` reads the lake.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

PRESENCA = 0.90
PISOS = (("liquid", 50_000.0), ("thin", 10_000.0))   # median daily volume in R$, best tier first
QUANTOS = 15


@dataclass(frozen=True)
class Setor:
    nome: str
    nota: str                       # chance by 2030: A high, B medium-high, C medium
    candidatos: tuple[str, ...]


SETORES: tuple[Setor, ...] = (
    Setor("AI compute", "A", ("NVDC34", "AVGO34", "A1MD34", "MUTC34", "A1NE34", "M2RV34", "D1EL34")),
    Setor("Power for data centers", "A", ("V1RT34", "G2EV34", "V1ST34", "E1TN34", "C1CJ34", "N1RG34", "Q1UA34",
                                          "NEXT34", "BURA39", "DUKB34")),
    Setor("Chip manufacturing", "A", ("TSMC34", "ASML34", "A1MT34", "L1RC34", "K1LA34", "BSOX39")),
    Setor("Obesity drugs", "A", ("LILY34", "N1VO34")),
    Setor("AI agents and cloud", "A", ("MSFT34", "GOGL34", "AMZO34", "ORCL34", "SSFO34", "N1OW34")),
    Setor("Robotaxi", "A", ("TSLA34", "U1BE34", "BIDU34")),
    Setor("Cybersecurity", "A", ("C2RW34", "P2AN34", "F1TN34", "N2ET34", "Z2SC34", "A1KA34", "CSCO34", "BBUG39")),
    Setor("Defense", "A", ("P2LT34", "LMTB34", "RYTT34", "NOCG34", "BAER39", "SHLD39")),
    Setor("Satellites and launch", "A", ("SPCX34", "ORBX39")),
    Setor("Grid batteries", "A", ("Q2SC34", "E2NP34", "BLBT39")),
    Setor("Critical minerals", "A", ("M2PM34", "FCXO34", "BCPX39", "S2GM34", "A1LB34", "RIOT34", "EART39")),
    Setor("Aging economy", "A", ("W1EL34", "V1TA34", "UNHH34", "CVSH34", "H1CA34")),
    Setor("Digital twins", "A", ("S1NP34", "A1UT34", "P2TC34")),
    Setor("Surgical robotics", "A", ("I1SR34", "S1YK34", "MDTC34")),
    Setor("Biopharma tools and CDMOs", "B", ("TMOS34", "DHER34")),
    Setor("Precision agriculture", "B", ("DEEC34", "C1TV34")),
    Setor("AI drug discovery", "C", ("BGNO39",)),
    Setor("Early diagnostics", "C", ("I1LM34", "D1EX34", "ABTT34")),
    Setor("Robotics and humanoids", "C", ("BOTZ39",)),
    Setor("Machine payments", "C", ("C2OI34", "PYPL34", "X1YZ34")),
)


def _medidas(precos: pd.DataFrame, volumes: pd.DataFrame, nomes: Sequence[str], inicio: pd.Timestamp,
             fim: pd.Timestamp) -> pd.DataFrame:
    """Presence, median volume on the traded sessions and total return of ``nomes`` over [inicio, fim]."""
    presentes = [p for p in nomes if p in precos.columns and p in volumes.columns]
    janela_v = volumes.loc[inicio:fim, presentes]
    negociou = janela_v.notna() & (janela_v > 0)
    janela_p = precos.loc[inicio:fim, presentes]
    primeiro = janela_p.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan"))
    ultimo = janela_p.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else float("nan"))
    return pd.DataFrame({"presenca": negociou.mean(), "mediana": janela_v.where(negociou).median(),
                         "retorno": ultimo / primeiro - 1.0}, index=presentes)


def _ranking(medidas: pd.DataFrame, presenca: float) -> list[tuple[str, str]]:
    """The sector's eligible names with their tier: the liquid ones by growth, then the thin ones by growth. The thin
    tier counts only when the sector has no liquid name (the spec's round 1); later rounds go down the same list."""
    ranking = []
    for nivel, piso in PISOS:
        ok = medidas[(medidas["presenca"] >= presenca) & (medidas["mediana"] >= piso) & medidas["retorno"].notna()]
        ok = ok.drop(index=[p for p, _ in ranking], errors="ignore")
        ranking += [(p, nivel) for p in sorted(ok.index, key=lambda p: (-float(ok.at[p, "retorno"]), p))]
    return ranking


def escolher(precos: pd.DataFrame, volumes: pd.DataFrame, setores: Sequence[Setor], inicio: pd.Timestamp,
             fim: pd.Timestamp, quantos: int = QUANTOS, presenca: float = PRESENCA) -> pd.DataFrame:
    """Picks in rounds until ``quantos``: round r takes each sector's r-th name, in the sectors' order. ``precos`` and
    ``volumes`` are on the B3 calendar (a session without a trade has no volume)."""
    medidas = {s.nome: _medidas(precos, volumes, s.candidatos, inicio, fim) for s in setores}
    rankings = {s.nome: _ranking(medidas[s.nome], presenca) for s in setores}
    rodadas = max((len(r) for r in rankings.values()), default=0)
    linhas = []
    for rodada in range(1, rodadas + 1):
        for ordem, setor in enumerate(setores, start=1):
            if len(linhas) == quantos:
                break
            if len(rankings[setor.nome]) < rodada:
                continue
            papel, nivel = rankings[setor.nome][rodada - 1]
            m = medidas[setor.nome]
            linhas.append({"rodada": rodada, "ordem": ordem, "setor": setor.nome, "nota": setor.nota, "papel": papel,
                           "nivel": nivel, "retorno": float(m.at[papel, "retorno"]),
                           "mediana": float(m.at[papel, "mediana"]), "presenca": float(m.at[papel, "presenca"])})
    return pd.DataFrame(linhas, columns=["rodada", "ordem", "setor", "nota", "papel", "nivel", "retorno", "mediana",
                                         "presenca"])
