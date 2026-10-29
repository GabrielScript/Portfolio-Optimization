"""
h2_rotacao_avaliar.py - the one reading of attempt 53, H2: the BR/US/gold sleeve rotation with a CDI filter
(docs/superpowers/specs/2026-10-29-h2-rotacao-design.md; trial row `66c6268`). No network: the Conservadora and
BOVA11 curves, SPY, GLD, PTAX and the CDI frozen and hash-checked.

    python -X utf8 scripts/h2_rotacao_avaliar.py > docs/h2-rotacao-2026-10-29.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.rotacao import CAIXA, rotacao, sleeve_em_reais  # noqa: E402
from core.vencedoras import indice_do_caixa  # noqa: E402
from scripts.avaliar_robustez import FASES, conferir_hashes, contar_tentativas, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI  # noqa: E402
from scripts.qpo_avaliar import _year_table  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, strategy_gate, years_beating_all  # noqa: E402

ARM = "H2-Rotacao"
CONSERVADORA = "Conservadora"
CURVES_FILE = FOLDER.parent / "composto-curvas-2026-10-28.parquet"
MACRO_FOLDER = FOLDER.parent / "2026-10-21-macro"
US_FOLDER = FOLDER.parent / "2026-10-29-us"
TAXAS = {"US": ("SPY", 0.0025), "Ouro": ("GLD", 0.0040)}       # spec §1: fee drag a year


def posse_diaria(escolhas: pd.Series, datas: pd.DatetimeIndex) -> pd.Series:
    """The sleeve held on each session after the first decision: a month-end choice holds from the next session."""
    posse = escolhas.reindex(datas).ffill().shift(1)
    return posse.loc[posse.first_valid_index():]


def trocas_por_ano(escolhas: pd.Series) -> pd.Series:
    """Changes of choice per calendar year (the first buy is not a switch)."""
    trocas = (escolhas != escolhas.shift()).astype(int)
    trocas.iloc[0] = 0
    return trocas.groupby(trocas.index.year).sum()


def main() -> int:
    sha256 = (conferir_hashes(FOLDER.parent, [CURVES_FILE.name]) | conferir_hashes(MACRO_FOLDER, ["cdi_daily.parquet"])
              | conferir_hashes(US_FOLDER, ["spy_gld_daily.parquet", "ptax_daily.parquet"]))
    todas = pd.read_parquet(CURVES_FILE)
    us = pd.read_parquet(US_FOLDER / "spy_gld_daily.parquet")
    fechamentos = {t: pd.Series(g["close"].to_numpy(), index=pd.DatetimeIndex(g["date"]).as_unit("ns"))
                   for t, g in us.groupby("ticker")}
    bruto = pd.read_parquet(US_FOLDER / "ptax_daily.parquet")
    ptax = pd.Series(bruto["ptax"].to_numpy(), index=pd.DatetimeIndex(bruto["date"]).as_unit("ns"))
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO_FOLDER / "cdi_daily.parquet")
    n_tentativas = contar_tentativas()
    print(f"Files (sha256): {sha256}; arm {ARM}; DSR N {n_tentativas}")
    print("Gross of income tax; before IVVB11 (2014) and GOLD11 (2020) the US and gold sleeves are proxies.")

    p_cons, p_bova, dsr = [], [], []
    for fase in FASES:
        curvas = todas.query("fase == @fase").set_index("date")[[CONSERVADORA, "BOVA11"]]
        curvas.index = pd.DatetimeIndex(curvas.index).as_unit("ns")
        datas = curvas.index
        caixa = indice_do_caixa(cdi, datas)
        sleeves = pd.DataFrame({"BR": curvas[CONSERVADORA],
                                **{nome: sleeve_em_reais(fechamentos[t], ptax, datas, taxa)
                                   for nome, (t, taxa) in TAXAS.items()}})
        valores_arm, escolhas = rotacao(sleeves, caixa.pct_change().fillna(0.0))
        inicio = valores_arm.index[0]
        valores = {ARM: valores_arm, CONSERVADORA: curvas[CONSERVADORA].loc[inicio:],
                   "BOVA11": curvas["BOVA11"].loc[inicio:]}
        rf = market_data.serie_rf_diaria(valores_arm.index, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        tabela = tabela_do_periodo({nome: valores[nome] for nome in (ARM, CONSERVADORA)}, valores["BOVA11"], rf,
                                   n_tentativas)
        for nome in (ARM, CONSERVADORA, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        tabela.loc[ARM, "P[> Conservadora]"] = psr_da_diferenca(valores[ARM], valores[CONSERVADORA])
        p_cons.append(float(tabela.loc[ARM, "P[> Conservadora]"]))
        p_bova.append(float(tabela.loc[ARM, "P aritmético"]))
        dsr.append(float(tabela.loc[ARM, "DSR"]))

        posse = posse_diaria(escolhas, valores_arm.index)
        partes = pd.crosstab(posse.index.year, posse).reindex(columns=["BR", "US", "Ouro", CAIXA], fill_value=0)
        partes = partes.div(partes.sum(axis=1), axis=0).assign(switches=trocas_por_ano(escolhas))
        print(f"\n=== Phase {fase}: {inicio:%d/%m/%Y} to {valores_arm.index[-1]:%d/%m/%Y}; first decision "
              f"{escolhas.index[0]:%d/%m/%Y}; {len(escolhas)} decisions, {int(partes['switches'].sum())} switches ===")
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        sleeve_cagr = {nome: cagr(sleeves[nome].loc[inicio:].dropna()) for nome in ("US", "Ouro")}
        print(f"Sleeve CAGRs on the same days (reported): US {sleeve_cagr['US']:.1%}, gold {sleeve_cagr['Ouro']:.1%}, "
              f"CDI {cagr(caixa.loc[inicio:]):.1%}")
        ganhos, anos = years_beating_all(valores[ARM], [valores["BOVA11"], valores[CONSERVADORA]])
        print(f"Years beating BOVA11 and the Conservadora (reported, not a gate): {ganhos} of {anos} "
              f"({ganhos / anos:.0%})")
        print("Sleeve shares of sessions by year, and switches:")
        print(partes.to_string(float_format=lambda x: f"{x:.0%}"))
        print(_year_table({nome: valores[nome] for nome in (ARM, CONSERVADORA, "BOVA11")})
              .to_string(float_format=lambda x: f"{x:.1%}"))

    passa = strategy_gate(p_cons, p_bova, dsr)
    print(f"\nStrategy gate: P[> Conservadora] {[round(p, 3) for p in p_cons]}, P[> BOVA11] "
          f"{[round(p, 3) for p in p_bova]}, DSR {[round(d, 3) for d in dsr]} (all >= {P_MIN}): "
          + ("CANDIDATE: pre-register a forward paper log against the same-day Conservadora" if passa else
             "not a candidate: H2 is closed; the allocation layer stays fixed weights"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
