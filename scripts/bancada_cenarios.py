"""
bancada_cenarios.py - the five-grade scenario Monte Carlo bench (docs/superpowers/specs/2026-10-28-bancada-cenarios-
design.md, DECISOES 165). Descriptive: no gate, no trial. No network: every input is frozen and hash-checked.

    python -X utf8 scripts/bancada_cenarios.py > docs/bancada-cenarios-2026-10-28.txt

Sleeves: the Conservadora and BOVA11 (phase-0 curves saved by attempt 51's reading) and the CDI; the composite is out
because attempt 51 was not a candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.bancada import (GRAUS, caminhos, componentes, cortes, estatisticas, grau_de, graus, resumir_meses,  # noqa: E402
                          sortear, transicoes)
from core.cdi_lago import daily_cdi  # noqa: E402
from core.vencedoras import indice_do_caixa  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402

CURVES_FILE = FOLDER.parent / "composto-curvas-2026-10-28.parquet"
MACRO_FOLDER = FOLDER.parent / "2026-10-21-macro"
FIM_DA_JANELA = pd.Timestamp("2026-09-30")        # the last complete month inside the curves (they stop on 16/10)
SLEEVES = ("Conservadora", "BOVA11", "CDI")
PARES = [("Conservadora", "BOVA11"), ("Conservadora", "CDI"), ("BOVA11", "CDI")]
CAMINHOS = 10_000
HORIZONTES = (12, 36, 60)                          # months
SEMENTE = 20261028
NOMES_DOS_GRAUS = {5: "Optimistic", 4: "Neutral-optimistic", 3: "Neutral", 2: "Neutral-pessimistic", 1: "Pessimistic"}

CAVEATS = """Caveats (read before the numbers):
- Descriptive bench, not a forecast and not a signal: it resamples 2010-2026 history; it cannot add information.
- The grades are labeled in-sample (full-window z and quintile cut points): they describe environments, not timing.
- A sampled month brings its own CDI level (2% in 2020, 14% in 2016): comparisons inside a path (P[a > b]) are
  consistent; terminal levels are not "what to expect from today".
- About 40 months per grade: each row is noisy. Taxes ignored; the curves are net of 10 bps per side.
- Method: whole historical months resampled jointly (all sleeves take the same month), not the t-copula mentioned in
  the chat. IVVB11 left out (history from 04/2014). No mixes, no leverage (steps 3 and 5)."""


def main() -> int:
    sha256 = (conferir_hashes(FOLDER.parent, [CURVES_FILE.name]) | conferir_hashes(FOLDER, ["universe_prices.parquet"])
              | conferir_hashes(MACRO_FOLDER, ["cdi_daily.parquet"]))
    curvas = pd.read_parquet(CURVES_FILE).query("fase == 0").set_index("date")[["Conservadora", "BOVA11"]]
    curvas.index = pd.DatetimeIndex(curvas.index)
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO_FOLDER / "cdi_daily.parquet")
    bruto = pd.read_parquet(MACRO_FOLDER / "cdi_daily.parquet")
    cdi_pct = pd.Series(bruto["cdi_pct"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(bruto["date"])))
    precos = pd.read_parquet(FOLDER / "universe_prices.parquet", columns=["date", "BOVA11"])
    bova = pd.Series(precos["BOVA11"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(precos["date"]))).dropna()

    valores = curvas.assign(CDI=indice_do_caixa(cdi, curvas.index)).loc[:FIM_DA_JANELA, list(SLEEVES)]
    resumo = resumir_meses(valores)
    comp = componentes(bova, cdi_pct, resumo.meses - pd.Timedelta(days=1))     # the state when each month starts
    comp.index = resumo.meses
    grau = graus(comp).to_numpy()
    contagem, matriz = transicoes(grau)
    hoje = componentes(bova, cdi_pct, pd.DatetimeIndex([LAST_SESSION]))
    grau_hoje = int(grau_de(hoje, comp).iloc[0])

    print(f"Files (sha256): {sha256}")
    print(CAVEATS)
    print(f"\nWindow: {resumo.meses[0]:%m/%Y} to {resumo.meses[-1]:%m/%Y} ({len(resumo.meses)} months); "
          f"score = z(trend 12m) - z(vol 63d) - z(change of the annual CDI over 6m); cut points "
          f"{np.round(cortes(comp), 3).tolist()}")
    rotulo = [f"{g} {NOMES_DOS_GRAUS[g]}" for g in range(1, GRAUS + 1)]
    por_grau = pd.DataFrame(np.expm1(resumo.fim), columns=SLEEVES).groupby(grau).mean().set_axis(rotulo)
    por_grau.insert(0, "months", pd.Series(grau).value_counts().sort_index().to_numpy())
    print("\nMonths per grade and each sleeve's mean monthly return in them:")
    print(por_grau.to_string(float_format=lambda x: f"{x:.2%}"))
    print("\nTransition counts (row = grade a month started in, column = the next month's):")
    print(pd.DataFrame(contagem, index=rotulo, columns=range(1, GRAUS + 1)).to_string())

    gerador = np.random.default_rng(SEMENTE)
    linhas = {}
    for grau0 in range(1, GRAUS + 1):
        sorteio = sortear(grau, matriz, grau0, CAMINHOS, max(HORIZONTES), gerador)
        for meses in HORIZONTES:
            terminal, mdd = caminhos(resumo, sorteio[:, :meses])
            linhas[(rotulo[grau0 - 1], f"{meses // 12}y")] = estatisticas(terminal, mdd, SLEEVES, PARES)
    tabela = pd.DataFrame(linhas).T
    tabela.index.names = ["starting grade", "horizon"]
    probabilidades = [f"P[{a} > {b}]" for a, b in PARES]
    print(f"\nWin shares over {CAMINHOS:,} paths per starting grade (seed {SEMENTE}):")
    print(tabela[probabilidades].to_string(float_format=lambda x: f"{x:.0%}"))
    for nome in SLEEVES[:2]:
        colunas = [f"{nome} ret p5", f"{nome} ret p50", f"{nome} ret p95", f"{nome} MDD p50", f"{nome} MDD worst 5%"]
        print(f"\n{nome}: terminal return quantiles and maximum drawdown (terminal levels carry the sampled CDI era):")
        print(tabela[colunas].to_string(float_format=lambda x: f"{x:.0%}"))
    print(f"\nToday (data to {LAST_SESSION:%d/%m/%Y}, before the 26/10 crash; a description, not a signal): grade "
          f"{grau_hoje} {NOMES_DOS_GRAUS[grau_hoje]}; trend {hoje['trend'].iloc[0]:.1%}, vol {hoje['vol'].iloc[0]:.1%}, "
          f"CDI change {hoje['dcdi'].iloc[0]:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
