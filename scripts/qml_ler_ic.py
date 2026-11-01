"""
qml_ler_ic.py - The forecast's reading (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 3). Only reads the
files of ``qml_painel`` and ``qml_prever``: one row per model, then the paired differences.

    python -X utf8 scripts/qml_ler_ic.py b3
    python -X utf8 scripts/qml_ler_ic.py us
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.qml_ic import IC_MINIMO, T_MINIMO, ic_do_alvo, passa_no_ic, t_newey_west  # noqa: E402

PASTA = Path("D:/b3-datalake/exports/qml")
# the frozen returns end in a partial month, so the last whole target starts 3 months before the partial one
JANELA = {"b3": (pd.Timestamp("2012-01-01"), pd.Timestamp("2026-07-01")),
          "us": (pd.Timestamp("2019-11-01"), pd.Timestamp("2026-07-01"))}
MODELOS = ("VQC", "Fourier", "MLP", "Ridge", "composto")
DIFERENCAS = (("VQC", "Fourier"), ("VQC", "MLP"), ("VQC", "composto"))


def tabela_das_previsoes(pasta: Path) -> pd.DataFrame:
    chaves = pd.read_parquet(pasta / "chaves.parquet")
    painel = pd.read_parquet(pasta / "painel.parquet", columns=["month", "security", "y", "composto"])
    with np.load(pasta / "previsoes.npz") as previsoes:
        modelos = {nome: previsoes[nome] for nome in previsoes.files}
    return chaves.assign(**modelos).merge(painel, on=["month", "security"], validate="one_to_one")


def main(argv: list[str]) -> int:
    mercado = argv[0]
    tabela = tabela_das_previsoes(PASTA / mercado)
    inicio, fim = JANELA[mercado]
    tabela = tabela[tabela["month"].between(inicio, fim)]
    ics = {nome: ic_do_alvo(tabela, nome, "y") for nome in MODELOS if nome in tabela.columns}
    print(f"{mercado}: monthly Spearman IC against the 3-month target, {inicio:%Y-%m} to {fim:%Y-%m}; "
          f"gate IC >= {IC_MINIMO} and Newey-West t >= {T_MINIMO}")
    linhas = {nome: {"months": len(serie), "IC": serie.mean(), "t": t_newey_west(serie),
                     "share > 0": (serie > 0).mean(), "passes": passa_no_ic(serie.mean(), t_newey_west(serie))}
              for nome, serie in ics.items()}
    print(pd.DataFrame(linhas).T.to_string(float_format=lambda valor: f"{valor:.3f}"))
    for um, outro in DIFERENCAS:
        if um in ics and outro in ics:
            diferenca = (ics[um] - ics[outro]).dropna()
            print(f"{um} - {outro}: {diferenca.mean():+.4f}, t {t_newey_west(diferenca):+.2f}")
    sementes = [nome for nome in tabela.columns if nome.startswith("VQC_")]
    if sementes:
        print("VQC by seed: " + ", ".join(f"{nome} {ic_do_alvo(tabela, nome, 'y').mean():.4f}" for nome in sementes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
