"""
arvores_ler_ic.py - The tree forecasts' reading (docs/superpowers/specs/2026-11-16-arvores-design.md, section 4): the
213 gate on the 213 window, the paired differences, the edge rule and the last-month importances. Only reads files.

    python -X utf8 scripts/arvores_ler_ic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.qml_ic import IC_MINIMO, T_MINIMO, ic_do_alvo, passa_no_ic, t_newey_west  # noqa: E402
from scripts.qml_ler_ic import JANELA, PASTA, tabela_das_previsoes  # noqa: E402

ARVORES = ("RF", "GBR", "LGBM")
MODELOS = (*ARVORES, "Ridge", "KAN-1", "MLP", "composto")
CONTRA = ("Ridge", "MLP", "composto")
ATRIBUTOS = ("momentum", "baixa_vol", "dividendos", "formula_magica", "trending_value", "crescimento")


def main() -> int:
    tabela = tabela_das_previsoes(PASTA / "b3")
    inicio, fim = JANELA["b3"]
    tabela = tabela[tabela["month"].between(inicio, fim)]
    ics = {nome: ic_do_alvo(tabela, nome, "y") for nome in MODELOS}
    print(f"B3: monthly Spearman IC against the 3-month target, {inicio:%Y-%m} to {fim:%Y-%m}; "
          f"gate IC >= {IC_MINIMO} and Newey-West t >= {T_MINIMO}")
    linhas = {nome: {"months": len(serie), "IC": serie.mean(), "t": t_newey_west(serie), "share > 0": (serie > 0).mean(),
                     "passes": passa_no_ic(serie.mean(), t_newey_west(serie))} for nome, serie in ics.items()}
    print(pd.DataFrame(linhas).T.to_string(float_format=lambda valor: f"{valor:.3f}"))
    for nome in ARVORES:
        t_das = {}
        for outro in CONTRA:
            diferenca = (ics[nome] - ics[outro]).dropna()
            t_das[outro] = t_newey_west(diferenca)
            print(f"{nome} - {outro}: {diferenca.mean():+.4f}, t {t_das[outro]:+.2f}")
        print(f"{nome} by seed: " + ", ".join(f"seed {s} {ic_do_alvo(tabela, f'{nome}_{s}', 'y').mean():.4f}" for s in (0, 1, 2)))
        vantagem = bool(linhas[nome]["passes"]) and t_das["Ridge"] >= T_MINIMO and t_das["composto"] >= T_MINIMO
        print(f"{nome} edge (passes, and t >= {T_MINIMO} against Ridge and against the Composite): "
              f"{'CLAIMED' if vantagem else 'not claimed'}\n")
    importancias = {}
    for nome in ARVORES:
        with np.load(PASTA / "b3" / f"arvores_{nome}.npz") as arquivo:
            importancias[nome] = arquivo[f"{nome}_importancia"]
    print("Last-month impurity / split importance, normalized (descriptive):")
    print(pd.DataFrame(importancias, index=ATRIBUTOS).to_string(float_format=lambda valor: f"{valor:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
