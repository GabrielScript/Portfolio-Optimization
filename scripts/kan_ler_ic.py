"""
kan_ler_ic.py - The KAN forecast's reading (docs/superpowers/specs/2026-11-07-kan-design.md, section 4): the 213 gate on
the 213 window, the paired differences, the edge rule and the learned curves. Only reads files.

    python -X utf8 scripts/kan_ler_ic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.qml_ic import IC_MINIMO, T_MINIMO, ic_do_alvo, passa_no_ic, t_newey_west  # noqa: E402
from scripts.qml_ler_ic import JANELA, PASTA, tabela_das_previsoes  # noqa: E402

MODELOS = ("KAN-1", "KAN-2", "Ridge", "MLP", "VQC", "Fourier", "composto")
DIFERENCAS = (("KAN-1", "Ridge"), ("KAN-1", "MLP"), ("KAN-1", "composto"), ("KAN-2", "KAN-1"))
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
    t_das = {}
    for um, outro in DIFERENCAS:
        diferenca = (ics[um] - ics[outro]).dropna()
        t_das[(um, outro)] = t_newey_west(diferenca)
        print(f"{um} - {outro}: {diferenca.mean():+.4f}, t {t_das[(um, outro)]:+.2f}")
    for nome in ("KAN-1", "KAN-2"):
        print(f"{nome} by seed: " + ", ".join(f"seed {s} {ic_do_alvo(tabela, f'{nome}_{s}', 'y').mean():.4f}" for s in (0, 1, 2)))
    kan1 = linhas["KAN-1"]
    vantagem = bool(kan1["passes"]) and t_das[("KAN-1", "Ridge")] >= T_MINIMO and t_das[("KAN-1", "composto")] >= T_MINIMO
    print(f"KAN edge (KAN-1 passes, and t >= {T_MINIMO} against Ridge and against the Composite): "
          f"{'CLAIMED' if vantagem else 'not claimed'}")

    curvas = pd.read_csv(PASTA / "b3" / "kan_curvas.csv")
    curvas["atributo"] = curvas["atributo"].map(dict(enumerate(ATRIBUTOS)))
    resumo = curvas.groupby(["atributo", "semente"])["nao_linearidade"].first().unstack()
    print("\nKAN-1 curves, last month (descriptive): non-linearity 1 - R^2 of the best line, by seed")
    print(resumo.to_string(float_format=lambda valor: f"{valor:.3f}"))
    media = curvas.groupby(["atributo", "u"])["phi"].mean().unstack("atributo")
    print("\nphi averaged over seeds, at 5 of the 21 points (angle -pi/2 = worst-ranked, +pi/2 = best-ranked)")
    print(media.iloc[::5].to_string(float_format=lambda valor: f"{valor:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
