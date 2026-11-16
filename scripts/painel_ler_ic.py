"""
painel_ler_ic.py - The broad panel's forecast reading (docs/superpowers/specs/2026-11-16-painel-amplo-design.md,
section 5): the 213 gate on the 213 window, the paired differences, the edge rule, each feature's own IC and the last
month's Ridge-29 coefficients. Only reads files.

    python -X utf8 scripts/painel_ler_ic.py > docs/painel-amplo-ic-2026-11-16.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.painel_amplo import NOVAS  # noqa: E402
from core.qml_ic import IC_MINIMO, T_MINIMO, ic_do_alvo, passa_no_ic, t_newey_west  # noqa: E402
from qpo_kaggle.qml_pacote import ler_pacote  # noqa: E402
from scripts.qml_ler_ic import JANELA, PASTA, tabela_das_previsoes  # noqa: E402

NOVOS = ("Ridge29", "PLS29")
MODELOS = (*NOVOS, "Ridge", "RF", "MLP", "composto")
SEIS = ("momentum", "baixa_vol", "dividendos", "formula_magica", "trending_value", "crescimento")
LOCAL = Path(__file__).resolve().parent.parent / "painel_run"


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
    for nome in NOVOS:
        t_das = {}
        for outro in ("Ridge", "composto", "RF"):
            diferenca = (ics[nome] - ics[outro]).dropna()
            t_das[outro] = t_newey_west(diferenca)
            print(f"{nome} - {outro}: {diferenca.mean():+.4f}, t {t_das[outro]:+.2f}")
        vantagem = bool(linhas[nome]["passes"]) and t_das["Ridge"] >= T_MINIMO and t_das["composto"] >= T_MINIMO
        print(f"{nome} edge (passes, and t >= {T_MINIMO} against Ridge and against the Composite): "
              f"{'CLAIMED' if vantagem else 'not claimed'}\n")

    pacote = ler_pacote(LOCAL / "pacote_amplo.npz")
    nomes = (*SEIS, *NOVAS)
    meses = pd.to_datetime([f"{m // 12}-{m % 12 + 1:02d}-01" for m in pacote.mes])
    atributos = pd.DataFrame(pacote.theta, columns=nomes).assign(month=meses, y=pacote.y)
    atributos = atributos[atributos["month"].between(inicio, fim)]
    print("Each feature alone (descriptive): monthly Spearman IC against the target, missing rows (angle 0) left out")
    proprias = {}
    for nome in nomes:
        parte = atributos.loc[atributos[nome] != 0.0, ["month", nome, "y"]]
        serie = ic_do_alvo(parte, nome, "y")
        proprias[nome] = {"IC": serie.mean(), "t": t_newey_west(serie), "rows/month": len(parte) / max(len(serie), 1)}
    print(pd.DataFrame(proprias).T.to_string(float_format=lambda valor: f"{valor:.3f}"))
    with np.load(LOCAL / "painel_previsoes.npz") as arquivo:
        coeficientes = pd.Series(arquivo["Ridge29_coeficientes"], index=nomes)
    print("\nRidge-29, last month's coefficients (descriptive):")
    print(coeficientes.sort_values().to_string(float_format=lambda valor: f"{valor:+.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
