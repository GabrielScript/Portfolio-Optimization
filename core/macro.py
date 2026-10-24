"""
macro.py - The screen's macro panel: the median expectations of the BCB's Focus survey (``core.focus``, live from
the Olinda API) for this year and the next, and how the current year's median moved in four weeks. Information only
(DECISOES 124): the macro filters were tested on BOVA11 and did not pass (DECISOES 112), so nothing here changes the
Fundo Alepha plan.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

QUATRO_SEMANAS = pd.Timedelta(days=28)


def resumo_focus(anuais: Mapping[str, pd.DataFrame], ano: int) -> pd.DataFrame:
    """One row per indicator (``anuais``: name -> ``focus.baixar_focus_anual``): the last survey's median for
    ``ano`` and ``ano + 1``, the change of the ``ano`` median since the last survey at least four weeks older, and the
    survey date."""
    linhas = {}
    for nome, tabela in anuais.items():
        ultima = tabela["data"].max()
        do_dia = tabela[tabela["data"] == ultima].set_index("ano_referencia")["mediana"]
        antes = tabela[(tabela["data"] <= ultima - QUATRO_SEMANAS) & (tabela["ano_referencia"] == ano)]
        mudanca = (do_dia.get(ano, float("nan")) - antes.sort_values("data")["mediana"].iloc[-1]
                   if not antes.empty else float("nan"))
        linhas[nome] = {str(ano): do_dia.get(ano, float("nan")), str(ano + 1): do_dia.get(ano + 1, float("nan")),
                        f"Mudança em 4 semanas ({ano})": mudanca, "Pesquisa": ultima}
    return pd.DataFrame.from_dict(linhas, orient="index")
