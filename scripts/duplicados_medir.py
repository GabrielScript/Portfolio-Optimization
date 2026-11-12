"""
The one measurement of the refiled-statements fix (``docs/superpowers/specs/2026-11-11-duplicate-filings-design.md``,
committed before this ran): the Composite with the old pick and with ``core.valor.documentos_conhecidos``.

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/duplicados_medir.py [lake exports folder] > docs/duplicados-2026-11-11.txt

1. Attempt 51's replay in the three phases against its published curve: largest relative gap, decisions whose names
   change, CAGR before and after.
2. The Composite's portfolio on the record in force (27/10/2026) from the lake's current exports (the folder given;
   the main checkout's ``data/b3lake`` by default), before and after.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import classicas, crescimento_lucro, valor  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import cagr  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, paper_v3  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes  # noqa: E402
from scripts.avaliar_universo_dinamico import CFG_BT  # noqa: E402
from scripts.composto_variantes_avaliar import ANCHOR_TOL, recording  # noqa: E402
from scripts.paper_v2 import treino_do_dia  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, CURVAS_51  # noqa: E402

RECORD = pd.Timestamp("2026-10-27")
LAKE = Path(__file__).resolve().parent.parent.parent / "Portfolio-Optimization" / "data" / "b3lake"
READERS = (valor, crescimento_lucro, classicas)


def old_pick(fundamentos: pd.DataFrame, mes: pd.Timestamp) -> pd.DataFrame:
    """Every row received before ``mes``, refilings included: what the three readers saw before the fix."""
    return fundamentos[fundamentos["available_from"] < mes]


@contextmanager
def reading_with(pick: Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]) -> Iterator[None]:
    fixed = valor.documentos_conhecidos
    for modulo in READERS:
        modulo.documentos_conhecidos = pick
    try:
        yield
    finally:
        for modulo in READERS:
            modulo.documentos_conhecidos = fixed


def backtest() -> None:
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    nota = _value_score(avaliar_conservadora.carregar_mensal(FOLDER), com_crescimento=True)[0]
    publicadas = pd.read_parquet(CURVAS_51)
    precos = dados.precos.loc[:LAST_SESSION]
    print(f"1. Attempt 51's replay, data to {LAST_SESSION:%d/%m/%Y}")
    for fase in FASES:
        curvas, nomes = {}, {}
        for rotulo, pick in (("before", old_pick), ("after", valor.documentos_conhecidos)):
            registro: dict = {}
            with reading_with(pick):
                curvas[rotulo] = walk_forward_dinamico(
                    _acoes(precos.iloc[fase:]), dados.universo,
                    recording(estrategia_por_nota(nota, dados.setores), registro), CFG_BT, COMPOSITE,
                    Congelamento()).valores
            nomes[rotulo] = registro
        publicada = publicadas.loc[publicadas["fase"] == fase].set_index("date")["Composto"].reindex(curvas["after"].index)
        desvio = {rotulo: float((curva / publicada - 1.0).abs().max()) for rotulo, curva in curvas.items()}
        mudadas = [data for data, (_, antes) in nomes["before"].items() if antes != nomes["after"][data][1]]
        print(f"  phase {fase}: gap to the published curve before {desvio['before']:.1e}, after {desvio['after']:.1e} "
              f"(tolerance {ANCHOR_TOL:.0e}); decisions with other names {len(mudadas)} of {len(nomes['after'])}; "
              f"CAGR before {cagr(curvas['before']):.4%}, after {cagr(curvas['after']):.4%}", flush=True)
        for data in mudadas:
            antes, depois = nomes["before"][data][1], nomes["after"][data][1]
            print(f"    {data:%d/%m/%Y}: out {sorted(antes - depois)}, in {sorted(depois - antes)}")


def record_in_force(lake: Path) -> None:
    dados = paper_v3.ler_dados_do_composto(lake)
    carteiras = {}
    for rotulo, pick in (("before", old_pick), ("after", valor.documentos_conhecidos)):
        with reading_with(pick):
            nota = paper_v3.nota_do_composto(dados.lago.retornos_mensais, dados.lago.proventos, dados.balancos)
            pesos = estrategia_por_nota(nota, dados.lago.setores)(treino_do_dia(dados.lago, RECORD))
        carteiras[rotulo] = set(pesos[pesos > 0].index)
    print(f"2. The Composite on {RECORD:%d/%m/%Y} from {lake}: {len(carteiras['after'])} names; "
          f"out {sorted(carteiras['before'] - carteiras['after'])}, in {sorted(carteiras['after'] - carteiras['before'])}")
    print(f"   after: {sorted(carteiras['after'])}")


def main(argv: list[str]) -> int:
    backtest()
    record_in_force(Path(argv[0]) if argv else LAKE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
