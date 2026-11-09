"""
qml_qubo_linha.py - the QUBO row of the QML comparison table, on its own run (test branch only).

The consolidated QML run is executed with ``--sem-qubo`` because the exact QUBO solve dominates the
wall-clock. This script runs only the QUBO arm (``core.qpo`` selection then maximum Sharpe, the same
strategy object the consolidated run uses) on the same frozen lake exports and the same phases, and
tables it with the identical ``_linha`` metrics of ``scripts/qml_avaliar.py``, so the comparison keeps
one definition of every number. Descriptive, not a pre-registered reading.

    .venv-qpo/Scripts/python -X utf8 scripts/qml_qubo_linha.py [--fases 0 21 42]
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.config import DIAS_UTEIS_ANO  # noqa: E402
from core.qpo import CFG_MAX_SHARPE, K, qubo_selector, selection_then_max_sharpe  # noqa: E402
from core.qubo_universo import FULL_UNIVERSE, SECTOR_CAP, anneal_solver  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, qubo_universo_avaliar  # noqa: E402
from scripts.avaliar_robustez import _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.curvas_ao_vivo import ARQUIVOS, LAGO, congelar  # noqa: E402
from scripts.qml_avaliar import COMPOSITE, QUBO, _rodar, _tabela  # noqa: E402

SAIDA = RAIZ / "docs" / "qml-qubo-2026-11-09.txt"
TABELA = RAIZ / "docs" / "dados" / "qml-qubo-linha.csv"


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    fases = tuple(int(valor) for valor in argv[argv.index("--fases") + 1:]) if "--fases" in argv else (0,)
    if not fases:
        raise ValueError("--fases needs at least one phase, e.g. --fases 0 21 42")
    with tempfile.TemporaryDirectory(prefix="qml-qubo-") as temporaria:
        entradas = Path(temporaria)
        somas = congelar(LAGO, entradas, ARQUIVOS)
        avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = entradas
        qubo_universo_avaliar.FOLDER = qubo_universo_avaliar.PASTA_CVM = entradas
        dados = avaliar_robustez.carregar()
        mensal = avaliar_conservadora.carregar_mensal(entradas)
        nota, _ = qubo_universo_avaliar._value_score(mensal, com_crescimento=True)
        ultima = dados.precos.index.max()
        cdi = daily_cdi(ANOS_DE_CDI, ultima.date(), path=entradas / "cdi_daily.parquet")
        cfg_ms = replace(CFG_MAX_SHARPE, taxa_livre_risco=float((1 + cdi.iloc[-1]) ** DIAS_UTEIS_ANO - 1))

        linhas = []
        with SAIDA.open("w", encoding="utf-8", newline="\n") as saida, contextlib.redirect_stdout(saida):
            print(f"QUBO row of the QML table (descriptive, test branch). Lake exports sha256: {somas}")
            for fase in fases:
                precos = dados.precos.loc[:ultima].iloc[fase:]
                acoes = _acoes(precos)
                print(f"\n=== Phase {fase}: {acoes.index[0]:%d/%m/%Y} to {acoes.index[-1]:%d/%m/%Y} ===",
                      flush=True)
                qubo = selection_then_max_sharpe(
                    qubo_selector(dados.setores, FULL_UNIVERSE, K,
                                  anneal_solver(dados.setores, SECTOR_CAP, {}), {}),
                    mensal.retornos, mensal.proventos, dados.setores, cfg_ms, cdi, None)
                curvas = {COMPOSITE: _rodar(acoes, dados.universo, estrategia_por_nota(nota, dados.setores),
                                            COMPOSITE, inicio),
                          QUBO: _rodar(acoes, dados.universo, qubo, QUBO, inicio)}
                bova = referencias(precos, CFG_BT)["BOVA11"]
                rf = cdi.reindex(curvas[COMPOSITE].valores.index).fillna(0.0)
                tabela = _tabela(curvas, bova.valores, rf, curvas[COMPOSITE].valores)
                tabela.insert(0, "Fase", fase)
                print(tabela.to_string(float_format=lambda x: f"{x:.4f}"))
                linhas.append(tabela)
        final = pd.concat(linhas)
        final.to_csv(TABELA, encoding="utf-8", lineterminator="\n")
        print(final.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n{SAIDA.relative_to(RAIZ)} and {TABELA.relative_to(RAIZ)} written "
          f"({time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
