"""
Daily curves of the Composite, the Fermiônica (LogDet-15 t0.5) and von Neumann (theta 0.5) on the code after the profit-growth age rule
(DECISOES 360), one phase per process, saved for readings that combine them with other legs (the crypto lake's
Bitcoin sleeve). Anchor: the Composite's CAGR equals the one printed in DECISOES 360 for the phase.

    python -X utf8 scripts/curvas_pos_idade.py 0     # -> docs/dados/pos-idade-curvas-fase0-2026-11-13.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.diversidade import von_neumann  # noqa: E402
from core.fermionica import estrategia_fermionica  # noqa: E402
from core.metrics import cagr  # noqa: E402
from core.qubo_universo import composite_score  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_robustez import _acoes  # noqa: E402
from scripts.avaliar_universo_dinamico import CFG_BT  # noqa: E402
from scripts.fermionica_grade_avaliar import memoized  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402

ANCHOR = {0: 0.2179, 21: 0.2034, 42: 0.1922}       # DECISOES 360, four decimals
OUT = FOLDER.parent / "pos-idade-curvas-fase{}-2026-11-13.parquet"


def main(fase: int) -> int:
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, _ = avaliar_valor.carregar_balancos(FOLDER.parent / "2026-10-24-cvm")
    nota = memoized(composite_score(lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c),
                                    nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
                                    nota_de_crescimento(balancos.fundamentos)))
    acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

    def rodar(estrategia, nome):
        return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento()).valores

    composite = rodar(estrategia_por_nota(nota, dados.setores), "Composite")
    if abs(cagr(composite.dropna()) - ANCHOR[fase]) > 5e-5:
        raise SystemExit(f"phase {fase}: anchor failed, CAGR {cagr(composite.dropna()):.5f} vs {ANCHOR[fase]}")
    fermi = rodar(estrategia_fermionica(nota, dados.setores, 0.5), "LogDet-15 t0.5")
    neumann = rodar(estrategia_fermionica(nota, dados.setores, 0.5, diversity=von_neumann), "vonNeumann-15 t0.5")
    tabela = pd.DataFrame({"Composite": composite, "LogDet-15 t0.5": fermi, "vonNeumann-15 t0.5": neumann}).rename_axis("date").reset_index()
    tabela.to_parquet(str(OUT).format(fase), index=False)
    print(f"phase {fase}: Composite {cagr(composite.dropna()):.4f}, Fermiônica {cagr(fermi.dropna()):.4f}, von Neumann {cagr(neumann.dropna()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1])))
