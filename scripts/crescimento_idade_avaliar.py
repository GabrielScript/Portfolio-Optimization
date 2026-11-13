"""
The Composite rerun with the profit-growth age rule (branch fix/crescimento-idade), one phase per process, against
its saved curve of DECISOES 350 (the code before the fix). A data-handling fix, as DECISOES 349: not an arm, N stays.

    python -X utf8 scripts/crescimento_idade_avaliar.py 0 > docs/crescimento-idade-fase0-2026-11-13.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.qubo_universo import composite_score  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_robustez import _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE  # noqa: E402
from scripts.qubo_universo_avaliar import with_retries  # noqa: E402
from scripts.xquad_avaliar import COMPOSITE, metricas  # noqa: E402

FIXED = "Composite (age rule)"
DATA = FOLDER.parent
GRID = "fermionica-grade-curvas-fase{}-2026-11-11.parquet"
PASTA_CVM = FOLDER.parent / "2026-10-24-cvm"


def main(fase: int) -> int:
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, _ = avaliar_valor.carregar_balancos(PASTA_CVM)
    nota = composite_score(lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c),
                           nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
                           nota_de_crescimento(balancos.fundamentos))
    precos = dados.precos.loc[:LAST_SESSION].iloc[fase:]
    novo = walk_forward_dinamico(_acoes(precos), dados.universo, estrategia_por_nota(nota, dados.setores), CFG_BT,
                                 FIXED, Congelamento())
    indice = novo.valores.index
    salva = pd.read_parquet(DATA / GRID.format(fase)).set_index("date")[COMPOSITE].reindex(indice)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = with_retries(lambda: market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE))
    rf = market_data.serie_rf_diaria(indice, cdi, selic.valor)
    curvas = {COMPOSITE: salva, FIXED: novo.valores,
              "BOVA11": referencias(precos, CFG_BT)["BOVA11"].valores.reindex(indice),
              "CDI": pd.Series(np.cumprod(1.0 + rf.to_numpy()), index=indice)}
    print(f"=== Phase {fase}: {indice[0]:%d/%m/%Y} to {indice[-1]:%d/%m/%Y}; max gap to the saved curve "
          f"{float((novo.valores / salva - 1).abs().max()):.4f}; turnover {novo.turnover_anual_medio:.2f}")
    print(metricas(curvas, rf, 132).to_string(float_format=lambda x: f"{x:.4f}"))
    alvos = novo.pesos_alvo
    print(f"decisions {len(alvos)}")
    alvos.to_csv(f"docs/crescimento-idade-alvos-fase{fase}-2026-11-13.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1])))
