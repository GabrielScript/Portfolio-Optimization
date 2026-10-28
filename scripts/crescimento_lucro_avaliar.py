"""
The profit-growth signal's pre-registered IC test (DECISOES 157, `docs/superpowers/specs/2026-10-27-crescimento-lucro-design.md`,
committed in `ba13304` before attempt 48 was read). Run once:

    python -X utf8 scripts/crescimento_lucro_avaliar.py > docs/crescimento-lucro-2026-10-27.txt

Signal (`core.crescimento_lucro.nota_de_crescimento`): mean of the normal scores of dNI and dGP (year-on-year change
over the total assets of a year before), point in time from the frozen CVM folder. Test: monthly Spearman IC against the
month's total return in the month's universe, 2012-01 to 2026-09 (`avaliar_valor.ics_e_cobertura`, DECISOES 79's
harness); gate IC >= 0.02 and t >= 2. Reported: subperiods, months with IC > 0, names scored, and the monthly rank
correlation with the Conservadora's and the value score (the overlap a composite would dilute). No network.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.sinais import conservadora  # noqa: E402
from core.valor import painel_da_nota  # noqa: E402
from core.views import nota_media_dos_escores  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

PASTA_CVM = FOLDER.parent / "2026-10-24-cvm"


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, sha256_cvm = avaliar_valor.carregar_balancos(PASTA_CVM)
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm}")

    crescimento = nota_de_crescimento(balancos.fundamentos)
    ics, cobertura = avaliar_valor.ics_e_cobertura(crescimento, mensal, dados.universo)
    periodos = {"2012-2026 (gate)": avaliar_valor.PERIODO_DO_IC,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in avaliar_valor.SUBPERIODOS_DO_IC.items()}}
    print(f"\nProfit growth (dNI, dGP over lagged assets): monthly Spearman IC, score of the month x total return of the "
          f"month, universe of the month; names scored per month: min {cobertura.min()}, median "
          f"{cobertura.median():.0f}, max {cobertura.max()}")
    print(pd.DataFrame({rotulo: resumo_do_ic(ics.loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T
          .to_string(float_format=lambda x: f"{x:.3f}"))

    meses = pd.date_range(*avaliar_valor.PERIODO_DO_IC, freq="MS")
    colunas = mensal.retornos.columns
    painel = painel_da_nota(crescimento, dados.universo, meses, colunas)
    outras = {"Conservadora": lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos),
              "value (Magic Formula + Trending Value)":
                  nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values())}
    for nome, nota in outras.items():
        correlacoes = ic_por_mes(painel, painel_da_nota(nota, dados.universo, meses, colunas)).dropna()
        print(f"Rank correlation with the {nome} score: mean {correlacoes.mean():.3f} over {len(correlacoes)} months")

    aprovada = avaliar_valor.PORTAO.aprova(ics.loc[avaliar_valor.PERIODO_DO_IC[0]:avaliar_valor.PERIODO_DO_IC[1]])
    print(f"\nGate ({avaliar_valor.PORTAO.nome}, 2012-01 to 2026-09): {'PASSED' if aprovada else 'NOT passed'}. "
          + ("Next: pre-register the QPO attempt with the Conservadora + value + profit growth composite."
             if aprovada else "The QPO line stays closed: no further mu attempt without another signal that passes."))
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
