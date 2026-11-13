"""
Stage 1 of the Inovemetria reading (``docs/superpowers/specs/2026-11-12-inovemetria-design.md``, committed in
``b7aad73`` before any return-based number): the monthly IC of each family and of INOV, read once.

    python -X utf8 scripts/inovemetria_ic.py > docs/inovemetria-ic-2026-11-12.txt

Harness of DECISOES 79 / 161 (``avaliar_valor.ics_e_cobertura``): monthly Spearman IC of the month's score against
the month's total return in the month's universe, 2012-01 to 2026-09; gate IC >= 0.02 and t >= 2. Months without a
score are left out. Also: subperiods, coverage, and the monthly rank correlation with the Conservadora, value and
profit-growth scores. No network.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.inovemetria.data import load  # noqa: E402
from core.inovemetria.signals import families  # noqa: E402
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
    inovacao, sha256_inov = load()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_inov}")

    sinais = families(inovacao, dados.setores, mensal.retornos)
    sinais["INOV"] = nota_media_dos_escores(*sinais.values())
    meses = pd.date_range(*avaliar_valor.PERIODO_DO_IC, freq="MS")
    colunas = mensal.retornos.columns
    outras = {"Conservadora": lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos),
              "value": nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
              "profit growth": nota_de_crescimento(balancos.fundamentos)}
    paineis_outras = {nome: painel_da_nota(nota, dados.universo, meses, colunas) for nome, nota in outras.items()}
    periodos = {"2012-2026 (gate)": avaliar_valor.PERIODO_DO_IC,
                **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in avaliar_valor.SUBPERIODOS_DO_IC.items()}}

    aprovadas = []
    for nome, nota in sinais.items():
        ics, cobertura = avaliar_valor.ics_e_cobertura(nota, mensal, dados.universo)
        ics = ics.dropna()
        print(f"\n=== {nome}: monthly Spearman IC; months with a score {len(ics)}; names scored per month (months "
              f"with any): min {cobertura[cobertura > 0].min()}, median {cobertura[cobertura > 0].median():.0f}, "
              f"max {cobertura.max()} ({time.perf_counter() - inicio:.0f} s)", flush=True)
        print(pd.DataFrame({rotulo: resumo_do_ic(ics.loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T
              .to_string(float_format=lambda x: f"{x:.3f}"))
        painel = painel_da_nota(nota, dados.universo, meses, colunas)
        for outro, painel_outro in paineis_outras.items():
            correlacoes = ic_por_mes(painel, painel_outro).dropna()
            print(f"  rank correlation with {outro}: mean {correlacoes.mean():.3f} over {len(correlacoes)} months")
        passou = avaliar_valor.PORTAO.aprova(ics.loc[avaliar_valor.PERIODO_DO_IC[0]:avaliar_valor.PERIODO_DO_IC[1]])
        print(f"  gate ({avaliar_valor.PORTAO.nome}): {'PASSED' if passou else 'not passed'}")
        if passou and nome != "INOV":
            aprovadas.append(nome)

    print(f"\nFamilies that pass (INOV-gate's members): {aprovadas or 'none'}")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
