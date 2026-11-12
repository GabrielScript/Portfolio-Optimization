"""
Stage 1 of the insider reading (``docs/superpowers/specs/2026-11-12-informados-design.md``, committed in ``fd57a77``
before any return-based number): the monthly IC of INS, read once.

    python -X utf8 scripts/informados_ic.py > docs/informados-ic-2026-11-12.txt

Harness of DECISOES 79 / 161 (``avaliar_valor.ics_e_cobertura``): monthly Spearman IC of the month's score against
the month's total return in the month's universe; gate IC >= 0.02 and t >= 2 over the months with a score. Also
coverage, the two halves, and the rank correlation with the Conservadora, value and profit-growth scores. No network.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.informados import insider_score, load_filings  # noqa: E402
from core.inovemetria.data import load  # noqa: E402
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
    filings, sha256_vlmo = load_filings()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_inov | sha256_vlmo}")

    nota = insider_score(filings, inovacao.roots)
    meses = pd.date_range(*avaliar_valor.PERIODO_DO_IC, freq="MS")
    colunas = mensal.retornos.columns
    ics, cobertura = avaliar_valor.ics_e_cobertura(nota, mensal, dados.universo)
    ics = ics.dropna()
    meio = ics.index[len(ics) // 2]
    periodos = {f"{ics.index[0]:%Y-%m} to {ics.index[-1]:%Y-%m} (gate)": (ics.index[0], ics.index[-1]),
                "first half": (ics.index[0], meio - pd.DateOffset(months=1)), "second half": (meio, ics.index[-1])}
    print(f"\n=== INS: monthly Spearman IC; months with a score {len(ics)}; names scored per month (months with "
          f"any): min {cobertura[cobertura > 0].min()}, median {cobertura[cobertura > 0].median():.0f}, max "
          f"{cobertura.max()}")
    print(pd.DataFrame({rotulo: resumo_do_ic(ics.loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T
          .to_string(float_format=lambda x: f"{x:.3f}"))
    painel = painel_da_nota(nota, dados.universo, meses, colunas)
    outras = {"Conservadora": lambda mes, candidatos: conservadora(mensal.retornos, mensal.proventos, mes, candidatos),
              "value": nota_media_dos_escores(*avaliar_valor.notas(balancos, mensal).values()),
              "profit growth": nota_de_crescimento(balancos.fundamentos)}
    for outro, nota_outra in outras.items():
        correlacoes = ic_por_mes(painel, painel_da_nota(nota_outra, dados.universo, meses, colunas)).dropna()
        print(f"  rank correlation with {outro}: mean {correlacoes.mean():.3f} over {len(correlacoes)} months")
    print(f"  share of scored names at +1 (only purchases) {float((painel == 1).sum().sum() / painel.notna().sum().sum()):.2f}"
          f", at -1 (only sales) {float((painel == -1).sum().sum() / painel.notna().sum().sum()):.2f}")
    passou = avaliar_valor.PORTAO.aprova(ics)
    print(f"  gate ({avaliar_valor.PORTAO.nome}): {'PASSED' if passou else 'not passed'}")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
