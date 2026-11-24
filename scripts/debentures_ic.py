"""
Stage 1 of the debenture reading (``docs/superpowers/specs/2026-11-24-debentures-design.md``, committed in
``7392e64`` before any return-based number): the monthly IC of CRD-N and CRD-V, read once.

    ../Portfolio-Optimization/.venv/Scripts/python -X utf8 scripts/debentures_ic.py > docs/debentures-ic-<date>.txt

Harness of DECISOES 79 / 161 (``avaliar_valor.ics_e_cobertura``): monthly Spearman IC of the month's score against
the month's total return in the month's universe, 2012-01 to 2026-09; months with fewer than 20 scored names dropped
before the gate (IC >= 0.02 and t >= 2). Described: coverage, the halves 2012-2018 and 2019-2026, and the rank
correlation with the Composite score. Reads the SND files on D: (hash checked). No network.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.debentures import MIN_NAMES, debenture_scores, gated_ics, load_inputs  # noqa: E402
from core.poder import ic_por_mes  # noqa: E402
from core.valor import painel_da_nota  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_conservadora import resumo_do_ic  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402
from scripts.qubo_universo_avaliar import _value_score  # noqa: E402

HALVES = {"2012-2018": ("2012-01-01", "2018-12-01"), "2019-2026": ("2019-01-01", "2026-09-01")}


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    composto, sha256_cvm = _value_score(mensal, com_crescimento=True)
    medida, raizes, sha256_snd = load_inputs(FOLDER / "sectors.parquet")
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_snd}")
    print(f"Issuers with a CNPJ root: {len(raizes)}; (month, root) measures: {len(medida)}")

    meses = pd.date_range(*avaliar_valor.PERIODO_DO_IC, freq="MS")
    colunas = mensal.retornos.columns
    painel_composto = painel_da_nota(composto, dados.universo, meses, colunas)
    for nome, nota in debenture_scores(medida, raizes).items():
        ics, cobertura = avaliar_valor.ics_e_cobertura(nota, mensal, dados.universo)
        portao = gated_ics(ics, cobertura)
        com_nota = cobertura[cobertura > 0]
        print(f"\n=== {nome}: monthly Spearman IC; months with a score {int((cobertura > 0).sum())}, kept "
              f"(>= {MIN_NAMES} names) {len(portao)}; names scored per month (months with any): min "
              f"{com_nota.min() if len(com_nota) else 0}, median {com_nota.median() if len(com_nota) else 0:.0f}, "
              f"max {cobertura.max()}")
        if portao.empty:
            print("  no month kept: gate not passed")
            continue
        periodos = {f"{portao.index[0]:%Y-%m} to {portao.index[-1]:%Y-%m} (gate)": (portao.index[0], portao.index[-1]),
                    **{rotulo: tuple(map(pd.Timestamp, datas)) for rotulo, datas in HALVES.items()}}
        print(pd.DataFrame({rotulo: resumo_do_ic(portao.loc[de:ate]) for rotulo, (de, ate) in periodos.items()}).T
              .to_string(float_format=lambda x: f"{x:.3f}"))
        correlacoes = ic_por_mes(painel_da_nota(nota, dados.universo, meses, colunas), painel_composto).dropna()
        print(f"  rank correlation with the Composite score: mean {correlacoes.mean():.3f} over {len(correlacoes)} months")
        passou = avaliar_valor.PORTAO.aprova(portao)
        print(f"  gate ({avaliar_valor.PORTAO.nome}): {'PASSED' if passou else 'not passed'}")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
