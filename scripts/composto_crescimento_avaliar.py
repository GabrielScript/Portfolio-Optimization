"""
composto_crescimento_avaliar.py - the one reading of attempt 51, the Conservadora's rule ordered by the Conservadora +
value + profit growth composite (docs/superpowers/specs/2026-10-28-composto-crescimento-design.md, DECISOES 164).
Runs only after the pre-registration commit (`22e7f14`). No network: prices, universe, statements and CDI frozen.

    python -X utf8 scripts/composto_crescimento_avaliar.py > docs/composto-crescimento-2026-10-28.txt

Also writes the three phases' daily curves (arm, Conservadora, BOVA11) to docs/dados/composto-curvas-2026-10-28.parquet
and its hash to docs/dados/SHA256SUMS: the Monte Carlo bench's input.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import (FASES, _acoes, conferir_hashes, contar_tentativas, referencias,  # noqa: E402
                                      tabela_do_periodo)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, _value_score, strategy_gate, years_beating_all  # noqa: E402

ARM = "Composto-crescimento"
MACRO_FOLDER = FOLDER.parent / "2026-10-21-macro"
CURVES_FILE = FOLDER.parent / "composto-curvas-2026-10-28.parquet"


def curves_frame(capturadas: Sequence[tuple[int, Mapping[str, pd.Series]]]) -> pd.DataFrame:
    """One row per (phase, session): the arm's, the Conservadora's and BOVA11's daily values."""
    partes = [pd.DataFrame({"fase": fase, "date": curvas[ARM].index, "Composto": curvas[ARM].to_numpy(),
                            "Conservadora": curvas[CONSERVADORA].reindex(curvas[ARM].index).to_numpy(),
                            "BOVA11": curvas["BOVA11"].reindex(curvas[ARM].index).to_numpy()})
              for fase, curvas in capturadas]
    return pd.concat(partes, ignore_index=True)


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    nota, sha256_cvm = _value_score(mensal, com_crescimento=True)          # attempt 49's score, same code
    sha256_cdi = conferir_hashes(MACRO_FOLDER, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO_FOLDER / "cdi_daily.parquet")
    n_tentativas = contar_tentativas()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; arm {ARM}; DSR N {n_tentativas}")

    p_cons, p_bova, dsr, capturadas = [], [], [], []
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome):
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))          # before the arm: fail fast
        braco = rodar(estrategia_por_nota(nota, dados.setores), ARM)
        curvas = {ARM: braco, CONSERVADORA: conservadora,
                  **referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT)}
        valores = {nome: curva.valores for nome, curva in curvas.items()}
        check_aligned(valores)
        capturadas.append((fase, valores))
        rf = market_data.serie_rf_diaria(valores[ARM].index, cdi, (1.0 + float(cdi.iloc[-1])) ** 252 - 1.0)
        tabela = tabela_do_periodo({nome: valores[nome] for nome in (ARM, CONSERVADORA)}, valores["BOVA11"], rf,
                                   n_tentativas)
        for nome in (ARM, CONSERVADORA, "BOVA11"):
            tabela.loc[nome, "CAGR"] = cagr(valores[nome].dropna())
            tabela.loc[nome, "MDD"] = drawdown(valores[nome].dropna())[1]
        tabela.loc[ARM, "P[> Conservadora]"] = psr_da_diferenca(valores[ARM], valores[CONSERVADORA])
        p_cons.append(float(tabela.loc[ARM, "P[> Conservadora]"]))
        p_bova.append(float(tabela.loc[ARM, "P aritmético"]))
        dsr.append(float(tabela.loc[ARM, "DSR"]))
        ganhos, anos = years_beating_all(valores[ARM], [valores["BOVA11"], valores[CONSERVADORA]])
        datas = valores[ARM].index
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y}; refusals {len(braco.recusas)} "
              f"(Conservadora {len(conservadora.recusas)}); annual turnover {braco.turnover_anual_medio():.2f} "
              f"(Conservadora {conservadora.turnover_anual_medio():.2f}) ===")
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"Years beating BOVA11 and the Conservadora (reported, not a gate): {ganhos} of {anos} "
              f"({ganhos / anos:.0%})")
        print(_year_table({nome: valores[nome] for nome in (ARM, CONSERVADORA, "BOVA11")})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    curves_frame(capturadas).to_parquet(CURVES_FILE, index=False)
    hash_curvas = hashlib.sha256(CURVES_FILE.read_bytes()).hexdigest()
    with (CURVES_FILE.parent / "SHA256SUMS").open("a", encoding="ascii", newline="\n") as somas:
        somas.write(f"{hash_curvas} *{CURVES_FILE.name}\n")
    print(f"\nCurves saved: {CURVES_FILE.name} sha256 {hash_curvas}")

    passa = strategy_gate(p_cons, p_bova, dsr)
    print(f"Strategy gate: P[> Conservadora] {[round(p, 3) for p in p_cons]}, P[> BOVA11] "
          f"{[round(p, 3) for p in p_bova]}, DSR {[round(d, 3) for d in dsr]} (all >= {P_MIN}): "
          + ("CANDIDATE: pre-register a forward paper log; the composite enters the Monte Carlo bench" if passa else
             "not a candidate: the composite line is closed; the bench uses the Conservadora, BOVA11 and the CDI"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
