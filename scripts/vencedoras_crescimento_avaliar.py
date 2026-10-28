"""
vencedoras_crescimento_avaliar.py - the one reading of attempt 50, extreme winners confirmed by profit growth
(docs/superpowers/specs/2026-10-28-vencedoras-crescimento-design.md, DECISOES 163). Runs only after the
pre-registration commit (`8236499`). No network: prices, universe, statements and CDI from frozen folders.

    python -X utf8 scripts/vencedoras_crescimento_avaliar.py > docs/vencedoras-crescimento-2026-10-28.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from core.vencedoras import CAIXA, estrategia_vencedoras, indice_do_caixa  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor  # noqa: E402
from scripts.avaliar_robustez import (FASES, _acoes, conferir_hashes, contar_tentativas, referencias,  # noqa: E402
                                      tabela_do_periodo)
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, PASTA_CVM, strategy_gate, years_beating_all  # noqa: E402

ARM = "Vencedoras-crescimento"
MACRO_FOLDER = FOLDER.parent / "2026-10-21-macro"        # the lake's daily CDI, frozen (DECISOES 111's folder)


def picks_profile(pesos_alvo: pd.DataFrame) -> str:
    """Stocks held per rebalance, rebalances with no pick and the mean CDI share, from the engine's target weights."""
    acoes = (pesos_alvo.drop(columns=CAIXA) > 1e-9).sum(axis=1)
    return (f"stocks per rebalance: min {acoes.min()}, median {acoes.median():.0f}, max {acoes.max()}; rebalances with "
            f"no pick {int((acoes == 0).sum())} of {len(acoes)}; mean CDI share {pesos_alvo[CAIXA].mean():.0%}")


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, sha256_cvm = avaliar_valor.carregar_balancos(PASTA_CVM)
    sha256_cdi = conferir_hashes(MACRO_FOLDER, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO_FOLDER / "cdi_daily.parquet")
    crescimento = nota_de_crescimento(balancos.fundamentos)
    n_tentativas = contar_tentativas()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_cvm | sha256_cdi}; arm {ARM}; DSR N {n_tentativas}")

    p_cons, p_bova, dsr = [], [], []
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(precos, estrategia, nome):
            return walk_forward_dinamico(precos, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora = rodar(acoes, estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                             CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))          # before the arm: fail fast
        com_caixa = acoes.assign(**{CAIXA: indice_do_caixa(cdi, acoes.index)})     # cash for this arm only
        braco = rodar(com_caixa, estrategia_vencedoras(mensal.retornos, crescimento, dados.setores), ARM)
        curvas = {ARM: braco, CONSERVADORA: conservadora,
                  **referencias(dados.precos.loc[:LAST_SESSION].iloc[fase:], CFG_BT)}
        valores = {nome: curva.valores for nome, curva in curvas.items()}
        check_aligned(valores)
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
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y}; refusals {len(braco.recusas)}; "
              f"failures {len(braco.falhas)} ===")
        print(picks_profile(braco.pesos_alvo))
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"Years beating BOVA11 and the Conservadora (reported, not a gate): {ganhos} of {anos} "
              f"({ganhos / anos:.0%})")
        print(_year_table({nome: valores[nome] for nome in (ARM, CONSERVADORA, "BOVA11")})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    passa = strategy_gate(p_cons, p_bova, dsr)
    print(f"\nStrategy gate: P[> Conservadora] {[round(p, 3) for p in p_cons]}, P[> BOVA11] "
          f"{[round(p, 3) for p in p_bova]}, DSR {[round(d, 3) for d in dsr]} (all >= {P_MIN}): "
          + ("CANDIDATE: pre-register a forward paper log" if passa else
             "not a candidate: the extreme-winners line is closed (no retuning)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
