"""
cda_best_ideas_avaliar.py - the one reading of attempt 52, BestIdeas-15: the Conservadora's portfolio rule ordered by
the CDA best-ideas score (docs/superpowers/specs/2026-10-29-cda-best-ideas-design.md, §5 and §7; trial row `6684c53`).
No network: prices, universe, dividends, CDI and the best-ideas panel frozen.

    python -X utf8 scripts/cda_best_ideas_avaliar.py > docs/cda-best-ideas-2026-10-29.txt
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.cdi_lago import daily_cdi  # noqa: E402
from core.composto import estrategia_por_nota  # noqa: E402
from core.metrics import cagr, drawdown  # noqa: E402
from core.risco import emissor  # noqa: E402
from core.robustez import psr_da_diferenca  # noqa: E402
from core.sinais import conservadora, estrategia_conservadora  # noqa: E402
from core.views import escore_normal  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, _acoes, conferir_hashes, contar_tentativas, referencias, tabela_do_periodo  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.cda_ic import FUNDOS_MINIMOS  # noqa: E402
from scripts.cda_painel import PAINEL, SAIDA  # noqa: E402
from scripts.composto_crescimento_avaliar import MACRO_FOLDER, phase_header  # noqa: E402
from scripts.qpo_avaliar import CONSERVADORA, _year_table, check_aligned, check_plumbing  # noqa: E402
from scripts.qpo_montar import FOLDER, LAST_SESSION  # noqa: E402
from scripts.qubo_universo_avaliar import P_MIN, strategy_gate, years_beating_all  # noqa: E402

ARM = "BestIdeas-15"
DESEMPATE = 1e-6        # the Conservadora's normal score only breaks ties of the best-ideas share

NotaDoMes = Callable[[pd.Timestamp, Iterable[str]], pd.Series]


def nota_com_desempate(painel: pd.DataFrame, nota_conservadora: NotaDoMes) -> NotaDoMes:
    """Best-ideas share of the security's issuer (0 when the month has no panel, fewer than ``FUNDOS_MINIMOS``
    qualifying funds, or the issuer is not in it) + ``DESEMPATE`` x the Conservadora's normal score; only names with a
    Conservadora score are candidates (the Conservadora's own set)."""
    por_mes = {mes: grupo.set_index("issuer")["score"] for mes, grupo in painel.groupby("month")
               if grupo["qualifying"].iloc[0] >= FUNDOS_MINIMOS}

    def _nota(mes: pd.Timestamp, candidatos: Iterable[str]) -> pd.Series:
        base = nota_conservadora(mes, candidatos).dropna()
        ideias = por_mes.get(mes, pd.Series(dtype=float))
        participacao = pd.Series([float(ideias.get(emissor(papel), 0.0)) for papel in base.index], index=base.index)
        return participacao + DESEMPATE * escore_normal(base)
    return _nota


def main() -> int:
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    sha256_painel = conferir_hashes(SAIDA, [PAINEL.name])
    sha256_cdi = conferir_hashes(MACRO_FOLDER, ["cdi_daily.parquet"])
    cdi = daily_cdi(ANOS_DE_CDI, LAST_SESSION.date(), path=MACRO_FOLDER / "cdi_daily.parquet")
    nota = nota_com_desempate(pd.read_parquet(PAINEL),
                              lambda mes, c: conservadora(mensal.retornos, mensal.proventos, mes, c))
    n_tentativas = contar_tentativas()
    print(f"Files (sha256): {dados.sha256 | mensal.sha256 | sha256_painel | sha256_cdi}; arm {ARM}; DSR N {n_tentativas}")

    p_cons, p_bova, dsr = [], [], []
    for fase in FASES:
        acoes = _acoes(dados.precos.loc[:LAST_SESSION].iloc[fase:])

        def rodar(estrategia, nome):
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        conservadora_bt = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, dados.setores),
                                CONSERVADORA)
        check_plumbing(fase, cagr(conservadora_bt.valores.dropna()))          # before the arm: fail fast
        braco = rodar(estrategia_por_nota(nota, dados.setores), ARM)
        curvas = {ARM: braco, CONSERVADORA: conservadora_bt,
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
        print("\n" + phase_header(fase, valores[ARM].index, braco, conservadora_bt))
        print(tabela.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"Years beating BOVA11 and the Conservadora (reported, not a gate): {ganhos} of {anos} "
              f"({ganhos / anos:.0%})")
        print(_year_table({nome: valores[nome] for nome in (ARM, CONSERVADORA, "BOVA11")})
              .to_string(float_format=lambda x: f"{x:.1%}"))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)

    passa = strategy_gate(p_cons, p_bova, dsr)
    print(f"\nStrategy gate: P[> Conservadora] {[round(p, 3) for p in p_cons]}, P[> BOVA11] "
          f"{[round(p, 3) for p in p_bova]}, DSR {[round(d, 3) for d in dsr]} (all >= {P_MIN}): "
          + ("CANDIDATE: pre-register a forward paper log against the same-day Conservadora" if passa else
             "not a candidate: the best-ideas line is closed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
