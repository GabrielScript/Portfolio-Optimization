"""
qpo_anos.py - the QPO year by year against BOVA11, IVVB11 (S&P 500 in reais) and the B3's crypto ETFs (user's
request, session 54). Descriptive only: no gate, no decision, nothing tuned. The pre-registered reading is
docs/qpo-2026-10-27.txt (QPO not adopted, DECISOES 147); this table does not change it.

Arms as in that reading (scripts/qpo_avaliar.py: same frozen folder, engine, dates and plumbing check): QPO-exact,
QPO-QAOA (the recorded Kaggle picks at the p the reading chose) and the Conservadora; BOVA11 and IVVB11 bought and
held in the same engine. The crypto ETFs (HASH11, BITH11, ETHE11, QBTC11, DECISOES 140) are not in the frozen folder:
their total-return closes come from the lake's gold (read on the run date, not frozen), bought and held from their
first close, gross of costs, cut at the same last session. Each column's first year is partial (from its first
value); 2026 is year to date (16/10/2026).

    python scripts/qpo_anos.py docs/qpo/results [GOLD] > docs/qpo-anos-2026-10-27.txt
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import market_data  # noqa: E402
from core.backtest import Congelamento, walk_forward_dinamico  # noqa: E402
from core.config import taxa_anual_para_aritmetica  # noqa: E402
from core.metrics import cagr  # noqa: E402
from core.qpo import CFG_MAX_SHARPE, K, exact_picks, qubo_selector, recorded_selector  # noqa: E402
from core.qpo import selection_then_max_sharpe  # noqa: E402
from core.sinais import estrategia_conservadora  # noqa: E402
from scripts import avaliar_conservadora, avaliar_robustez  # noqa: E402
from scripts.avaliar_robustez import FASES, REFERENCIAS, _acoes, referencias  # noqa: E402
from scripts.avaliar_universo_dinamico import ANOS_DE_CDI, CFG_BT  # noqa: E402
from scripts.qpo_avaliar import (CONSERVADORA, _results, _year_table, check_aligned, check_plumbing,  # noqa: E402
                                 choose_p, load_qaoa_picks)
from scripts.qpo_montar import FOLDER, LAST_SESSION, RUN_DATE, SAIDA  # noqa: E402

GOLD = Path("D:/b3-datalake/gold/equity_adjusted.parquet")
ETFS_DE_CRIPTO = ("HASH11", "BITH11", "ETHE11", "QBTC11")   # as ui/servicos.py (DECISOES 140)
QPO_ARMS = ("QPO-exact", "QPO-QAOA")
N_MAX = 20                                                  # the reading's (spike, DECISOES 145)


def gold_closes(arquivo: Path, tickers: Sequence[str], last: pd.Timestamp) -> pd.DataFrame:
    """Total-return closes (date x ticker, in ``tickers`` order) up to ``last``. A ``return_break`` row in that
    window (a print the lake does not trust) or a ticker the file lacks stops the run."""
    tabela = pd.read_parquet(arquivo, columns=["date", "ticker", "tr_close", "return_break"],
                             filters=[("ticker", "in", list(tickers))])
    tabela["date"] = pd.to_datetime(tabela["date"])
    tabela = tabela[tabela["date"] <= last]
    faltam = sorted(set(tickers) - set(tabela["ticker"]))
    if faltam:
        raise ValueError(f"{arquivo}: no rows for {faltam} up to {last:%d/%m/%Y}")
    quebras = tabela[tabela["return_break"] == 1]
    if not quebras.empty:
        raise ValueError(f"{arquivo}: return_break rows {quebras[['date', 'ticker']].to_dict('records')}")
    return tabela.pivot(index="date", columns="ticker", values="tr_close")[list(tickers)]


def wins(tabela: pd.DataFrame, arm: str, refs: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Per reference: (years the arm's return beat it, years both have a return)."""
    placar = {}
    for ref in refs:
        ambos = tabela[[arm, ref]].dropna()
        placar[ref] = (int((ambos[arm] > ambos[ref]).sum()), len(ambos))
    return placar


def main(argv: list[str]) -> int:
    results_dir = Path(argv[0])
    gold = Path(argv[1]) if len(argv) > 1 else GOLD
    inicio = time.perf_counter()
    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    precos = dados.precos.loc[:LAST_SESSION]
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    selic = market_data.obter_selic_meta(RUN_DATE)
    cdi = market_data.baixar_cdi_diario(ANOS_DE_CDI, hoje=RUN_DATE)
    cfg = replace(CFG_MAX_SHARPE, taxa_livre_risco=taxa_anual_para_aritmetica(selic.valor))
    setores = dados.setores
    p = choose_p({p: [r["ratio"] for r in _results(results_dir, 0, p).values()] for p in (1, 2, 3)})
    cripto = _year_table(dict(gold_closes(gold, ETFS_DE_CRIPTO, LAST_SESSION).items()))
    print(f"Files (sha256): {dados.sha256 | mensal.sha256}; crypto ETFs from {gold} (read {date.today():%d/%m/%Y}, "
          f"not frozen, gross); p {p}. Descriptive: no gate, no decision (DECISOES 147 stands).")

    for fase in FASES:
        acoes = _acoes(precos.iloc[fase:])

        def rodar(estrategia, nome):
            return walk_forward_dinamico(acoes, dados.universo, estrategia, CFG_BT, nome, Congelamento())

        def com_max_sharpe(seletor):
            return selection_then_max_sharpe(seletor, mensal.retornos, mensal.proventos, setores, cfg, cdi)

        conservadora = rodar(estrategia_conservadora(mensal.retornos, mensal.proventos, setores), CONSERVADORA)
        check_plumbing(fase, cagr(conservadora.valores.dropna()))
        curvas = {
            "QPO-exact": rodar(com_max_sharpe(qubo_selector(setores, N_MAX, K, exact_picks)), "QPO-exact"),
            "QPO-QAOA": rodar(com_max_sharpe(recorded_selector(load_qaoa_picks(results_dir, SAIDA, fase, p))),
                              "QPO-QAOA"),
            CONSERVADORA: conservadora,
            **referencias(precos.iloc[fase:], CFG_BT),
        }
        valores = {nome: curva.valores for nome, curva in curvas.items()}
        check_aligned(valores)
        tabela = _year_table(valores).rename(columns={CONSERVADORA: "Conservadora"}).join(cripto)
        datas = valores["QPO-exact"].index
        print(f"\n=== Phase {fase}: {datas[0]:%d/%m/%Y} to {datas[-1]:%d/%m/%Y} (2026 = year to date) ===")
        print(tabela.to_string(float_format=lambda x: f"{x:.1%}"))
        refs = ("Conservadora", *REFERENCIAS, *ETFS_DE_CRIPTO)
        for arm in QPO_ARMS:
            placar = wins(tabela, arm, refs)
            print(f"{arm} beat: " + ", ".join(f"{ref} {a}/{b}" for ref, (a, b) in placar.items()))
        print(f"(phase {fase} done in {time.perf_counter() - inicio:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
