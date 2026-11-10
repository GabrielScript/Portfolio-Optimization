"""The reading: one table per era, zero-cost against costed, agents against the classical baselines. Nothing here
averages across a protocol change — each row is one (agent, reward, cost, window, seed) cell as it was run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "drl" / "saidas" / "resultados.csv"
COLUNAS = ["algoritmo", "recompensa", "custo_bps", "ano_teste", "semente", "retorno_anual", "vol_anual", "sharpe",
           "mdd", "calmar", "giro_anual", "total"]


def tabela(arquivo: Path = ARQUIVO) -> pd.DataFrame:
    """The runs, longest first, with a column saying how many names the run traded."""
    d = pd.read_csv(arquivo)
    return d[[c for c in COLUNAS + ["nomes"] if c in d.columns]].sort_values(["ano_teste", "custo_bps", "sharpe"],
                                                                             ascending=[True, True, False])


def comparacao(d: pd.DataFrame, ano: int, custo: float) -> pd.DataFrame:
    """One window's table: every run of that window at that cost, the baselines included, ranked by Sharpe."""
    recorte = d[(d["ano_teste"] == ano) & (d["custo_bps"] == custo)]
    return recorte.sort_values("sharpe", ascending=False)


def resumo(d: pd.DataFrame) -> pd.DataFrame:
    """The agent cells that beat equal weight on the same window and cost: the only ones worth a second seed."""
    linhas = []
    for (ano, custo), grupo in d.groupby(["ano_teste", "custo_bps"]):
        base = grupo[grupo["algoritmo"] == "iguais"]
        if base.empty:
            continue
        melhor = base["sharpe"].max()
        for r in grupo[(grupo["algoritmo"] != "iguais") & (grupo["sharpe"] > melhor)].itertuples():
            linhas.append({"ano_teste": ano, "custo_bps": custo, "agente": f"{r.algoritmo}/{r.recompensa}",
                           "sharpe": r.sharpe, "sharpe_iguais": melhor, "mdd": r.mdd, "giro": r.giro_anual})
    return pd.DataFrame(linhas)


def main() -> int:
    d = tabela()
    for custo in sorted(d["custo_bps"].unique()):
        print(f"\n=== custo {custo:.0f} bps ===")
        for ano in sorted(d["ano_teste"].unique()):
            print(f"\n-- {ano} --")
            print(comparacao(d, ano, custo).to_string(index=False))
    vencedores = resumo(d)
    print("\n=== cells that beat equal weight ===")
    print(vencedores.to_string(index=False) if not vencedores.empty else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
