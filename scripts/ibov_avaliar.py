"""
The one reading of the Ibovespa rebalance-flow test (spec ``docs/superpowers/specs/2026-11-08-ibov-recomposicao-design.md``
and its amendment 1): R1 buys the entrants at the 3rd preview's close, R2 at the 1st preview's, both sell at the close
of the old portfolio's last session; net of the app's costs, against BOVA11, clustered by rebalance date.

Usage: python -X utf8 scripts/ibov_avaliar.py   (writes docs/ibov-recomposicao-2026-11-08.txt)
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core import recomposicao as rc  # noqa: E402
from core.execucao import Custos, custo_da_ordem  # noqa: E402

LAGO = Path("D:/b3-datalake/exports")
DADOS = RAIZ / "docs" / "dados" / "ibov-recomposicao"
SAIDA = RAIZ / "docs" / "ibov-recomposicao-2026-11-08.txt"
CUSTOS = Custos(emolumentos=0.0003, impacto=1.0)          # the simulator's and the real plan's costs
CONTA = 10_000.0                                          # the fictitious account the rule would run in
SESSOES_MINIMAS = 252                                     # amendment 1: CVM entrants that traded a year before


def _tuplas(coluna: pd.Series) -> pd.Series:
    return coluna.fillna("()").map(lambda texto: tuple(ast.literal_eval(texto)))


def eventos_das_regras(pregoes: pd.DatetimeIndex, sessoes_ate: dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    """One row per (rule, rebalance, security) with its entry and exit sessions and the source."""
    previas = pd.read_csv(DADOS / "previas.csv", parse_dates=["rebalanceamento", "divulgacao"])
    previas["entradas"] = _tuplas(previas["entradas"])
    cvm = pd.read_csv(DADOS / "eventos.csv", parse_dates=["rebalanceamento"])
    cvm = cvm[cvm["tipo"] == "entrada"]
    linhas = []
    rebalanceamentos = sorted(set(cvm["rebalanceamento"]) | set(previas["rebalanceamento"]))
    for mes in rebalanceamentos:
        if mes < pregoes[0] or mes.replace(day=1) > pregoes[-1]:
            continue
        saida = rc.ultimo_pregao_antigo(mes, pregoes)
        terceira = previas[(previas["rebalanceamento"] == mes) & (previas["previa"] == 3)]
        if len(terceira):
            entrada = rc.data_de_entrada(rc.data_da_terceira_previa(mes, pregoes), terceira["divulgacao"].iloc[0],
                                         saida, pregoes)
            nomes, fonte = terceira["entradas"].iloc[0], "3a previa B3"
        else:
            entrada = rc.data_da_terceira_previa(mes, pregoes)
            nomes = tuple(papel for papel in cvm.loc[cvm["rebalanceamento"] == mes, "papel"]
                          if (sessoes_ate.get(papel, pd.DatetimeIndex([])) < entrada).sum() >= SESSOES_MINIMAS)
            fonte = "CVM, 252 sessoes"
        linhas += [{"regra": "R1", "rebalanceamento": mes, "papel": papel, "entrada": entrada, "saida": saida,
                    "fonte": fonte} for papel in nomes]
        primeira = previas[(previas["rebalanceamento"] == mes) & (previas["previa"] == 1)]
        if len(primeira):
            entrada = rc.data_de_entrada(rc.data_da_primeira_previa(mes, pregoes), primeira["divulgacao"].iloc[0],
                                         saida, pregoes)
            linhas += [{"regra": "R2", "rebalanceamento": mes, "papel": papel, "entrada": entrada, "saida": saida,
                        "fonte": "1a previa B3"} for papel in primeira["entradas"].iloc[0]]
    return pd.DataFrame(linhas)


def main() -> int:
    painel = pd.read_parquet(LAGO / "daily_liquid_panel.parquet", columns=["date", "security", "tr_close"])
    painel["date"] = pd.to_datetime(painel["date"])
    bova = painel[painel["security"] == "BOVA11"].set_index("date")["tr_close"]
    pregoes = pd.DatetimeIndex(sorted(bova.index.unique()))
    sessoes_ate = {papel: pd.DatetimeIndex(grupo["date"]) for papel, grupo in painel.groupby("security")}
    eventos = eventos_das_regras(pregoes, sessoes_ate)
    precos = {papel: grupo.set_index("date")["tr_close"] for papel, grupo in
              painel[painel["security"].isin(set(eventos["papel"]))].groupby("security")}
    metricas = pd.read_parquet(LAGO / "monthly_metrics.parquet",
                               columns=["month", "security", "quoted_spread", "vol_daily", "median_dtv_brl"])
    metricas["month"] = pd.to_datetime(metricas["month"])
    resultados, descartes = [], []
    for linha in eventos.itertuples():
        if linha.entrada >= linha.saida:
            descartes.append((linha.regra, linha.rebalanceamento.date(), linha.papel, "entrada no ultimo pregao"))
            continue
        mes_antes = (linha.entrada - pd.offsets.MonthEnd(1)).normalize()
        liquidez = metricas[(metricas["security"] == linha.papel) & (metricas["month"] <= mes_antes)].tail(1)
        n = int(((eventos["regra"] == linha.regra) & (eventos["rebalanceamento"] == linha.rebalanceamento)).sum())
        ordem = CONTA / n
        if liquidez.empty or not liquidez["median_dtv_brl"].iloc[0] > 0:
            descartes.append((linha.regra, linha.rebalanceamento.date(), linha.papel, "sem liquidez no mes anterior"))
            continue
        custo = custo_da_ordem(ordem, float(liquidez["quoted_spread"].iloc[0]) / 2, float(liquidez["vol_daily"].iloc[0]),
                               float(liquidez["median_dtv_brl"].iloc[0]), CUSTOS) / ordem
        evento = rc.resultado_do_evento(precos.get(linha.papel, pd.Series(dtype=float)), bova, linha.entrada,
                                        linha.saida, custo)
        if evento is None:
            descartes.append((linha.regra, linha.rebalanceamento.date(), linha.papel, "sem preco"))
            continue
        resultados.append({"regra": linha.regra, "rebalanceamento": linha.rebalanceamento, "papel": linha.papel,
                           "entrada": linha.entrada.date(), "saida": linha.saida.date(), "fonte": linha.fonte,
                           "custo_lado": custo, **evento})
    tabela = pd.DataFrame(resultados)
    tabela.to_csv(DADOS / "resultado_eventos.csv", index=False, lineterminator="\n", date_format="%Y-%m-%d")
    linhas = ["Ibovespa rebalance flows: the one reading (2026-11-08). Spec 2026-11-08 + amendment 1.", ""]
    leituras = {"R1 (3rd preview, 2009-2026)": (tabela[tabela["regra"] == "R1"], True),
                "R1 on 2020+ only (no look-ahead years)": (tabela[(tabela["regra"] == "R1")
                                                             & (tabela["rebalanceamento"] >= "2020-01-01")], False),
                "R2 (1st preview, 2019-2026)": (tabela[tabela["regra"] == "R2"], False)}
    for nome, (parte, blocos) in leituras.items():
        resumo = rc.resumo(parte, exige_blocos=blocos)
        linhas += [f"== {nome} ==", json.dumps({chave: (round(valor, 5) if isinstance(valor, float) else valor)
                                                 for chave, valor in resumo.items() if chave != "blocos"}),
                   "blocks: " + json.dumps({k: round(v, 5) for k, v in resumo["blocos"].items()}), ""]
    linhas += ["mean cost per side: " + json.dumps(tabela.groupby("regra")["custo_lado"].mean().round(5).to_dict()),
               "mean holding sessions: " + json.dumps(tabela.assign(
                   s=[int(((pregoes > pd.Timestamp(e)) & (pregoes <= pd.Timestamp(s))).sum())
                      for e, s in zip(tabela["entrada"], tabela["saida"])]).groupby("regra")["s"].mean().round(1)
                   .to_dict()),
               f"dropped events: {len(descartes)}"] + [f"  {d}" for d in descartes]
    SAIDA.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
