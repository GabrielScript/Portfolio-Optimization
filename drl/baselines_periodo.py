"""The three classical rules over the whole out-of-sample period, stitched into one continuous curve so they can be
read against the Composite on the same ruler. Run with `.venv-drl/Scripts/python.exe -m drl.baselines_periodo`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from drl import baselines, comparar, dados, executar, periodo, protocolo, relatorio

SAIDA = Path(__file__).resolve().parent / "saidas"


def curva_da_regra(fechamentos, elegiveis, janelas, regra: str, custo_bps: float = 10.0,
                   quantos: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """The continuous curve of ``regra`` over the period and its turnover per session.

    ``quantos=0`` trades the whole eligible universe of each test year instead of the first ``quantos`` names, which is
    the comparison the Composite demands: same universe, same cost."""
    curvas, giros = [], []
    for janela in janelas:
        if quantos:
            nomes = executar.nomes_elegiveis(elegiveis, janela.teste.start, quantos)
        else:
            nomes = list(elegiveis.columns[elegiveis.iloc[janela.teste.start].to_numpy(dtype=bool)])
        painel = fechamentos[nomes]
        inicio = max(janela.teste.start, 253)
        pesos = comparar.pesos_por_sessao(painel, inicio, janela.teste.stop - 1, regra)
        m = painel.to_numpy(dtype=float)
        rets = np.nan_to_num(m[inicio + 1: janela.teste.stop] / m[inicio: janela.teste.stop - 1] - 1.0)
        giros.append(np.array([float(np.abs(w - (pesos[i - 1] if i else np.zeros_like(w))).sum())
                               for i, w in enumerate(pesos)]))
        curvas.append(baselines.curva(pesos, rets, custo_bps=custo_bps))
    return periodo.emendar(curvas), np.concatenate(giros)


def main(primeiro: int = 2010, ultimo: int = 2026, custo_bps: float = 10.0, quantos: int = 30,
         regras: tuple[str, ...] = ("iguais", "minvariancia", "mvo")) -> int:
    fechamentos = dados.painel()
    elegiveis = dados.elegiveis(fechamentos)
    janelas = protocolo.janelas(fechamentos.index, primeiro, ultimo)
    tabela = {}
    for regra in regras:
        curva, giros = curva_da_regra(fechamentos, elegiveis, janelas, regra, custo_bps, quantos)
        tabela[regra] = relatorio.metricas(curva, giros=giros)
        print(regra, {k: round(v, 4) for k, v in tabela[regra].items()}, flush=True)
    SAIDA.mkdir(parents=True, exist_ok=True)
    nome = "baseline_periodo.json" if quantos else "baseline_universo.json"
    (SAIDA / nome).write_text(json.dumps(tabela, indent=1, default=float))
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primeiro", type=int, default=2010)
    p.add_argument("--ultimo", type=int, default=2026)
    p.add_argument("--custo-bps", type=float, default=10.0)
    p.add_argument("--quantos", type=int, default=30, help="0 = the whole eligible universe of each year")
    p.add_argument("--regras", default="iguais,minvariancia,mvo")
    args = p.parse_args()
    raise SystemExit(main(args.primeiro, args.ultimo, args.custo_bps, args.quantos,
                          tuple(r.strip() for r in args.regras.split(",") if r.strip())))
