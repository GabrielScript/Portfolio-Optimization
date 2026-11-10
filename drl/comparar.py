"""The classical comparators on the same window, same names and same cost as the agents: equal weights, minimum
variance and the Sharpe MVO, rebalanced on every session, plus the equal-weight buy-and-hold. Their rows go to the same
``resultados.csv`` so one filter produces the whole table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from drl import baselines, dados, protocolo, relatorio

RAIZ = Path(__file__).resolve().parent.parent
SAIDAS = RAIZ / "drl" / "saidas"


def pesos_por_sessao(fechamentos: pd.DataFrame, inicio: int, fim: int, regra: str) -> list[np.ndarray]:
    """The weight path ``rule`` produces over the sessions ``inicio..fim``: recomputed from the window ending at the
    session, applied to the next session's return (the same timing the agents get)."""
    saida: list[np.ndarray] = []
    for t in range(inicio, fim):
        ret = baselines._retornos(fechamentos, t)
        if regra == "iguais":
            saida.append(baselines.pesos_iguais(fechamentos.shape[1]))
        elif regra == "minvariancia":
            saida.append(baselines.pesos_min_variancia(ret))
        else:
            saida.append(baselines.pesos_sharpe(ret))
    return saida


def comparar(fechamentos: pd.DataFrame, janela: protocolo.Janela, nomes: list[str], regra: str,
             custo_bps: float = 0.0) -> dict:
    painel = fechamentos[nomes]
    inicio = max(janela.teste.start, 253)
    pesos = pesos_por_sessao(painel, inicio, janela.teste.stop - 1, regra)
    retornos = painel.to_numpy(dtype=float)[inicio + 1: janela.teste.stop]
    retornos = retornos / painel.to_numpy(dtype=float)[inicio: janela.teste.stop - 1] - 1.0
    retornos = np.nan_to_num(retornos)
    giros = [float(np.abs(w - (pesos[i - 1] if i else np.zeros_like(w))).sum()) for i, w in enumerate(pesos)]
    curva = baselines.curva(pesos, retornos, custo_bps=custo_bps)
    linha = relatorio.metricas(curva, giros=np.array(giros))
    return {"algoritmo": regra, "recompensa": "-", "custo_bps": custo_bps, "ano_teste": janela.ano_teste,
            "semente": 0, "nomes": len(nomes), "modo": "sessao", "timesteps": 0, **linha}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nomes", type=int, default=30)
    p.add_argument("--ano-teste", type=int, default=2026)
    p.add_argument("--custo-bps", type=float, default=0.0)
    args = p.parse_args(argv)
    fechamentos = dados.painel()
    elegiveis = dados.elegiveis(fechamentos)
    janelas = protocolo.janelas(fechamentos.index, primeiro_teste=args.ano_teste, ultimo_teste=args.ano_teste)
    if not janelas:
        raise SystemExit(f"no complete window for {args.ano_teste}")
    janela = janelas[0]
    disponiveis = elegiveis.iloc[janela.teste.start]
    nomes = sorted(disponiveis[disponiveis].index)[:args.nomes]
    linhas = [comparar(fechamentos, janela, nomes, regra, args.custo_bps)
              for regra in ("iguais", "minvariancia", "mvo")]
    SAIDAS.mkdir(parents=True, exist_ok=True)
    arquivo = SAIDAS / "resultados.csv"
    pd.DataFrame(linhas).to_csv(arquivo, mode="a", header=not arquivo.exists(), index=False)
    for linha in linhas:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in linha.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
