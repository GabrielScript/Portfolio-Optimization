"""The pre-registered grid, run in one go and resumable: every row already in ``resultados.csv`` is skipped, so a run
that dies halfway is restarted by the same command and never repeats a configuration.

The grid is fixed here, in code, before any test number is read (DECISOES 330):
  windows  2016, 2019, 2022, 2025    one per era, plus the earliest window with a usable training tape
  agents   P1 = ppo/dsr, P2 = ppo|a2c|ddpg/valor, P3 = ppo|a2c|ddpg/dsr_drawdown
  seeds    0 (screening; the surviving cells get seeds 1-2 later)
  costs    0 bps (fidelity to the papers) and 10 bps (the decision number)
  names    30 (the papers' scale)"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import pandas as pd

from drl import dados, executar, protocolo

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "drl" / "saidas" / "resultados.csv"

JANELAS = (2016, 2019, 2022, 2025)
AGENTES = (("ppo", "dsr"), ("ppo", "valor"), ("a2c", "valor"), ("ddpg", "valor"),
           ("ppo", "dsr_drawdown"), ("a2c", "dsr_drawdown"), ("ddpg", "dsr_drawdown"))
SEMENTES = (0,)          # screening pass: one seed per cell, the survivors get the seed fan-out
CUSTOS = (0.0, 10.0)
NOMES = 30
TIMESTEPS = 150_000


def grade(janelas=JANELAS, agentes=AGENTES, sementes=SEMENTES, custos=CUSTOS, nomes=NOMES,
          timesteps=TIMESTEPS, arquivo: Path = ARQUIVO) -> list[dict]:
    """Every configuration of the grid that is not already in ``arquivo``."""
    feitos = set()
    if arquivo.exists():
        anterior = pd.read_csv(arquivo)
        feitos = {(r.algoritmo, r.recompensa, float(r.custo_bps), int(r.ano_teste), int(r.semente))
                  for r in anterior.itertuples()}
    pendentes = [c for c in itertools.product(agentes, sementes, custos, janelas)
                 if (c[0][0], c[0][1], c[2], c[3], c[1]) not in feitos]
    return pendentes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=TIMESTEPS)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--limite", type=int, default=0, help="stop after this many runs (0 = the whole grid)")
    args = p.parse_args(argv)

    fechamentos = dados.painel()
    elegiveis = dados.elegiveis(fechamentos)
    janelas = {j.ano_teste: j for j in protocolo.janelas(fechamentos.index)}
    faltando = grade(timesteps=args.timesteps)
    if args.limite:
        faltando = faltando[: args.limite]
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    for i, ((algoritmo, recompensa), semente, custo, ano) in enumerate(faltando, 1):
        janela = janelas[ano]
        nomes = executar.nomes_elegiveis(elegiveis, janela.teste.start, NOMES)
        try:
            linha = executar.corrida(fechamentos, elegiveis, janela, nomes, algoritmo=algoritmo,
                                     recompensa=recompensa, custo_bps=custo, timesteps=args.timesteps,
                                     seed=semente, n_envs=args.n_envs)
        except Exception as erro:                     # noqa: BLE001 - one broken config must not stop the grid
            print(f"[{i}/{len(faltando)}] {algoritmo}/{recompensa}/{ano}/s{semente}/c{custo:.0f} FALHOU: {erro}",
                  flush=True)
            continue
        pd.DataFrame([linha]).to_csv(ARQUIVO, mode="a", header=not ARQUIVO.exists(), index=False)
        decorrido = time.time() - inicio
        print(f"[{i}/{len(faltando)}] {algoritmo}/{recompensa}/{custo:.0f}bps/{ano}/s{semente} "
              f"ret={linha['retorno_anual']:.1%} sharpe={linha['sharpe']:.2f} mdd={linha['mdd']:.1%} "
              f"giro={linha['giro_anual']:.1f} ({decorrido / i:.0f}s/run)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
