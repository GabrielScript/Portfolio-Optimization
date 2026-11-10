"""One run of the grid: load the lake's panel, build a walk-forward window, train one agent and write its test-year
result to ``drl/saidas/resultados.csv``. Every row carries the configuration, the window, the seed and whether the run
was costed, so the reading can filter by any of them and nothing is ever averaged across a change of protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from drl import agentes, ambiente, dados, protocolo, relatorio

RAIZ = Path(__file__).resolve().parent.parent
SAIDAS = RAIZ / "drl" / "saidas"


def nomes_elegiveis(elegiveis: pd.DataFrame, linha: int, quantos: int) -> list[str]:
    """The ``quantos`` tradeable names at ``linha``, in alphabetical order so the action vector is reproducible. The
    papers trade a fixed small universe (DJIA-30); alphabetical rather than by liquidity because the lake carries no
    traded value per session, and ranking by a proxy we do not have would be a choice we cannot defend."""
    disponiveis = elegiveis.iloc[linha]
    return sorted(disponiveis[disponiveis].index)[:quantos]


def corrida(fechamentos: pd.DataFrame, elegiveis: pd.DataFrame, janela: protocolo.Janela, nomes: list[str], *,
            algoritmo: str, recompensa: str, custo_bps: float, timesteps: int, seed: int, n_envs: int = 1,
            modo: str = "matriz", lookback: int = 60, rebalancear_a_cada: int = 1) -> dict:
    """Train on the window's training range and evaluate on its untouched test year."""
    treino_fim = janela.treino.stop
    env_treino = ambiente.AmbienteAcoes(fechamentos[nomes], elegiveis[nomes], max(janela.treino.start, lookback + 1),
                                        treino_fim + 1, recompensa=recompensa, custo_bps=custo_bps,
                                        lookback=lookback, modo=modo, rebalancear_a_cada=rebalancear_a_cada)
    inicio_teste = max(janela.teste.start, treino_fim + 1)
    env_teste = ambiente.AmbienteAcoes(fechamentos[nomes], elegiveis[nomes], inicio_teste, janela.teste.stop,
                                       recompensa=recompensa, custo_bps=custo_bps, lookback=lookback,
                                       modo=modo, rebalancear_a_cada=rebalancear_a_cada)
    modelo, venv = agentes.treinar(algoritmo, env_treino, total_timesteps=timesteps, seed=seed, n_envs=n_envs)
    try:
        resultado = agentes.avaliar(modelo, env_teste)
    finally:
        venv.close()
    linha = relatorio.metricas(resultado["curva"], giros=resultado["giros"])
    return {"algoritmo": algoritmo, "recompensa": recompensa, "custo_bps": custo_bps, "ano_teste": janela.ano_teste,
            "semente": seed, "nomes": len(nomes), "modo": modo, "timesteps": timesteps, **linha}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algoritmo", default="ppo", choices=agentes.ALGORITMOS)
    p.add_argument("--recompensa", default="dsr", choices=ambiente.RECOMPENSAS)
    p.add_argument("--custo-bps", type=float, default=0.0)
    p.add_argument("--nomes", type=int, default=30)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--ano-teste", type=int, default=2026)
    p.add_argument("--modo", default="matriz", choices=("matriz", "compacto"))
    p.add_argument("--rebalancear-a-cada", type=int, default=1)
    p.add_argument("--saida", default="resultados.csv")
    args = p.parse_args(argv)

    fechamentos = dados.painel()
    elegiveis = dados.elegiveis(fechamentos)
    janelas = protocolo.janelas(fechamentos.index, ultimo_teste=args.ano_teste)
    janela = next((j for j in janelas if j.ano_teste == args.ano_teste), None)
    if janela is None:
        raise SystemExit(f"no complete window for {args.ano_teste}")
    nomes = nomes_elegiveis(elegiveis, janela.teste.start, args.nomes)
    linha = corrida(fechamentos, elegiveis, janela, nomes, algoritmo=args.algoritmo, recompensa=args.recompensa,
                    custo_bps=args.custo_bps, timesteps=args.timesteps, seed=args.semente, n_envs=args.n_envs, modo=args.modo,
                    rebalancear_a_cada=args.rebalancear_a_cada)
    SAIDAS.mkdir(parents=True, exist_ok=True)
    arquivo = SAIDAS / args.saida
    pd.DataFrame([linha]).to_csv(arquivo, mode="a", header=not arquivo.exists(), index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in linha.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
