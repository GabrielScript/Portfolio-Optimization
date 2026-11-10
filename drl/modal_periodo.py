"""The whole out-of-sample period, 2010 to 2026, one test year per cell, fanned out on Modal.

The screening grid of ``drl.grade`` picked four years to decide *whether* any agent was worth a second look. This is the
other question: build the continuous 2010-2026 curve by stitching seventeen one-year out-of-sample segments, each
trained only on the five years before it, so the agent can be read against the Composite on the same ruler (CAGR,
Sharpe, Sortino, drawdown).

    modal run drl/modal_periodo.py --configuracoes ppo/dsr,ppo/dsr_drawdown,a2c/valor,ddpg/valor

Resumable: a cell already in ``drl/saidas/resultados_periodo.csv`` is skipped.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import modal

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "drl" / "saidas" / "resultados_periodo.csv"
PRIMEIRO_ANO = 2010
ULTIMO_ANO = 2026

app = modal.App("aleph-drl-periodo")

imagem = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gymnasium>=1.0", "stable-baselines3>=2.4", "pandas", "pyarrow", "numpy<3")
    .add_local_file(str(RAIZ / "lago" / "precos_diarios.parquet"), remote_path="/root/lago/precos_diarios.parquet")
    .add_local_python_source("drl")
)


@app.function(image=imagem, cpu=4, timeout=3600, max_containers=85)
def uma_celula(config: tuple[str, str, float, int, int, int]) -> dict:
    """One cell: (algoritmo, recompensa, custo_bps, ano_teste, semente, timesteps)."""
    from drl import dados, executar, protocolo

    algoritmo, recompensa, custo, ano, semente, timesteps = config
    fechamentos = dados.painel(Path("/root/lago/precos_diarios.parquet"))
    elegiveis = dados.elegiveis(fechamentos)
    janelas = {j.ano_teste: j for j in protocolo.janelas(fechamentos.index, PRIMEIRO_ANO, ULTIMO_ANO)}
    janela = janelas[ano]
    nomes = executar.nomes_elegiveis(elegiveis, janela.teste.start, 30)
    try:
        return executar.corrida(fechamentos, elegiveis, janela, nomes, algoritmo=algoritmo, recompensa=recompensa,
                                custo_bps=custo, timesteps=timesteps, seed=semente, n_envs=4)
    except Exception as erro:                    # noqa: BLE001 - a broken cell must not take the grid down
        return {"algoritmo": algoritmo, "recompensa": recompensa, "custo_bps": custo, "ano_teste": ano,
                "semente": semente, "erro": f"{type(erro).__name__}: {erro}"}


@app.local_entrypoint()
def principal(configuracoes: str = "ppo/dsr,ppo/dsr_drawdown,a2c/valor,ddpg/valor", custo_bps: float = 10.0,
              timesteps: int = 150_000):
    """Every (configuration, year) of the period that is not in the result file yet."""
    import pandas as pd

    pares = [tuple(p.strip().split("/")) for p in configuracoes.split(",")]
    feitos = set()
    if ARQUIVO.exists():
        d = pd.read_csv(ARQUIVO, on_bad_lines="skip")
        feitos = {(r.algoritmo, r.recompensa, float(r.custo_bps), int(r.ano_teste), int(r.semente))
                  for r in d.itertuples()}
    celulas = [(*par, custo_bps, ano, 0, timesteps)
               for par, ano in itertools.product(pares, range(PRIMEIRO_ANO, ULTIMO_ANO + 1))
               if (par[0], par[1], float(custo_bps), ano, 0) not in feitos]
    print(f"{len(celulas)} cells", flush=True)
    if not celulas:
        return
    linhas = list(uma_celula.map(celulas))
    ok = [linha for linha in linhas if "erro" not in linha]
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    if ok:
        pd.DataFrame(ok).to_csv(ARQUIVO, mode="a", header=not ARQUIVO.exists(), index=False)
    for linha in linhas:
        if "erro" in linha:
            print("FAIL", linha, flush=True)
        else:
            print(f"{linha['algoritmo']}/{linha['recompensa']}/{linha['ano_teste']} "
                  f"ret={linha['retorno_anual']:.1%} sharpe={linha['sharpe']:.2f} mdd={linha['mdd']:.1%}",
                  flush=True)
    print(f"{len(ok)}/{len(celulas)} cells ok -> {ARQUIVO}", flush=True)
