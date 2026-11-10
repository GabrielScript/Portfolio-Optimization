"""The pre-registered grid, fanned out on Modal: one container per cell, each with four cores, so the whole grid
finishes in about the time of its slowest cell instead of the sum. Same code path as ``drl.grade`` — this file only
moves the cells off this laptop.

    modal run drl/modal_grade.py --timesteps 150000

The results come back as rows and are appended to ``drl/saidas/resultados_modal.csv``.
"""
from __future__ import annotations

from pathlib import Path

import modal

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO = RAIZ / "drl" / "saidas" / "resultados_modal.csv"

app = modal.App("aleph-drl-grade")

imagem = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gymnasium>=1.0", "stable-baselines3>=2.4", "pandas", "pyarrow", "numpy<3")
    .add_local_file(str(RAIZ / "lago" / "precos_diarios.parquet"), remote_path="/root/lago/precos_diarios.parquet")
    .add_local_python_source("drl")
)


@app.function(image=imagem, cpu=4, timeout=3600, max_containers=56)
def uma_celula(config: tuple[str, str, float, int, int, int]) -> dict:
    """One cell of the grid: (algoritmo, recompensa, custo_bps, ano_teste, semente, timesteps)."""
    from drl import dados, executar, protocolo

    algoritmo, recompensa, custo, ano, semente, timesteps = config
    fechamentos = dados.painel(Path("/root/lago/precos_diarios.parquet"))
    elegiveis = dados.elegiveis(fechamentos)
    janelas = {j.ano_teste: j for j in protocolo.janelas(fechamentos.index)}
    janela = janelas[ano]
    nomes = executar.nomes_elegiveis(elegiveis, janela.teste.start, 30)
    try:
        return executar.corrida(fechamentos, elegiveis, janela, nomes, algoritmo=algoritmo, recompensa=recompensa,
                                custo_bps=custo, timesteps=timesteps, seed=semente, n_envs=4)
    except Exception as erro:                    # noqa: BLE001 - a broken cell must not take the grid down
        return {"algoritmo": algoritmo, "recompensa": recompensa, "custo_bps": custo, "ano_teste": ano,
                "semente": semente, "erro": f"{type(erro).__name__}: {erro}"}


@app.local_entrypoint()
def principal(timesteps: int = 150_000, limite: int = 0):
    import itertools

    import pandas as pd

    from drl import grade as g

    celulas = [(*algoritmo_recompensa, custo, ano, semente, timesteps)
               for (algoritmo_recompensa, semente, custo, ano) in
               itertools.product(g.AGENTES, g.SEMENTES, g.CUSTOS, g.JANELAS)]
    if limite:
        celulas = celulas[:limite]
    print(f"{len(celulas)} cells", flush=True)
    linhas = list(uma_celula.map(celulas))
    ok = [linha for linha in linhas if "erro" not in linha]
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ok).to_csv(ARQUIVO, mode="a", header=not ARQUIVO.exists(), index=False)
    for linha in linhas:
        if "erro" in linha:
            print("FAIL", linha, flush=True)
        else:
            print(f"{linha['algoritmo']}/{linha['recompensa']}/{linha['custo_bps']:.0f}bps/{linha['ano_teste']} "
                  f"ret={linha['retorno_anual']:.1%} sharpe={linha['sharpe']:.2f} giro={linha['giro_anual']:.1f}",
                  flush=True)
    print(f"{len(ok)}/{len(celulas)} cells ok -> {ARQUIVO}", flush=True)
