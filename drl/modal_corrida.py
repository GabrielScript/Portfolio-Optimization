"""Fan a run out on Modal: one container per (strategy, test year), each writing the same cell file ``drl.executar``
writes locally. Replaces the four near-identical Modal scripts of the first version.

    DRL_LAGO=<lake exports> modal run drl/modal_corrida.py --run <name> --strategies ppo/dsr,ppo/dsr_drawdown

Set ``DRL_LAGO`` to the lake exports folder when running from a worktree.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import asdict
from pathlib import Path

import modal

RAIZ = Path(__file__).resolve().parent.parent
REMOTE_LAKE = "/root/lake"
LAKE_FILES = ("universe_prices.parquet", "universe_monthly.parquet", "cdi_daily.parquet")

app = modal.App("aleph-drl")

LAGO_LOCAL = Path(os.environ.get("DRL_LAGO", RAIZ / "data" / "b3lake"))   # a worktree has no data/: set DRL_LAGO

imagem = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gymnasium>=1.0", "stable-baselines3>=2.4", "pandas", "pyarrow", "numpy<3", "scipy", "scikit-learn")
)
for _arquivo in LAKE_FILES:
    imagem = imagem.add_local_file(str(LAGO_LOCAL / _arquivo), remote_path=f"{REMOTE_LAKE}/{_arquivo}")
imagem = imagem.add_local_python_source("drl")


@app.function(image=imagem, cpu=4, memory=8192, timeout=3 * 3600, max_containers=60)
def one_cell(strategy: str, year: int, spec: dict) -> dict:
    """Runs one cell and returns it; a failure propagates, so the local entry point reports it by name."""
    from drl import executar, protocolo
    from drl.dados import load_panel

    run_spec = executar.spec_from_dict(spec)
    panel = load_panel(Path(REMOTE_LAKE), run_spec.last_session)
    janelas = {j.ano_teste: j for j in protocolo.janelas(panel.sessions, executar.FIRST_YEAR, executar.LAST_YEAR)}
    return asdict(executar.run_cell(panel, janelas[year], strategy, run_spec))


@app.local_entrypoint()
def main(run: str, strategies: str = "ppo/dsr_cdi", timesteps: int = 150_000, seeds: str = "0,1,2,3,4",
         hold: int = 21, observation: str = "cross", cost_bps: float = 10.0, last_session: str = "",
         years: str = ""):
    from drl import executar

    spec = executar.RunSpec(executar.corrected_env(cost_bps, hold, observation),
                            tuple(int(s) for s in seeds.split(",")), timesteps, 4, last_session or None)
    anos = ([int(ano) for ano in years.split(",")] if years
            else list(range(executar.FIRST_YEAR, executar.LAST_YEAR + 1)))
    pedidos = [(estrategia.strip(), ano)
               for estrategia, ano in itertools.product(strategies.split(","),
                                                        anos)
               if not executar.cell_path(run, estrategia.strip(), ano).exists()]
    print(f"{len(pedidos)} cells", flush=True)
    resultados = one_cell.starmap([(estrategia, ano, asdict(spec)) for estrategia, ano in pedidos],
                                  return_exceptions=True)
    falhas = []
    for (estrategia, ano), resultado in zip(pedidos, resultados):
        if isinstance(resultado, Exception):
            falhas.append(f"{estrategia} {ano}: {type(resultado).__name__}: {resultado}")
            continue
        executar.write_cell(executar.Cell(**resultado), spec, executar.cell_path(run, estrategia, ano))
    print(f"{len(pedidos) - len(falhas)}/{len(pedidos)} cells written", flush=True)
    if falhas:
        raise SystemExit("failed cells:\n" + "\n".join(falhas))
