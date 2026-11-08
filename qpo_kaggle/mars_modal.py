"""
mars_modal.py - the MARS trainings on Modal (spec section 6): one GPU call per (arm, seed, year), up to 10 at once
(the free tier's GPU limit); the chosen snapshots go to the Volume ``mars-b3``; then one CPU call per (arm, seed)
chains the test years and returns the daily curve. ``mars.py`` and the data package ride in the image unchanged.

    modal run qpo_kaggle/mars_modal.py --etapa cronometrar
    modal run qpo_kaggle/mars_modal.py --etapa treinar
    modal run qpo_kaggle/mars_modal.py --etapa encadear --saida kan_run/mars/saidas
"""

from __future__ import annotations

import io
import itertools
import time
from pathlib import Path

import modal

RAIZ = Path(__file__).resolve().parent.parent
PACOTE = RAIZ / "kan_run" / "mars" / "mars_pacote.npz"
BRACOS = ("MARS-50", "MARS-50+score")
SEMENTES = (42, 0, 1)
ANOS = range(2010, 2027)
GPU = "L4"

imagem = (modal.Image.debian_slim(python_version="3.12")
          .pip_install("torch==2.14.1", "numpy==2.5.4")
          .add_local_file(Path(__file__).resolve().parent / "mars.py", "/root/mars.py")
          .add_local_file(PACOTE, "/root/mars_pacote.npz"))
volume = modal.Volume.from_name("mars-b3", create_if_missing=True)
app = modal.App("mars-b3", image=imagem)


def _cronometrar() -> dict[str, float]:
    import numpy as np
    import torch

    import mars
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pacote = np.load("/root/mars_pacote.npz")
    painel = mars.carregar_painel(pacote, 2022, com_nota=False)
    treino, validacao = int(painel.cortes[0]), int(painel.cortes[1])
    aprendiz = mars.Aprendiz(301, 50, mars.Treino(), dispositivo, validacao - treino)
    inicio = time.perf_counter()
    aprendiz.episodio(painel, treino, treino + 1_100, mars.Execucao(), np.random.default_rng(0))
    passos = time.perf_counter() - inicio
    inicio = time.perf_counter()
    for _ in range(300):
        aprendiz.atualizar()
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()
    return {"device": str(torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else "cpu"),
            "ms_por_passo_sem_atualizar": 1000 * passos / 1_100, "ms_por_atualizacao": 1000 * (time.perf_counter() - inicio) / 300}


@app.function(gpu="T4", timeout=900)
def cronometrar_t4() -> dict[str, float]:
    return _cronometrar()


@app.function(gpu="L4", timeout=900)
def cronometrar_l4() -> dict[str, float]:
    return _cronometrar()


@app.function(gpu=GPU, timeout=3 * 3600, volumes={"/vol": volume}, max_containers=10, retries=1)
def treinar(braco: str, semente: int, ano: int) -> tuple[str, int, int, int, float, float]:
    import numpy as np
    import torch

    import mars
    torch.set_num_threads(4)
    pacote = np.load("/root/mars_pacote.npz")
    episodio, nota, segundos = mars.treinar_e_guardar(pacote, braco, semente, ano, Path("/vol/estados"), mars.Treino(),
                                                      torch.device("cuda"))
    volume.commit()
    return braco, semente, ano, episodio, nota, segundos


@app.function(cpu=2, timeout=3600, volumes={"/vol": volume})
def encadear(braco: str, semente: int) -> bytes:
    import numpy as np

    import mars
    volume.reload()
    resultado = mars.encadear(np.load("/root/mars_pacote.npz"), braco, semente, Path("/vol/estados"))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **resultado)
    return buffer.getvalue()


@app.local_entrypoint()
def main(etapa: str, saida: str = "kan_run/mars/saidas") -> None:
    if etapa == "cronometrar":
        for funcao in (cronometrar_t4, cronometrar_l4):
            print(funcao.remote())
    elif etapa == "treinar":
        tarefas = list(itertools.product(BRACOS, SEMENTES, ANOS))
        inicio = time.perf_counter()
        for feito, resultado in enumerate(treinar.starmap(tarefas, order_outputs=False), start=1):
            braco, semente, ano, episodio, nota, segundos = resultado
            print(f"[{feito}/{len(tarefas)}, {time.perf_counter() - inicio:.0f} s] {braco} seed {semente} {ano}: "
                  f"episode {episodio}, validation Sharpe {nota:.2f}, {segundos:.0f} s", flush=True)
    elif etapa == "encadear":
        pasta = RAIZ / saida
        pasta.mkdir(parents=True, exist_ok=True)
        pares = list(itertools.product(BRACOS, SEMENTES))
        for (braco, semente), dados in zip(pares, encadear.starmap(pares)):
            destino = pasta / f"mars_{braco.replace('+', '_')}_s{semente}.npz"
            destino.write_bytes(dados)
            print(f"wrote {destino}", flush=True)
    else:
        raise SystemExit(f"unknown etapa {etapa}")
