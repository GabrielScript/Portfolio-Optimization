"""
solve.py - QAOA on every exported QUBO (design doc, sections 7 and 8, step 2).

Reads ``fase{F}/{date}.npz`` (from scripts/qpo_montar.py) and writes ``fase{F}/{date}_p{p}.json`` with the QUBO's
sha256, the picks, the approximation ratio against the exact solution stored in the npz, the angles, the seed, the
backend and the versions. A chain = one (phase, p): its rebalances run in date order, each warm-started from the
previous one's angles (parameter concentration), so chains are independent and run in parallel processes. An
existing JSON is kept and its angles carry the warm start, so an interrupted run resumes where it stopped.

Backends: ``subspace`` (production: ``qaoa_subspace.py``, exact on the weight-k subspace, CUDA if visible else CPU)
and the PennyLane reference ``qpo_qaoa.py`` on ``pennylane-gpu`` (lightning.gpu, never a CPU fallback) or
``pennylane-cpu`` (lightning.qubit). The same circuit, optimizer and seeds; only the sampler's random stream differs.

    python -m qpo_kaggle.solve docs/qpo/qubos RESULTS 1,2,3 0 [BACKEND] [WORKERS]
"""

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from qpo_kaggle.qaoa_subspace import solve_qaoa as solve_subspace

STEPS = 150
LR = 0.05
SHOTS = 4096
SEED = 20261027
BACKENDS = ("subspace", "pennylane-gpu", "pennylane-cpu")


def _ratio(e: float, e_best: float, e_worst: float) -> float:
    return 1.0 if e_worst == e_best else (e_worst - e) / (e_worst - e_best)


def _solver(backend: str):
    """(solve, versions); PennyLane is imported only for the reference backends."""
    if backend == "subspace":
        return partial(solve_subspace, device="cuda" if torch.cuda.is_available() else "cpu"), {}
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r} is not one of {BACKENDS}")
    import pennylane as qml

    from qpo_kaggle.qpo_qaoa import solve_qaoa as solve_reference
    gpu = backend == "pennylane-gpu"
    return (partial(solve_reference, device_name="lightning.gpu" if gpu else "lightning.qubit", require_gpu=gpu),
            {"pennylane": qml.__version__})


def _run_chain(chain: tuple[int, int], input_dir: Path, output_dir: Path, backend: str, steps: int,
               shots: int) -> int:
    fase, p = chain
    torch.set_num_threads(1)                 # one chain per process: the vectors are small, threads only contend
    resolver, versoes = _solver(backend)
    anterior, resolvidos = None, 0
    for indice, arquivo in enumerate(sorted((Path(input_dir) / f"fase{fase}").glob("*.npz"))):
        destino = Path(output_dir) / f"fase{fase}" / f"{arquivo.stem}_p{p}.json"
        if destino.exists():
            anterior = json.loads(destino.read_text(encoding="utf-8"))["angles"]
            continue
        with np.load(arquivo) as dados:
            q, k, tickers = dados["q"], int(dados["k"]), dados["tickers"]
            exatas = tuple(int(i) for i in dados["exact_picks"])
            e_exata, e_pior = float(dados["exact_energy"]), float(dados["exact_worst"])
        semente = SEED + 1000 * fase + indice
        resultado = resolver(q, k, p=p, steps=steps, lr=LR, shots=shots, seed=semente, init=anterior)
        registro = {
            "phase": fase, "date": arquivo.stem, "p": p, "sha256": hashlib.sha256(arquivo.read_bytes()).hexdigest(),
            "picks": list(resultado.picks), "exact_picks": list(exatas),
            "tickers": [str(tickers[i]) for i in resultado.picks],
            "energy": resultado.energy, "exact_energy": e_exata, "ratio": _ratio(resultado.energy, e_exata, e_pior),
            "match": resultado.picks == exatas, "p_pick": resultado.p_optimum,
            "expval_initial": resultado.expval_inicial, "expval_final": resultado.expval_final,
            "n_infeasible": resultado.n_infeasible, "init": anterior, "angles": resultado.angles, "seed": semente,
            "steps": steps, "shots": shots, "seconds": resultado.seconds, "backend": backend,
            "device": resultado.device, "torch": torch.__version__, **versoes,
        }
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(registro, indent=1), encoding="utf-8")
        anterior = resultado.angles
        resolvidos += 1
        print(f"fase {fase} {arquivo.stem} p{p}: match {registro['match']} r {registro['ratio']:.4f} "
              f"{resultado.seconds:.1f} s", flush=True)
    return resolvidos


def run(input_dir: Path, output_dir: Path, ps: tuple[int, ...], phases: tuple[int, ...], backend: str = "subspace",
        workers: int = 1, steps: int = STEPS, shots: int = SHOTS) -> int:
    """Every (phase, p) chain; ``workers`` > 1 runs chains in parallel processes (same results, see the tests)."""
    cadeias = [(fase, p) for fase in phases for p in ps]
    tarefa = partial(_run_chain, input_dir=Path(input_dir), output_dir=Path(output_dir), backend=backend,
                     steps=steps, shots=shots)
    if workers <= 1:
        return sum(map(tarefa, cadeias))
    with ProcessPoolExecutor(max_workers=min(workers, len(cadeias))) as executor:
        return sum(executor.map(tarefa, cadeias))


def find_input(root: Path) -> Path:
    """The folder with the exported QUBOs' SHA256SUMS anywhere under ``root`` (Kaggle has mounted datasets at
    /kaggle/input/<slug> and at /kaggle/input/datasets/<user>/<slug>)."""
    achados = sorted(Path(root).rglob("SHA256SUMS"))
    if len(achados) != 1:
        raise FileNotFoundError(f"{len(achados)} SHA256SUMS under {root}: need exactly one")
    return achados[0].parent


def main(argv: list[str]) -> int:
    """``solve.py INPUT OUTPUT PS PHASES [BACKEND] [WORKERS]``; INPUT ``auto`` = the dataset under /kaggle/input."""
    entrada = find_input(Path("/kaggle/input")) if argv[0] == "auto" else Path(argv[0])
    backend = argv[4] if len(argv) > 4 else "subspace"
    workers = int(argv[5]) if len(argv) > 5 else 1
    total = run(entrada, Path(argv[1]), tuple(int(p) for p in argv[2].split(",")),
                tuple(int(f) for f in argv[3].split(",")), backend, workers)
    print(f"{total} QUBOs solved ({backend})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
