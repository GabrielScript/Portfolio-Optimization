"""
solve.py - Kaggle job of the QPO (design doc, sections 7 and 8, step 2): QAOA on every exported QUBO.

Reads ``fase{F}/{date}.npz`` (from scripts/qpo_montar.py) and writes ``fase{F}/{date}_p{p}.json`` with the QUBO's
sha256, the picks, the approximation ratio against the exact solution stored in the npz, the angles, the seed, the
device and the versions. Rebalances run in date order, each warm-started from the previous one's angles (parameter
concentration). An existing JSON is kept and its angles carry the warm start, so a session cut by Kaggle's time
limit resumes where it stopped.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from qpo_kaggle.qpo_qaoa import solve_qaoa  # KERNEL: inlined

STEPS = 150
LR = 0.05
SHOTS = 4096
SEED = 20261027


def _ratio(e: float, e_best: float, e_worst: float) -> float:
    return 1.0 if e_worst == e_best else (e_worst - e) / (e_worst - e_best)


def run(input_dir: Path, output_dir: Path, ps: tuple[int, ...], phases: tuple[int, ...], device_name: str,
        require_gpu: bool, steps: int = STEPS, shots: int = SHOTS) -> int:
    resolvidos = 0
    for fase in phases:
        arquivos = sorted((Path(input_dir) / f"fase{fase}").glob("*.npz"))
        for p in ps:
            anterior = None
            for indice, arquivo in enumerate(arquivos):
                destino = Path(output_dir) / f"fase{fase}" / f"{arquivo.stem}_p{p}.json"
                if destino.exists():
                    anterior = json.loads(destino.read_text(encoding="utf-8"))["angles"]
                    continue
                dados = np.load(arquivo)
                q, k = dados["q"], int(dados["k"])
                semente = SEED + 1000 * fase + indice
                resultado = solve_qaoa(q, k, p=p, steps=steps, lr=LR, shots=shots, seed=semente,
                                       device_name=device_name, init=anterior, require_gpu=require_gpu)
                exatas = tuple(int(i) for i in dados["exact_picks"])
                registro = {
                    "phase": fase, "date": arquivo.stem, "p": p,
                    "sha256": hashlib.sha256(arquivo.read_bytes()).hexdigest(),
                    "picks": list(resultado.picks), "exact_picks": list(exatas),
                    "tickers": [str(dados["tickers"][i]) for i in resultado.picks],
                    "energy": resultado.energy, "exact_energy": float(dados["exact_energy"]),
                    "ratio": _ratio(resultado.energy, float(dados["exact_energy"]), float(dados["exact_worst"])),
                    "match": resultado.picks == exatas, "p_pick": resultado.p_optimum,
                    "expval_initial": resultado.expval_inicial, "expval_final": resultado.expval_final,
                    "n_infeasible": resultado.n_infeasible, "init": anterior, "angles": resultado.angles,
                    "seed": semente, "steps": steps, "shots": shots, "seconds": resultado.seconds,
                    "device": resultado.device, "pennylane": qml.__version__, "torch": torch.__version__,
                }
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_text(json.dumps(registro, indent=1), encoding="utf-8")
                anterior = resultado.angles
                resolvidos += 1
                print(f"fase {fase} {arquivo.stem} p{p}: match {registro['match']} r {registro['ratio']:.4f} "
                      f"{resultado.seconds:.0f} s", flush=True)
    return resolvidos


def find_input(root: Path) -> Path:
    """The folder with the exported QUBOs' SHA256SUMS anywhere under ``root`` (Kaggle has mounted datasets at
    /kaggle/input/<slug> and at /kaggle/input/datasets/<user>/<slug>)."""
    achados = sorted(Path(root).rglob("SHA256SUMS"))
    if len(achados) != 1:
        raise FileNotFoundError(f"{len(achados)} SHA256SUMS under {root}: need exactly one")
    return achados[0].parent


def main(argv: list[str]) -> int:
    """``solve.py INPUT OUTPUT PS PHASES``, e.g. ``auto /kaggle/working 1,2,3 0`` (auto = find under /kaggle/input)."""
    entrada = find_input(Path("/kaggle/input")) if argv[0] == "auto" else Path(argv[0])
    saida = Path(argv[1])
    ps = tuple(int(p) for p in argv[2].split(","))
    fases = tuple(int(f) for f in argv[3].split(","))
    print(f"{run(entrada, saida, ps, fases, 'lightning.gpu', require_gpu=True)} QUBOs solved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
