"""
spike.py - QPO timing spike (design doc, section 7): synthetic QUBOs only, no market data, so it is leak-free and
runs before the pre-registration commit. It fixes N_MAX = the largest n whose projected GPU time for the whole
study is <= BUDGET_HOURS.

Kaggle: ``python qpo_kaggle/build_kernel.py spike`` writes the single-file kernel (the solver is inlined).
Local check: ``python qpo_kaggle/spike.py --cpu-smoke`` (lightning.qubit, tiny n).
"""

import json
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from qpo_kaggle.qpo_qaoa import _linear_ramp, open_device, qaoa_qnodes  # KERNEL: inlined

K = 15
STEPS = 150                    # Adam steps per QUBO (design doc)
SHOTS = 4096
REBALANCES = 198               # 66 per phase x 3 phases, counted on docs/dados/2026-10-22 cut at 16/10/2026
PHASE0 = 66                    # p in {1, 2, 3} is chosen on phase 0, so phase 0 runs all three
BUDGET_HOURS = 20.0
SEED = 20261027


def synthetic_qubo(n: int, k: int, seed: int) -> np.ndarray:
    """One-factor Sigma (market vol 20%, idiosyncratic 20-40%) and Grinold-Kahn alphas, as the real QUBO is built."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, n)
    cov = 0.04 * np.outer(beta, beta) + np.diag(rng.uniform(0.2, 0.4, n) ** 2)
    alpha = 0.059 * np.sqrt(np.diag(cov)) * rng.normal(size=n)
    return 1.0 / (2.0 * k * k) * cov - np.diag(alpha) / k


def measure(n: int, k: int, p: int, device_name: str, require_gpu: bool) -> dict:
    q = synthetic_qubo(n, k, SEED + n)
    device = open_device(device_name, n, SEED, require_gpu)
    custo, amostrar = qaoa_qnodes(q, k, p, device, SHOTS)
    angulos = torch.tensor(_linear_ramp(p), dtype=torch.float64, requires_grad=True)
    inicio = time.perf_counter()
    custo(angulos).backward()
    primeira = time.perf_counter() - inicio
    tempos = []
    for _ in range(3):
        angulos.grad = None
        inicio = time.perf_counter()
        custo(angulos).backward()
        tempos.append(time.perf_counter() - inicio)
    inicio = time.perf_counter()
    amostras = np.asarray(amostrar(angulos.detach()))
    amostra = time.perf_counter() - inicio
    pesos = set(amostras.reshape(SHOTS, n).sum(axis=1).tolist())
    return {"n": n, "k": k, "p": p, "device": device.name, "first_s": primeira, "grad_s": statistics.median(tempos),
            "sample_s": amostra, "weights_seen": sorted(pesos), "qubo_s": STEPS * statistics.median(tempos) + amostra}


def project(linhas: list[dict]) -> dict[int, float]:
    """GPU hours per n: phase 0 runs p = 1, 2, 3; phases 21 and 42 run the chosen p (worst case: p = 3)."""
    horas = {}
    for n in sorted({linha["n"] for linha in linhas}):
        por_p = {linha["p"]: linha["qubo_s"] for linha in linhas if linha["n"] == n and "qubo_s" in linha}
        if len(por_p) == 3:
            horas[n] = (PHASE0 * sum(por_p.values()) + (REBALANCES - PHASE0) * por_p[3]) / 3600.0
    return horas


def _gpu_report() -> str:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return "nvidia-smi not found"


def main(argv: list[str]) -> int:
    smoke = "--cpu-smoke" in argv
    device_name, require_gpu = ("lightning.qubit", False) if smoke else ("lightning.gpu", True)
    tamanhos, k = ((6, 8), 3) if smoke else ((16, 20, 24, 28), K)
    import pennylane as qml
    print(json.dumps({"pennylane": qml.__version__, "torch": torch.__version__, "numpy": np.__version__,
                      "cuda": torch.cuda.is_available(),
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))
    print(_gpu_report())
    linhas = []
    for n in tamanhos:
        for p in (1, 2, 3):
            try:
                linha = measure(n, k, p, device_name, require_gpu)
            except Exception as erro:          # an OOM at large n is a result, not a crash: record it, go on
                if require_gpu and "GPU required" in str(erro):
                    raise
                linha = {"n": n, "k": k, "p": p, "error": f"{type(erro).__name__}: {erro}"}
            print(json.dumps(linha), flush=True)
            linhas.append(linha)
    horas = project(linhas)
    n_max = max((n for n, h in horas.items() if h <= BUDGET_HOURS), default=None)
    resumo = {"projected_gpu_hours": horas, "budget_hours": BUDGET_HOURS, "N_MAX": n_max}
    print(json.dumps(resumo))
    print(_gpu_report())
    with open("spike.json", "w", encoding="utf-8") as saida:
        json.dump({"rows": linhas, **resumo}, saida, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
