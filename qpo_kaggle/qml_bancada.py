"""
qml_bancada.py - The solver bench of the hybrid QPO (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, section 6), in
``.venv-qpo``. Each circuit gives a distribution over bitstrings, and one checker written from the problem's definition scores
every sample, so no solver grades itself. Wire i is bit n - 1 - i of the basis index, PennyLane's order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch

from qpo_kaggle.qaoa_subspace import SubspaceQaoa, linear_ramp
from qpo_kaggle.qpo_qaoa import float64_default
from qpo_kaggle.qubo_math import approximation_ratio

AMOSTRAS_DA_PENALIDADE = 1_000
QUANTOS_DO_VQE_RESTRITO = 10    # Mugel's post-processing
TOLERANCIA = 1e-12


@dataclass(frozen=True)
class Gabarito:
    minimo: float
    maximo: float
    x: np.ndarray


def pontuar(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The checker: x'Qx of each row, from the definition."""
    return np.einsum("ij,jk,ik->i", x, q, x)


def todos_os_estados(n: int) -> np.ndarray:
    indices = np.arange(2 ** n, dtype=np.uint32)
    return ((indices[:, None] >> (n - 1 - np.arange(n))) & 1).astype(float)


def exato(q: np.ndarray, k: int) -> Gabarito:
    """Plain enumeration of every |x| = k."""
    estados = todos_os_estados(len(q))
    viaveis = estados[estados.sum(axis=1) == k]
    energias = pontuar(q, viaveis)
    return Gabarito(float(energias.min()), float(energias.max()), viaveis[int(np.argmin(energias))])


def penalidade_de_hodson(q: np.ndarray, k: int, semente: int) -> float:
    """A = max E - min E over a sample of feasible portfolios (Hodson et al. 2019, eq. 25), so one name too many or too few
    costs at least the whole range of the objective."""
    energias = pontuar(q, uniforme_viavel(len(q), k, AMOSTRAS_DA_PENALIDADE, semente))
    return float(energias.max() - energias.min())


def energias_com_penalidade(q: np.ndarray, k: int, penalidade: float) -> np.ndarray:
    """x'Qx + A (|x| - k)^2 on the whole basis."""
    estados = todos_os_estados(len(q))
    return pontuar(q, estados) + penalidade * (estados.sum(axis=1) - k) ** 2


def _escalada(diagonal: np.ndarray) -> np.ndarray:
    """Centred and divided by its range, so the angles are O(1) in any units; the spectrum's order does not change."""
    return (diagonal - diagonal.mean()) / max(float(diagonal.max() - diagonal.min()), TOLERANCIA)


def _mixer_x(estado: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    """RX(2 beta) on every wire: the same angle on all, so the wire order does not matter."""
    cosseno, seno = torch.cos(beta), -1j * torch.sin(beta)
    for fio in range(n):
        pares = estado.reshape(-1, 2, 2 ** fio)
        estado = torch.stack((cosseno * pares[:, 0] + seno * pares[:, 1],
                              seno * pares[:, 0] + cosseno * pares[:, 1]), dim=1).reshape(-1)
    return estado


def _no_qaoa_soft(n: int, diagonal: torch.Tensor):
    """Simulated by hand from the diagonal: PennyLane's DiagonalQubitUnitary builds the dense 2^n x 2^n matrix, 68.7 GB at
    n = 16. A test holds its probabilities and gradient to PennyLane's circuit on small n."""
    def circuito(angulos: torch.Tensor) -> torch.Tensor:
        estado = torch.full((2 ** n,), 2.0 ** (-n / 2.0), dtype=torch.complex128)
        for gama, beta in zip(angulos[0], angulos[1], strict=True):
            estado = _mixer_x(torch.exp(-1j * gama * diagonal) * estado, beta, n)
        return estado.real ** 2 + estado.imag ** 2
    return circuito


def _no_vqe(n: int):
    @qml.qnode(qml.device("default.qubit", wires=n), interface="torch", diff_method="backprop")
    def circuito(angulos: torch.Tensor):
        for posicao, camada in enumerate(angulos):
            for fio in range(n):
                qml.RY(camada[fio], wires=fio)
            if posicao < len(angulos) - 1:
                for fio in range(n - 1):
                    qml.CNOT(wires=[fio, fio + 1])
        return qml.probs(wires=range(n))
    return circuito


def _treinar(no, inicio: np.ndarray, custo: torch.Tensor, passos: int, taxa: float) -> np.ndarray:
    angulos = torch.tensor(inicio, dtype=torch.float64, requires_grad=True)
    otimizador = torch.optim.Adam([angulos], lr=taxa)
    for _ in range(passos):
        otimizador.zero_grad(set_to_none=True)
        (no(angulos) * custo).sum().backward()
        otimizador.step()
    with torch.no_grad():
        return no(angulos).numpy()


def probabilidades_qaoa_soft(diagonal: np.ndarray, angulos: np.ndarray) -> np.ndarray:
    """|+> start, cost layer exp(-i gamma diag), X mixer; ``angulos`` (2, p): gammas, then betas."""
    with float64_default(), torch.no_grad():
        no = _no_qaoa_soft(int(np.log2(len(diagonal))), torch.tensor(diagonal, dtype=torch.float64))
        return no(torch.tensor(angulos, dtype=torch.float64)).numpy()


def treinar_qaoa_soft(diagonal: np.ndarray, p: int, passos: int, taxa: float) -> np.ndarray:
    with float64_default():
        custo = torch.tensor(_escalada(diagonal), dtype=torch.float64)
        return _treinar(_no_qaoa_soft(int(np.log2(len(diagonal))), custo), linear_ramp(p), custo, passos, taxa)


def probabilidades_vqe(angulos: np.ndarray) -> np.ndarray:
    """RY layers with a CNOT ladder between them (Mugel et al. 2022, fig. 1); ``angulos`` (layers + 1, n)."""
    with float64_default(), torch.no_grad():
        return _no_vqe(angulos.shape[1])(torch.tensor(angulos, dtype=torch.float64)).numpy()


def treinar_vqe(diagonal: np.ndarray, camadas: int, passos: int, taxa: float, semente: int) -> np.ndarray:
    n = int(np.log2(len(diagonal)))
    inicio = np.random.default_rng(semente).normal(0.0, 0.1, (camadas + 1, n))
    with float64_default():
        return _treinar(_no_vqe(n), inicio, torch.tensor(_escalada(diagonal), dtype=torch.float64), passos, taxa)


def treinar_qaoa_hard(q: np.ndarray, k: int, p: int, passos: int, taxa: float) -> tuple[np.ndarray, np.ndarray]:
    """Dicke start and XY ring on the weight-k subspace: its distribution and the states it lives on."""
    simulador = SubspaceQaoa(q, k)
    angulos = torch.tensor(linear_ramp(p), dtype=torch.float64, requires_grad=True)
    otimizador = torch.optim.Adam([angulos], lr=taxa)
    for _ in range(passos):
        otimizador.zero_grad(set_to_none=True)
        simulador.expval(angulos).backward()
        otimizador.step()
    with torch.no_grad():
        return simulador.probabilities(angulos).numpy(), simulador.bits.astype(float)


def amostrar(probabilidades: np.ndarray, estados: np.ndarray, tiros: int, semente: int) -> np.ndarray:
    limpas = np.clip(probabilidades, 0.0, None)
    return estados[np.random.default_rng(semente).choice(len(estados), size=tiros, p=limpas / limpas.sum())]


def uniforme_viavel(n: int, k: int, tiros: int, semente: int) -> np.ndarray:
    """The null: uniform over |x| = k."""
    escolhidos = np.random.default_rng(semente).random((tiros, n)).argsort(axis=1)[:, :k]
    amostras = np.zeros((tiros, n))
    np.put_along_axis(amostras, escolhidos, 1.0, axis=1)
    return amostras


def uniforme_na_base(n: int, tiros: int, semente: int) -> np.ndarray:
    """The control for repair: uniform over all 2^n, so repair earns credit only above what it gives to noise."""
    return np.random.default_rng(semente).integers(0, 2, (tiros, n)).astype(float)


def _reparar_um(s: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    x = x.copy()
    while x.sum() != k:
        campo = s @ x
        if x.sum() > k:
            dentro = np.flatnonzero(x)
            x[dentro[int(np.argmin(np.diag(s)[dentro] - 2.0 * campo[dentro]))]] = 0.0
        else:
            fora = np.flatnonzero(x == 0.0)
            x[fora[int(np.argmin(np.diag(s)[fora] + 2.0 * campo[fora]))]] = 1.0
    return x


def reparar(q: np.ndarray, amostras: np.ndarray, k: int) -> np.ndarray:
    """Each sample to |x| = k by the single removal or addition that gives the lowest energy, repeated; a feasible one stays."""
    s = (q + q.T) / 2.0
    unicos, inverso = np.unique(amostras, axis=0, return_inverse=True)
    return np.array([_reparar_um(s, x, k) for x in unicos])[np.ravel(inverso)]


def vqe_restrito(amostras: np.ndarray, q_sem_giro: np.ndarray, q_com_giro: np.ndarray, k: int,
                 quantos: int = QUANTOS_DO_VQE_RESTRITO) -> np.ndarray:
    """Mugel's post-processing: the ``quantos`` lowest-energy feasible samples without turnover, then the best with it."""
    viaveis = np.unique(amostras[amostras.sum(axis=1) == k], axis=0)
    melhores = viaveis[np.argsort(pontuar(q_sem_giro, viaveis), kind="stable")[:quantos]]
    return melhores[int(np.argmin(pontuar(q_com_giro, melhores)))]


def medir(amostras: np.ndarray, q: np.ndarray, k: int, gabarito: Gabarito) -> dict[str, float]:
    """Share of shots with the right size, share on the optimum, and the approximation ratio of the feasible shots' mean and
    of the best one; NaN when no shot is feasible."""
    viaveis = amostras[amostras.sum(axis=1) == k]
    if len(viaveis) == 0:
        return {"viavel": 0.0, "p_otimo": 0.0, "razao_media": float("nan"), "razao_melhor": float("nan")}
    energias = pontuar(q, viaveis)
    return {"viavel": len(viaveis) / len(amostras),
            "p_otimo": float((energias <= gabarito.minimo + 1e-10).sum() / len(amostras)),
            "razao_media": approximation_ratio(float(energias.mean()), gabarito.minimo, gabarito.maximo),
            "razao_melhor": approximation_ratio(float(energias.min()), gabarito.minimo, gabarito.maximo)}
