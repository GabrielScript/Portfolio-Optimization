"""
qml_sinal.py - The forecast models of the hybrid QPO (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md, sections
2-3): a PennyLane circuit, an MLP of its size and the circuit's dequantized twin, the ridge on its 729 Fourier features.
Runs in ``.venv-qpo``: the app imports neither torch nor PennyLane, so inputs and predictions cross as files.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
from torch import nn

from qpo_kaggle.qml_pacote import Pacote
from qpo_kaggle.qpo_qaoa import float64_default

N_ATRIBUTOS = 6
CAMADAS = 3
OCULTOS = 7
GRADE_DE_PENALIDADES = np.logspace(-6.0, 4.0, 41)      # times the mean squared singular value
TETO_DE_GRAUS = 0.9                                     # effective degrees of freedom / rows


def atributos_de_fourier(theta: np.ndarray) -> np.ndarray:
    """(rows, 6) angles -> (rows, 729): every product of one of {1, cos, sin} per wire, Prop. 1's basis."""
    base = np.ones((len(theta), 1))
    for j in range(theta.shape[1]):
        fator = np.stack([np.ones(len(theta)), np.cos(theta[:, j]), np.sin(theta[:, j])], axis=1)
        base = (base[:, :, None] * fator[:, None, :]).reshape(len(theta), -1)
    return base


class Vqc(nn.Module):
    """RY angle embedding, ``CAMADAS`` strongly entangling layers, <Z_j> on every wire, a linear head: (rows, 6) -> (rows,)."""

    def __init__(self, semente: int) -> None:
        super().__init__()
        dispositivo = qml.device("default.qubit", wires=N_ATRIBUTOS)

        @qml.qnode(dispositivo, interface="torch", diff_method="backprop")
        def circuito(inputs: torch.Tensor, weights: torch.Tensor) -> list:
            qml.AngleEmbedding(inputs, wires=range(N_ATRIBUTOS), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(N_ATRIBUTOS))
            return [qml.expval(qml.Z(j)) for j in range(N_ATRIBUTOS)]

        with torch.random.fork_rng(devices=[]), float64_default():
            torch.manual_seed(semente)
            self.circuito = qml.qnn.TorchLayer(circuito, {"weights": (CAMADAS, N_ATRIBUTOS, 3)},
                                               init_method=lambda t: nn.init.uniform_(t, 0.0, 2.0 * np.pi))
            self.cabeca = nn.Linear(N_ATRIBUTOS, 1)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        with float64_default():
            return self.cabeca(self.circuito(theta)).squeeze(-1)


class Mlp(nn.Module):
    """6-7-1 with tanh, the circuit's size: the classical control of the same training loop. (rows, 6) -> (rows,)."""

    def __init__(self, semente: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]), float64_default():
            torch.manual_seed(semente)
            self.rede = nn.Sequential(nn.Linear(N_ATRIBUTOS, OCULTOS), nn.Tanh(), nn.Linear(OCULTOS, 1))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.rede(theta).squeeze(-1)


# --- the walk-forward ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Treino:
    """Fixed by the pre-registration: 400 steps on the first month, then 50 warm-started."""

    passos_iniciais: int = 400
    passos_seguintes: int = 50
    lote: int = 1_024
    taxa: float = 0.02


@dataclass(frozen=True)
class AjusteRidge:
    coeficientes: np.ndarray
    media_x: np.ndarray
    media_y: float
    penalidade: float

    def prever(self, x: np.ndarray) -> np.ndarray:
        return (x - self.media_x) @ self.coeficientes + self.media_y


def ridge_gcv(x: np.ndarray, y: np.ndarray) -> AjusteRidge:
    """Ridge with an intercept, its penalty by generalized cross-validation on these rows alone, from one thin SVD."""
    media_x, media_y = x.mean(axis=0), float(y.mean())
    u, s, vt = np.linalg.svd(x - media_x, full_matrices=False)
    projecao = u.T @ (y - media_y)
    quadrados = s ** 2
    linhas = len(y)
    fora = float(((y - media_y) ** 2).sum() - (projecao ** 2).sum())

    def gcv(penalidade: float) -> float:
        encolhe = quadrados / (quadrados + penalidade)
        if encolhe.sum() > TETO_DE_GRAUS * linhas:         # near interpolation GCV is 0/0 and favours memorizing the rows
            return float("inf")
        residuo = fora + float((((1.0 - encolhe) * projecao) ** 2).sum())
        return residuo / linhas / (1.0 - encolhe.sum() / linhas) ** 2

    melhor = min(GRADE_DE_PENALIDADES * float(quadrados.mean()), key=gcv)
    return AjusteRidge(vt.T @ (s / (quadrados + melhor) * projecao), media_x, media_y, float(melhor))


def _prever_linear(pacote: Pacote, atributos: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    x = atributos(pacote.theta)
    previsao = np.full(len(pacote.y), np.nan)
    for mes, linhas in zip(pacote.previstos, pacote.treinos, strict=True):
        alvo = pacote.mes == mes
        previsao[alvo] = ridge_gcv(x[linhas], pacote.y[linhas]).prever(x[alvo])
    return previsao


def prever_fourier(pacote: Pacote) -> np.ndarray:
    """The circuit's dequantized twin: each month's ridge on the 729 features."""
    return _prever_linear(pacote, atributos_de_fourier)


def prever_ridge(pacote: Pacote) -> np.ndarray:
    """The linear control: each month's ridge on the angles."""
    return _prever_linear(pacote, lambda theta: theta)


def _ler_estado(arquivo: Path) -> dict | None:
    return torch.load(arquivo, map_location="cpu", weights_only=True) if arquivo.exists() else None


def _gravar_estado(arquivo: Path, estado: dict) -> None:
    temporario = arquivo.with_suffix(".tmp")
    torch.save(estado, temporario)
    temporario.replace(arquivo)


def prever_torch(classe: Callable[..., nn.Module], pacote: Pacote, semente: int, treino: Treino, pasta: Path) -> np.ndarray:
    """One seed of a torch model through the months: weights warm-started from the month before, a new Adam each month,
    the batch stream fixed by (seed, month). ``pasta`` keeps the state after each month, so a stopped run resumes."""
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / f"estado_{semente}.pt"
    modelo = classe(semente=semente)
    theta = torch.tensor(pacote.theta, dtype=torch.float64)
    y = torch.tensor(pacote.y, dtype=torch.float64)
    previsao = torch.full((len(pacote.y),), float("nan"), dtype=torch.float64)
    feitos = 0
    estado = _ler_estado(arquivo)
    if estado is not None:
        modelo.load_state_dict(estado["modelo"])
        previsao, feitos = estado["previsao"], int(estado["feitos"])
    for posicao, (mes, linhas) in enumerate(zip(pacote.previstos, pacote.treinos, strict=True)):
        if posicao < feitos:
            continue
        linhas_t = torch.as_tensor(linhas)
        gerador = torch.Generator().manual_seed(semente * 1_000_003 + int(mes))
        otimizador = torch.optim.Adam(modelo.parameters(), lr=treino.taxa)
        modelo.train()
        for _ in range(treino.passos_iniciais if posicao == 0 else treino.passos_seguintes):
            lote = linhas_t[torch.randint(len(linhas_t), (treino.lote,), generator=gerador)]
            otimizador.zero_grad(set_to_none=True)
            nn.functional.mse_loss(modelo(theta[lote]), y[lote]).backward()
            otimizador.step()
        modelo.eval()
        alvo = torch.as_tensor(np.flatnonzero(pacote.mes == mes))
        with torch.no_grad():
            previsao[alvo] = modelo(theta[alvo])
        _gravar_estado(arquivo, {"modelo": modelo.state_dict(), "previsao": previsao, "feitos": posicao + 1})
    return previsao.numpy()
