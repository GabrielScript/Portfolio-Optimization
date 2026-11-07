"""
kan.py - Kolmogorov-Arnold networks for the forecast test (docs/superpowers/specs/2026-11-07-kan-design.md, section 3).
Written from the paper's definition (Liu et al., arXiv 2404.19756v5, eq. 2.10), after efficient-kan's ``KANLinear`` (MIT):
each edge is phi(u) = w_b silu(u) + w_s sum_j c_j B_j(u), with cubic B-splines on a uniform grid extended by the degree on
each side. Runs in ``.venv-qpo`` beside ``qml_sinal``; the app never imports torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from qpo_kaggle.qml_sinal import N_ATRIBUTOS

GRADE = 5
GRAU = 3
FAIXA_DOS_ANGULOS = (-np.pi / 2, np.pi / 2)     # theta = pi (u - 1/2) never leaves it
FAIXA_OCULTA = (-2.0, 2.0)
OCULTOS = 2
RUIDO_DOS_COEFICIENTES = 0.1


class KanCamada(nn.Module):
    """``entradas`` -> ``saidas``: every output is the sum over inputs of one learned univariate function per edge."""

    def __init__(self, entradas: int, saidas: int, semente: int, faixa: tuple[float, float],
                 grade: int = GRADE, grau: int = GRAU) -> None:
        super().__init__()
        inicio, fim = faixa
        passo = (fim - inicio) / grade
        nos = inicio + passo * torch.arange(-grau, grade + grau + 1, dtype=torch.float64)
        self.register_buffer("nos", nos)                                  # (grade + 2 grau + 1,)
        self.grau = grau
        funcoes = grade + grau
        gerador = torch.Generator().manual_seed(semente)
        limite = 1.0 / np.sqrt(entradas)
        self.peso_base = nn.Parameter(
            (torch.rand(saidas, entradas, generator=gerador, dtype=torch.float64) * 2.0 - 1.0) * limite)
        self.coeficientes = nn.Parameter(
            (torch.rand(saidas, entradas, funcoes, generator=gerador, dtype=torch.float64) * 2.0 - 1.0)
            * RUIDO_DOS_COEFICIENTES)
        self.escala = nn.Parameter(torch.ones(saidas, entradas, dtype=torch.float64))

    def base_de_splines(self, x: torch.Tensor) -> torch.Tensor:
        """(rows, entradas) -> (rows, entradas, grade + grau) by Cox-de Boor; zero outside the grid range."""
        t = self.nos
        u = x.unsqueeze(-1)
        base = ((u >= t[:-1]) & (u < t[1:])).to(x.dtype)
        for k in range(1, self.grau + 1):
            esquerda = (u - t[:-(k + 1)]) / (t[k:-1] - t[:-(k + 1)]) * base[..., :-1]
            direita = (t[k + 1:] - u) / (t[k + 1:] - t[1:-k]) * base[..., 1:]
            base = esquerda + direita
        return base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (rows, entradas) -> (rows, saidas)
        base = nn.functional.silu(x) @ self.peso_base.T
        splines = torch.einsum("rik,oik,oi->ro", self.base_de_splines(x), self.coeficientes, self.escala)
        return base + splines


class _Kan(nn.Module):
    def __init__(self, camadas: list[KanCamada]) -> None:
        super().__init__()
        self.camadas = nn.ModuleList(camadas)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # theta: (rows, 6) -> (rows,)
        saida = theta
        for camada in self.camadas:
            saida = camada(saida)
        return saida.squeeze(-1)


class Kan1(_Kan):
    """[6,1]: an additive model of the six angles, 60 parameters."""

    def __init__(self, semente: int) -> None:
        super().__init__([KanCamada(N_ATRIBUTOS, 1, semente * 10, FAIXA_DOS_ANGULOS)])


class Kan2(_Kan):
    """[6,2,1]: two hidden sums, so the inputs can interact; 140 parameters."""

    def __init__(self, semente: int) -> None:
        super().__init__([KanCamada(N_ATRIBUTOS, OCULTOS, semente * 10, FAIXA_DOS_ANGULOS),
                          KanCamada(OCULTOS, 1, semente * 10 + 1, FAIXA_OCULTA)])


@torch.no_grad()
def curvas(modelo: Kan1, pontos: int = 21) -> pd.DataFrame:
    """Each input's learned function on ``pontos`` points of the angle range, and 1 - R^2 of its best line."""
    camada = modelo.camadas[0]
    u = torch.linspace(*FAIXA_DOS_ANGULOS, pontos, dtype=torch.float64)
    entradas = u.unsqueeze(-1).expand(pontos, camada.peso_base.shape[1])
    base = camada.base_de_splines(entradas)                                # (pontos, entradas, funcoes)
    phi = (nn.functional.silu(entradas) * camada.peso_base[0]
           + torch.einsum("pik,ik,i->pi", base, camada.coeficientes[0], camada.escala[0])).numpy()
    linhas = []
    for atributo in range(phi.shape[1]):
        y = phi[:, atributo]
        reta = np.polyval(np.polyfit(u.numpy(), y, 1), u.numpy())
        total = float(((y - y.mean()) ** 2).sum())
        nao_linearidade = 0.0 if total == 0.0 else float(np.clip(((y - reta) ** 2).sum() / total, 0.0, 1.0))
        linhas.append(pd.DataFrame({"atributo": atributo, "u": u.numpy(), "phi": y, "nao_linearidade": nao_linearidade}))
    return pd.concat(linhas, ignore_index=True)
