"""
poder.py - Poder dos portões do S2 por simulação.

Um sinal com IC de Spearman conhecido é plantado no painel mensal real do lago (universo ponto no tempo, retornos
totais do mês) e o portão é aplicado como seria aplicado a um sinal de verdade. A taxa de aprovação com IC 0 é o falso
positivo; com IC 0,03, o poder. O painel entra só como o ruído de corte transversal (quantos papéis e como os retornos
se distribuem); nenhum sinal real é olhado.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


def ic_por_mes(sinal: pd.DataFrame, retornos: pd.DataFrame) -> pd.Series:
    """IC de Spearman de cada mês (linha): correlação de postos entre sinal e retorno, nos papéis com os dois."""
    validos = sinal.notna() & retornos.notna()
    postos_sinal = sinal.where(validos).rank(axis=1)
    postos_retorno = retornos.where(validos).rank(axis=1)
    postos_sinal = postos_sinal.sub(postos_sinal.mean(axis=1), axis=0)
    postos_retorno = postos_retorno.sub(postos_retorno.mean(axis=1), axis=0)
    covariancia = (postos_sinal * postos_retorno).sum(axis=1)
    return covariancia / np.sqrt((postos_sinal**2).sum(axis=1) * (postos_retorno**2).sum(axis=1))


def sinal_plantado(retornos: pd.DataFrame, ic: float | np.ndarray, gerador: np.random.Generator) -> pd.DataFrame:
    """
    Sinal com IC de Spearman esperado ``ic`` em cada mês: rho * z + sqrt(1 - rho^2) * ruído normal, com z o escore normal
    do posto do retorno no mês e rho = 2 sen(pi ic / 6), a correlação de Pearson cujo Spearman, na normal, é ``ic``.
    Papel sem retorno no mês fica sem sinal; ``ic`` pode ser um por mês.
    """
    contagem = retornos.notna().sum(axis=1)
    quantis = retornos.rank(axis=1).sub(0.5).div(contagem, axis=0)
    escores = norm.ppf(quantis.to_numpy())
    rho = 2.0 * np.sin(np.pi * np.clip(np.asarray(ic, dtype=float), -1.0, 1.0) / 6.0).reshape(-1, 1)
    sinal = rho * escores + np.sqrt(1.0 - rho**2) * gerador.standard_normal(retornos.shape)
    return pd.DataFrame(sinal, index=retornos.index, columns=retornos.columns).where(retornos.notna())


@dataclass(frozen=True)
class Portao:
    """Portão do IC mensal: IC médio >= ``ic_minimo`` e t de Student da média >= ``t_minimo``."""

    nome: str
    ic_minimo: float
    t_minimo: float

    def aprova(self, ics: pd.Series) -> bool:
        ics = ics.dropna()
        media = float(ics.mean())
        t = media / (float(ics.std(ddof=1)) / np.sqrt(len(ics)))
        return media >= self.ic_minimo and t >= self.t_minimo


def taxa_de_aprovacao(retornos: pd.DataFrame, ic: float, portoes: Sequence[Portao], simulacoes: int,
                      gerador: np.random.Generator, desvio_do_ic: float = 0.0) -> dict[str, float]:
    """
    Fração das ``simulacoes`` em que cada portão aprova um sinal plantado com IC verdadeiro médio ``ic``. Com
    ``desvio_do_ic``, o IC de cada mês é sorteado em torno de ``ic`` (o de um sinal real varia de mês a mês).
    """
    aprovacoes: Counter[str] = Counter()
    for _ in range(simulacoes):
        ic_do_mes = ic + desvio_do_ic * gerador.standard_normal(len(retornos)) if desvio_do_ic else ic
        ics = ic_por_mes(sinal_plantado(retornos, ic_do_mes, gerador), retornos)
        aprovacoes.update(portao.nome for portao in portoes if portao.aprova(ics))
    return {portao.nome: aprovacoes[portao.nome] / simulacoes for portao in portoes}
