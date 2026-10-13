"""
reamostragem.py - Carteira reamostrada (bagging de Michaud) contra o erro de estimação de mu.

Por quê: com mu amostral, o otimizador concentra nos ativos de maior mu
estimado, o "maximizador de erro" de Michaud (1989). O shrinkage de Jorion
reduz o problema, mas não o resolve: no diagnóstico de 2026-10-13 (5 anos,
177 ativos, Jorion 0,53), 3 dos 8 ativos da carteira Kelly ficavam no teto de
20%, e só 3,1 dos 8, em média, reapareciam numa reamostra.

Método (Michaud e Michaud, 1998; Palomar, cap. 14; Cajas, cap. 11):
1. B reamostras por bootstrap CIRCULAR EM BLOCOS (blocos contíguos, com volta
   ao início), que preserva a dependência de curto prazo (volatilidade em
   cluster). O bootstrap iid destruiria essa dependência.
2. Em cada reamostra: Ledoit-Wolf + Jorion, o ajuste opcional de mu (tilt
   growth) e a otimização com o objetivo escolhido, SEM cardinalidade (a média
   precisa ver todos os ativos).
3. Média dos pesos. Ela satisfaz as restrições lineares (conjunto convexo),
   mas pode ter mais de k ativos e romper o teto de vol sob a covariância da
   amostra inteira. Por isso a projeção euclidiana final (``projetar_pesos``).

Premissas e limites:
- Os retornos são estacionários na janela; os blocos de 21 pregões (~1 mês)
  preservam a dependência de até ~1 mês, não a de prazo mais longo.
- A média das carteiras corta a cauda esquerda e também parte da direita: a
  carteira fica DENTRO da fronteira estimada, por construção.
- O HRP não é suportado: ele não usa o conjunto viável W, e a projeção o
  transformaria em outra coisa.
- Custo: B otimizações (~40-90 ms cada com 74-115 ativos). Paralelizar com
  threads ou processos não compensou (medido em 2026-10-13).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from core.config import ConfigOtimizacao, Objetivo
from core.estimation import estimar_momentos
from core.optimization import (ErroOtimizacao, ResultadoOtimizacao, metricas_carteira, otimizar,
                               preparar_covariancia, projetar_pesos)

logger = logging.getLogger(__name__)

LIMIAR_FREQUENCIA = 0.01         # peso acima disso conta como "ativo presente" na reamostra
_FALHAS_DA_REAMOSTRA = (ErroOtimizacao, ValueError, np.linalg.LinAlgError)

AjusteMu = Callable[[pd.Series, pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ConfigReamostragem:
    """Fixados antes da avaliação (pré-registro de 2026-10-13): B=100, blocos de 21, semente 0."""

    n_reamostras: int = 100
    tamanho_bloco: int = 21
    semente: int = 0

    def __post_init__(self) -> None:
        if self.n_reamostras < 1:
            raise ValueError(f"n_reamostras deve ser >= 1: {self.n_reamostras}")
        if self.tamanho_bloco < 1:
            raise ValueError(f"tamanho_bloco deve ser >= 1: {self.tamanho_bloco}")


@dataclass(frozen=True)
class ResultadoReamostragem:
    resultado: ResultadoOtimizacao
    frequencia: pd.Series        # fração das reamostras válidas em que o ativo teve peso > 1%
    reamostras_validas: int
    mu: pd.Series                # momentos da amostra INTEIRA (com o ajuste), os da projeção e
    cov: pd.DataFrame            # das métricas: a UI desenha a fronteira com eles
    intensidade_shrinkage: float


def indices_bootstrap_circular(n_obs: int, tamanho_bloco: int,
                               gerador: np.random.Generator) -> np.ndarray:
    """n_obs índices em blocos contíguos de ``tamanho_bloco``, com início sorteado e volta circular."""
    if not 1 <= tamanho_bloco <= n_obs:
        raise ValueError(f"tamanho_bloco precisa estar em [1, {n_obs}]: {tamanho_bloco}")
    n_blocos = -(-n_obs // tamanho_bloco)
    inicios = gerador.integers(0, n_obs, size=n_blocos)
    blocos = (inicios[:, None] + np.arange(tamanho_bloco)[None, :]) % n_obs
    return blocos.ravel()[:n_obs]


def _momentos_ajustados(retornos: pd.DataFrame, shrink_mu: bool,
                        ajustar_mu: AjusteMu | None) -> tuple[pd.Series, pd.DataFrame, float]:
    momentos = estimar_momentos(retornos, shrink_mu=shrink_mu)
    mu = ajustar_mu(momentos.mu, momentos.cov) if ajustar_mu is not None else momentos.mu
    return mu, momentos.cov, momentos.intensidade_shrinkage


def otimizar_reamostrado(retornos: pd.DataFrame, cfg: ConfigOtimizacao, *, shrink_mu: bool = True,
                         config: ConfigReamostragem = ConfigReamostragem(),
                         ajustar_mu: AjusteMu | None = None) -> ResultadoReamostragem:
    """
    Carteira reamostrada. ``retornos``: simples, diários, sem NaN. ``ajustar_mu``
    é aplicado em CADA reamostra e na amostra inteira (usada para as métricas
    exibidas e para o teto de vol da projeção).

    Lança ValueError (HRP, bloco maior que a amostra) e ErroOtimizacao se todas
    as reamostras falharem. Reamostras que falham são descartadas e contadas.
    """
    if cfg.objetivo is Objetivo.HRP:
        raise ValueError("Reamostragem não suportada para HRP (não usa o conjunto viável).")
    n_obs = len(retornos)
    if config.tamanho_bloco > n_obs:
        raise ValueError(f"tamanho_bloco {config.tamanho_bloco} > {n_obs} observações.")

    gerador = np.random.default_rng(config.semente)
    sorteios = [indices_bootstrap_circular(n_obs, config.tamanho_bloco, gerador)
                for _ in range(config.n_reamostras)]
    cfg_reamostra = replace(cfg, n_ativos_max=None)
    valores = retornos.values
    soma = pd.Series(0.0, index=retornos.columns)
    presencas = pd.Series(0.0, index=retornos.columns)
    falhas = 0
    for numero, indices in enumerate(sorteios):
        amostra = pd.DataFrame(valores[indices], columns=retornos.columns)
        try:
            mu, cov, _ = _momentos_ajustados(amostra, shrink_mu, ajustar_mu)
            pesos = otimizar(mu, cov, cfg_reamostra).pesos
        except _FALHAS_DA_REAMOSTRA as erro:
            falhas += 1
            logger.warning("Reamostra %d descartada: %s", numero, erro)
            continue
        soma += pesos
        presencas += (pesos > LIMIAR_FREQUENCIA).astype(float)

    validas = config.n_reamostras - falhas
    if validas == 0:
        raise ErroOtimizacao(f"Todas as {config.n_reamostras} reamostras falharam.")

    mu_inteira, cov_inteira, intensidade = _momentos_ajustados(retornos, shrink_mu, ajustar_mu)
    projecao = projetar_pesos(soma / validas, cov_inteira, cfg)
    retorno, vol, sharpe = metricas_carteira(projecao.pesos.values, mu_inteira.values,
                                             preparar_covariancia(cov_inteira), cfg.taxa_livre_risco)

    status = "impreciso" if falhas and projecao.status == "otimo" else projecao.status
    partes = [f"Reamostrada ({cfg.objetivo.value}): média de {validas} carteiras "
              f"(bootstrap em blocos de {config.tamanho_bloco} pregões)."]
    if falhas:
        partes.append(f"{falhas} de {config.n_reamostras} reamostras falharam e foram descartadas.")
    if projecao.mensagem:
        partes.append(projecao.mensagem)
    resultado = ResultadoOtimizacao(
        pesos=projecao.pesos, retorno_esperado=retorno, volatilidade=vol, sharpe=sharpe,
        objetivo=cfg.objetivo, status=status, mensagem=" ".join(partes),
    )
    return ResultadoReamostragem(resultado, presencas / validas, validas,
                                 mu_inteira, cov_inteira, intensidade)
