"""
estimation.py - Estimadores de retorno esperado e covariância.

Decisões metodológicas (ver README, seção Metodologia):

1. Retornos SIMPLES, não log. A versão anterior usava média de log-retornos
   como mu. Para cada ativo, E[log(1+R)] ~ mu - sigma^2/2, então o otimizador
   punia ativos voláteis. Além disso, por Jensen, w' E[log(1+R)] <= E[log(1+w'R)],
   com folga zero só em carteiras de um ativo: a média de log-retornos
   subestima o crescimento de carteiras diversificadas em relação ao de um
   ativo isolado. O efeito sobre o argmax não é garantido (a folga não é
   monotônica no grau de diversificação).
2. Covariância Ledoit-Wolf sobre retornos diários simples.
3. Shrinkage de mu "estilo Jorion" (Bayes-Stein, Jorion 1986) com alvo na média
   da carteira de mínima variância. É heurística: a otimalidade de Jorion supõe
   covariância amostral de retornos normais iid, e aqui entra Ledoit-Wolf.
   A versão anterior usava a média simples e nunca era chamada pelo app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from core.config import DIAS_UTEIS_ANO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Momentos:
    """mu e Sigma anualizados (convenção aritmética) e diagnóstico do shrinkage."""

    mu: pd.Series
    cov: pd.DataFrame
    intensidade_shrinkage: float     # 0 = média amostral pura, 1 = alvo de Jorion
    n_observacoes: int


def retornos_simples(precos: pd.DataFrame) -> pd.DataFrame:
    """
    Retornos simples diários na amostra comum (linhas sem NaN em nenhum ativo).

    A amostra comum começa no IPO mais recente do universo. É a escolha
    conservadora: estimar a covariância par a par com janelas diferentes
    produz matrizes não-PSD.
    """
    retornos = precos.pct_change(fill_method=None)
    return retornos.dropna(how="any")


def covariancia_ledoit_wolf(retornos_diarios: pd.DataFrame) -> pd.DataFrame:
    """Covariância diária com encolhimento Ledoit-Wolf (bem condicionada mesmo com N ~ T)."""
    modelo = LedoitWolf().fit(retornos_diarios.values)
    return pd.DataFrame(modelo.covariance_, index=retornos_diarios.columns,
                        columns=retornos_diarios.columns)


def pesos_ewma(n_obs: int, meia_vida: float) -> np.ndarray:
    """Pesos exponenciais do mais antigo ao mais recente: dobram a cada ``meia_vida`` pregões; somam 1."""
    if meia_vida <= 0:
        raise ValueError(f"meia_vida deve ser > 0 pregões: {meia_vida}")
    defasagens = np.arange(n_obs - 1, -1, -1, dtype=float)
    pesos = 0.5 ** (defasagens / meia_vida)
    return pesos / pesos.sum()


def covariancia_ewma(retornos_diarios: pd.DataFrame, meia_vida: float) -> pd.DataFrame:
    """
    Covariância diária com pesos exponenciais (meia-vida em pregões) e encolhimento Ledoit-Wolf.

    Reponderação: y_t = sqrt(T w_t) (x_t - m), com m = sum_t w_t x_t. Assim y'y / T é a
    covariância ponderada, e o Ledoit-Wolf centrado a encolhe para a identidade escalada.
    É heurística: a intensidade de Ledoit-Wolf supõe observações iid de mesmo peso. Com
    pesos iguais, o resultado é a Ledoit-Wolf amostral.
    """
    valores = retornos_diarios.values
    pesos = pesos_ewma(len(valores), meia_vida)
    centrados = valores - pesos @ valores
    reponderados = np.sqrt(len(valores) * pesos)[:, None] * centrados
    modelo = LedoitWolf(assume_centered=True).fit(reponderados)
    return pd.DataFrame(modelo.covariance_, index=retornos_diarios.columns,
                        columns=retornos_diarios.columns)


def shrinkage_jorion(media_diaria: np.ndarray, cov_diaria: np.ndarray,
                     n_obs: int) -> tuple[np.ndarray, float]:
    """
    Estimador Bayes-Stein de Jorion (1986), em unidades diárias.

        m0 = 1' S^-1 m / 1' S^-1 1            (média da carteira de mínima variância)
        w  = (N+2) / [(N+2) + T (m - m0 1)' S^-1 (m - m0 1)]
        m_s = (1 - w) m + w m0 1

    Calculado em unidades diárias, e não anualizadas, para que T entre na
    fórmula sem conversão. Com insumos anualizados seria T/252 (verificado
    algebricamente).

    Escopo: Jorion deriva w com a covariância amostral escalada por
    (T-1)/(T-N-2). Aqui S é Ledoit-Wolf (plug-in), uma aproximação.

    Returns:
        (média encolhida, intensidade w em [0, 1])
    """
    n_ativos = len(media_diaria)
    if n_ativos < 3 or n_obs <= n_ativos + 2:
        return media_diaria, 0.0
    uns = np.ones(n_ativos)
    s_inv_1 = np.linalg.solve(cov_diaria, uns)
    s_inv_m = np.linalg.solve(cov_diaria, media_diaria)
    m0 = float(uns @ s_inv_m / (uns @ s_inv_1))
    desvio = media_diaria - m0
    forma_quadratica = float(desvio @ np.linalg.solve(cov_diaria, desvio))
    intensidade = (n_ativos + 2) / ((n_ativos + 2) + n_obs * max(forma_quadratica, 0.0))
    intensidade = float(np.clip(intensidade, 0.0, 1.0))
    return (1.0 - intensidade) * media_diaria + intensidade * m0, intensidade


def estimar_momentos(retornos_diarios: pd.DataFrame, shrink_mu: bool = True,
                     meia_vida_cov: float | None = None) -> Momentos:
    """
    mu e Sigma anualizados a partir de retornos simples diários. Com ``meia_vida_cov``
    (pregões), Sigma é a ``covariancia_ewma``, e o Jorion usa esse mesmo Sigma; mu continua
    a média simples da janela inteira.
    """
    if retornos_diarios.isna().any().any():
        raise ValueError("retornos_diarios contém NaN; use retornos_simples() antes.")
    n_obs, n_ativos = retornos_diarios.shape
    if n_obs < 30:
        raise ValueError(f"Amostra curta demais: T={n_obs} pregões para N={n_ativos} ativos.")

    cov_diaria = (covariancia_ledoit_wolf(retornos_diarios) if meia_vida_cov is None
                  else covariancia_ewma(retornos_diarios, meia_vida_cov))
    media_diaria = retornos_diarios.mean().values
    intensidade = 0.0
    if shrink_mu:
        media_diaria, intensidade = shrinkage_jorion(media_diaria, cov_diaria.values, n_obs)
        logger.info("Shrinkage de Jorion: intensidade=%.2f (N=%d, T=%d)", intensidade, n_ativos, n_obs)

    return Momentos(
        mu=pd.Series(media_diaria * DIAS_UTEIS_ANO, index=retornos_diarios.columns),
        cov=cov_diaria * DIAS_UTEIS_ANO,
        intensidade_shrinkage=intensidade,
        n_observacoes=n_obs,
    )


def aplicar_tilt_growth(mu: pd.Series, cov: pd.DataFrame, scores: pd.Series,
                        peso_tilt: float) -> pd.Series:
    """
    View fundamentalista aditiva no retorno esperado ("Black-Litterman lite").

        mu_i' = mu_i + peso_tilt * z_i * sigma_i,   z_i em [-1, 1]

    onde z_i é o score centrado na mediana e escalado pela amplitude. Ativos
    sem score (fundamentos indisponíveis) recebem z = 0, ou seja, nenhuma view. Aditivo para
    não inverter o sinal de mu negativo.

    Não é Black-Litterman formal: não há prior de equilíbrio nem matriz de
    confiança Omega. É um ajuste heurístico, calibrado em unidades de vol.
    """
    if peso_tilt <= 0:
        return mu
    alinhados = scores.reindex(mu.index)
    validos = alinhados.dropna()
    if validos.empty:
        return mu
    amplitude = float(validos.max() - validos.min()) or 1.0
    z = ((alinhados - validos.median()) / amplitude * 2.0).clip(-1.0, 1.0).fillna(0.0)
    sigma = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index).reindex(mu.index)
    return mu + peso_tilt * z * sigma
