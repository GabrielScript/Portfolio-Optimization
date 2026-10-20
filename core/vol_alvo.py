"""
vol_alvo.py - Alvo de volatilidade com EGARCH (DECISOES 55, item depois do S2): a exposição de uma curva fora da
amostra ao próprio risco é cortada quando a vol prevista passa da vol recente, e o resto fica no CDI.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import StudentsT

TETO = 1.0              # sem alavancagem: exposição de 0 a 100%
JANELA = 504            # retornos diários da curva no ajuste (2 anos)
PASSO = 21              # pregões entre reajustes da exposição
CUSTO_BPS = 10.0        # por lado, sobre a fração da curva que muda
SIMULACOES = 1_000      # caminhos da previsão de vários pregões (o EGARCH não tem fórmula fechada)
SEMENTE = 20_261_020
RAZAO_MAXIMA = 10.0     # previsão mais de 10x longe da vol realizada da janela = ajuste degenerado

PrevisaoDeVol = Callable[[np.ndarray], "float | None"]     # retornos da janela -> vol anual prevista (None: falhou)


def exposicao(prevista: float, alvo: float, teto: float = TETO) -> float:
    """Fração da curva exposta: ``alvo / prevista`` (vols anuais), no máximo ``teto``."""
    return min(teto, alvo / prevista)


@dataclass(frozen=True)
class CurvaComAlvo:
    valores: pd.Series          # do fechamento anterior ao 1º reajuste ao fim da curva original
    exposicoes: pd.Series       # exposição escolhida em cada reajuste (índice: o pregão da decisão)
    giro: pd.Series             # |exposição nova - a que derivou| em cada reajuste
    falhas: list[str] = field(default_factory=list)


def curva_com_alvo(valores: pd.Series, rf_diaria: pd.Series, prever: PrevisaoDeVol, janela: int = JANELA,
                   passo: int = PASSO, custo_bps: float = CUSTO_BPS) -> CurvaComAlvo:
    """
    A curva ``valores`` com alvo de vol: no fechamento de cada ``passo``-ésimo pregão, a partir do que completa
    ``janela`` retornos, ``prever`` recebe só os ``janela`` retornos até ali e a exposição vira ``exposicao(prevista,
    alvo)``, com alvo = desvio-padrão desses retornos x raiz(252); entre os reajustes, a parte exposta rende a curva e o
    resto, ``rf_diaria`` (cada parte deriva). Custo de ``custo_bps`` sobre |exposição nova - a que derivou|; antes do
    1º reajuste a exposição é 1 (a curva começa investida). ``prever`` devolve None (não convergiu) ou uma vol a mais
    de ``RAZAO_MAXIMA`` vezes do alvo, para cima ou para baixo (ajuste degenerado): a exposição derivada fica, sem
    negócio, e a data entra em ``falhas``.
    """
    retornos = valores.pct_change().iloc[1:]
    rf = rf_diaria.reindex(retornos.index).fillna(0.0)
    inicio = retornos.index[janela - 1]
    exposto, caixa = float(valores.loc[inicio]), 0.0
    datas, curva, exposicoes, giros, falhas = [inicio], [exposto], {}, {}, []
    for decisao in range(janela, len(retornos), passo):
        data = retornos.index[decisao - 1]
        historico = retornos.iloc[decisao - janela:decisao].to_numpy()
        valor = exposto + caixa
        derivada = exposto / valor
        prevista, alvo = prever(historico), float(np.std(historico, ddof=1) * np.sqrt(252))
        if prevista is None or not alvo / RAZAO_MAXIMA <= prevista <= alvo * RAZAO_MAXIMA:
            falhas.append(f"{data:%Y-%m-%d}: previsão de vol {prevista} sem convergência ou degenerada (alvo "
                          f"{alvo:.3f}); exposição mantida em {derivada:.3f}")
            nova = derivada
        else:
            nova = exposicao(prevista, alvo)
        giros[data], exposicoes[data] = abs(nova - derivada), nova
        valor -= valor * giros[data] * custo_bps / 1e4
        exposto, caixa = valor * nova, valor * (1.0 - nova)
        for dia in range(decisao, min(decisao + passo, len(retornos))):
            exposto *= 1.0 + retornos.iloc[dia]
            caixa *= 1.0 + rf.iloc[dia]
            datas.append(retornos.index[dia])
            curva.append(exposto + caixa)
    return CurvaComAlvo(pd.Series(curva, index=pd.DatetimeIndex(datas)), pd.Series(exposicoes), pd.Series(giros),
                        falhas)


def vol_anual_da_previsao(variancias: np.ndarray) -> float | None:
    """Vol anual (fração) da média das variâncias diárias previstas em %²; None se a previsão é 0, infinita ou NaN
    (o ajuste "convergiu" para um ponto degenerado, comum em ruído sem agrupamento de vol)."""
    vol = float(np.sqrt(np.mean(variancias) * 252) / 100.0)
    return vol if np.isfinite(vol) and vol > 0.0 else None


def vol_egarch(retornos: np.ndarray, horizonte: int = PASSO, simulacoes: int = SIMULACOES,
               semente: int = SEMENTE) -> float | None:
    """
    Vol anual média prevista para os próximos ``horizonte`` pregões por um EGARCH(1,1) de Nelson com o termo de
    assimetria (queda pesa diferente de alta) e erros t de Student, média constante, ajustado por máxima
    verossimilhança nos ``retornos`` diários x 100; a previsão sai de ``simulacoes`` caminhos com ``semente`` fixa (a
    mesma janela dá a mesma vol). None quando o otimizador não converge ou a previsão é degenerada.
    """
    modelo = arch_model(100.0 * retornos, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
    modelo.distribution = StudentsT(seed=np.random.default_rng(semente))    # os caminhos saem do gerador da t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # passos do otimizador por regiões sem verossimilhança
        ajuste = modelo.fit(disp="off", show_warning=False)
        if ajuste.convergence_flag != 0:
            return None
        previsao = ajuste.forecast(horizon=horizonte, method="simulation", simulations=simulacoes, reindex=False)
    return vol_anual_da_previsao(previsao.variance.to_numpy()[-1])
