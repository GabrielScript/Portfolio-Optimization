"""
optimizer.py - Motor de Otimização de Carteiras (Markowitz)
TCC: Otimização de Carteiras de Investimentos
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ResultadoOtimizacao:
    """Resultado da otimização de portfólio."""
    pesos: np.ndarray
    retorno_esperado: float
    volatilidade: float
    sharpe: float
    tickers: List[str]
    sucesso: bool
    mensagem: str


def calcular_retorno_portfolio(pesos: np.ndarray, retornos_medios: np.ndarray) -> float:
    """E(Rp) = Σ(wi * E(Ri))"""
    return np.dot(pesos, retornos_medios)


def calcular_volatilidade_portfolio(pesos: np.ndarray, matriz_cov: np.ndarray) -> float:
    """σp = √(w' * Σ * w)"""
    return np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos)))


def calcular_sharpe(pesos, retornos_medios, matriz_cov, taxa_livre_risco) -> float:
    """Sharpe = (E(Rp) - Rf) / σp"""
    ret = calcular_retorno_portfolio(pesos, retornos_medios)
    vol = calcular_volatilidade_portfolio(pesos, matriz_cov)
    return (ret - taxa_livre_risco) / vol if vol > 0 else 0


def _limitar_n_ativos(pesos: np.ndarray, n_max: int) -> np.ndarray:
    indices = np.argsort(pesos)[::-1]
    pesos_lim = np.zeros_like(pesos)
    for i in indices[:n_max]:
        pesos_lim[i] = pesos[i]
    return pesos_lim


def otimizar_max_sharpe(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                        taxa_livre_risco: float = 0.1325, peso_maximo: float = 0.20,
                        n_ativos_max: Optional[int] = None) -> ResultadoOtimizacao:
    n = len(retornos_medios)
    tickers = retornos_medios.index.tolist()
    ret_medio, cov_matrix = retornos_medios.values, matriz_cov.values
    
    def neg_sharpe(w):
        return -calcular_sharpe(w, ret_medio, cov_matrix, taxa_livre_risco)
    
    res = minimize(neg_sharpe, [1/n]*n, method='SLSQP',
                   bounds=[(0, peso_maximo)]*n,
                   constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    pesos = res.x
    if n_ativos_max and n_ativos_max < n:
        pesos = _limitar_n_ativos(pesos, n_ativos_max)
        pesos = pesos / pesos.sum()
    
    return ResultadoOtimizacao(
        pesos=pesos, retorno_esperado=calcular_retorno_portfolio(pesos, ret_medio),
        volatilidade=calcular_volatilidade_portfolio(pesos, cov_matrix),
        sharpe=calcular_sharpe(pesos, ret_medio, cov_matrix, taxa_livre_risco),
        tickers=tickers, sucesso=res.success, mensagem="OK" if res.success else res.message)


def otimizar_min_volatilidade(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                              taxa_livre_risco: float = 0.1325, peso_maximo: float = 0.20,
                              n_ativos_max: Optional[int] = None) -> ResultadoOtimizacao:
    n = len(retornos_medios)
    tickers = retornos_medios.index.tolist()
    ret_medio, cov_matrix = retornos_medios.values, matriz_cov.values
    
    def vol(w):
        return calcular_volatilidade_portfolio(w, cov_matrix)
    
    res = minimize(vol, [1/n]*n, method='SLSQP',
                   bounds=[(0, peso_maximo)]*n,
                   constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    pesos = res.x
    if n_ativos_max and n_ativos_max < n:
        pesos = _limitar_n_ativos(pesos, n_ativos_max)
        pesos = pesos / pesos.sum()
    
    return ResultadoOtimizacao(
        pesos=pesos, retorno_esperado=calcular_retorno_portfolio(pesos, ret_medio),
        volatilidade=calcular_volatilidade_portfolio(pesos, cov_matrix),
        sharpe=calcular_sharpe(pesos, ret_medio, cov_matrix, taxa_livre_risco),
        tickers=tickers, sucesso=res.success, mensagem="OK" if res.success else res.message)


def otimizar_max_retorno(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                         taxa_livre_risco: float = 0.1325, peso_maximo: float = 0.20,
                         vol_maxima: float = 0.40, n_ativos_max: Optional[int] = None) -> ResultadoOtimizacao:
    n = len(retornos_medios)
    tickers = retornos_medios.index.tolist()
    ret_medio, cov_matrix = retornos_medios.values, matriz_cov.values
    
    def neg_ret(w):
        return -calcular_retorno_portfolio(w, ret_medio)
    
    res = minimize(neg_ret, [1/n]*n, method='SLSQP',
                   bounds=[(0, peso_maximo)]*n,
                   constraints=[
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: vol_maxima - calcular_volatilidade_portfolio(x, cov_matrix)}
                   ],
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    pesos = res.x
    if n_ativos_max and n_ativos_max < n:
        pesos = _limitar_n_ativos(pesos, n_ativos_max)
        pesos = pesos / pesos.sum()
    
    return ResultadoOtimizacao(
        pesos=pesos, retorno_esperado=calcular_retorno_portfolio(pesos, ret_medio),
        volatilidade=calcular_volatilidade_portfolio(pesos, cov_matrix),
        sharpe=calcular_sharpe(pesos, ret_medio, cov_matrix, taxa_livre_risco),
        tickers=tickers, sucesso=res.success, mensagem="OK" if res.success else res.message)


def gerar_fronteira_eficiente(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                               taxa_livre_risco: float = 0.1325, n_pontos: int = 50,
                               peso_maximo: float = 0.20) -> pd.DataFrame:
    n = len(retornos_medios)
    ret_medio, cov_matrix = retornos_medios.values, matriz_cov.values
    retornos_alvo = np.linspace(ret_medio.min(), ret_medio.max(), n_pontos)
    fronteira = []
    
    for ret_alvo in retornos_alvo:
        try:
            res = minimize(lambda w: calcular_volatilidade_portfolio(w, cov_matrix),
                          [1/n]*n, method='SLSQP', bounds=[(0, peso_maximo)]*n,
                          constraints=[
                              {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                              {'type': 'eq', 'fun': lambda x, r=ret_alvo: calcular_retorno_portfolio(x, ret_medio) - r}
                          ],
                          options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                vol = res.fun
                fronteira.append({'retorno': ret_alvo, 'volatilidade': vol,
                                 'sharpe': (ret_alvo - taxa_livre_risco) / vol if vol > 0 else 0})
        except:
            continue
    return pd.DataFrame(fronteira)


def otimizar_por_perfil(perfil: str, retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                        taxa_livre_risco: float = 0.1325, peso_maximo: float = 0.20,
                        n_ativos_max: Optional[int] = None) -> ResultadoOtimizacao:
    if perfil == "Conservador":
        return otimizar_min_volatilidade(retornos_medios, matriz_cov, taxa_livre_risco, peso_maximo, n_ativos_max)
    elif perfil == "Agressivo":
        return otimizar_max_retorno(retornos_medios, matriz_cov, taxa_livre_risco, peso_maximo, 0.40, n_ativos_max)
    return otimizar_max_sharpe(retornos_medios, matriz_cov, taxa_livre_risco, peso_maximo, n_ativos_max)
