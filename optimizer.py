"""
optimizer.py - Motor de Otimização de Carteiras (Convexidade Estrita)
TCC: Otimização de Carteiras de Investimentos
"""

import numpy as np
import pandas as pd
import logging
import cvxpy as cp
from typing import Optional, List, Tuple
from dataclasses import dataclass
import risk_profiles

# Configuração de logging para debug
logger = logging.getLogger(__name__)


@dataclass
class ResultadoOtimizacao:
    """Estrutura de dados para o resultado da otimização de portefólio."""
    pesos: np.ndarray
    retorno_esperado: float
    volatilidade: float
    sharpe: float
    tickers: List[str]
    sucesso: bool
    mensagem: str


def calcular_retorno_portfolio(pesos: np.ndarray, retornos_medios: np.ndarray) -> float:
    """E(Rp) = Σ(wi * E(Ri))"""
    return float(np.dot(pesos, retornos_medios))


def calcular_volatilidade_portfolio(pesos: np.ndarray, matriz_cov: np.ndarray) -> float:
    """σp = √(w' * Σ * w)"""
    return float(np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))))


def calcular_sharpe(pesos: np.ndarray, retornos_medios: np.ndarray, matriz_cov: np.ndarray, taxa_livre_risco: float) -> float:
    """Sharpe = (E(Rp) - Rf) / σp"""
    ret = calcular_retorno_portfolio(pesos, retornos_medios)
    vol = calcular_volatilidade_portfolio(pesos, matriz_cov)
    return float((ret - taxa_livre_risco) / vol) if vol > 0 else 0.0


def _selecionar_melhores_ativos(pesos: np.ndarray, n_max: int) -> List[int]:
    """
    OBSERVAÇÃO TÉCNICA (Heurística de Cardinalidade):
    A restrição estrita de cardinalidade transforma o QP em MIQP (Mixed-Integer QP),
    que é NP-Hard. Utilizamos uma heurística gulosa em duas etapas:
    1. Resolvemos o QP relaxado.
    2. Selecionamos os 'n_max' ativos com maior alocação teórica.
    3. Resolvemos um novo QP estrito apenas para este subconjunto.
    """
    return np.argsort(pesos)[::-1][:n_max].tolist()


def _resolver_problema(prob: cp.Problem, warm_start: bool = False) -> None:
    """
    Orquestrador de Solvers Institucional. 
    Tenta OSQP (nativo para Quadratic Programming), faz fallback para ECOS (SOCP) 
    e finalmente SCS. Inclui controlo restrito de tolerância para alta performance.
    """
    try:
        # OSQP é o padrão ouro da indústria para problemas de Markowitz.
        # Adicionamos tolerância estrita (1e-6) para evitar loops infinitos em matrizes singulares.
        prob.solve(
            solver=cp.OSQP, 
            warm_start=warm_start, 
            eps_abs=1e-6, 
            eps_rel=1e-6, 
            max_iter=4000
        )
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"OSQP não convergiu. Status: {prob.status}")
    except Exception as e_osqp:
        logger.debug(f"OSQP falhou ({e_osqp}), a tentar ECOS...")
        try:
            prob.solve(solver=cp.ECOS, warm_start=warm_start)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"ECOS não convergiu. Status: {prob.status}")
        except Exception as e_ecos:
            logger.debug(f"ECOS falhou ({e_ecos}), a tentar SCS como último recurso...")
            prob.solve(solver=cp.SCS, warm_start=warm_start)


def _preparar_matriz_covariancia(matriz_cov: pd.DataFrame) -> np.ndarray:
    """
    Assegura que a matriz é estritamente Positiva Semi-Definida (PSD)
    aplicando simetrização e regularização de Tikhonov (Ridge penalty).
    Isto previne falhas catastróficas nos solvers convexos devido a ruído flutuante.
    """
    n = len(matriz_cov)
    cov_matrix = np.array(matriz_cov.values, dtype=float)
    # Garante simetria perfeita
    cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
    # Adiciona um epsilon microscópico à diagonal para forçar eigenvalues positivos
    cov_matrix = cov_matrix + np.eye(n) * 1e-8
    return cov_matrix


def otimizar_min_volatilidade(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                              taxa_livre_risco: float = None, peso_maximo: float = 0.20,
                              n_ativos_max: Optional[int] = None) -> ResultadoOtimizacao:
    """Otimiza a carteira minimizando estritamente a variância."""
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
        
    n = len(retornos_medios)
    tickers = retornos_medios.index.tolist()
    ret_medio = retornos_medios.values
    cov_matrix = _preparar_matriz_covariancia(matriz_cov)

    w = cp.Variable(n)
    risco = cp.quad_form(w, cov_matrix)
    prob = cp.Problem(cp.Minimize(risco), [cp.sum(w) == 1, w >= 0, w <= peso_maximo])
    
    try:
        _resolver_problema(prob)
    except Exception as e:
        return ResultadoOtimizacao(np.zeros(n), 0, 0, 0, tickers, False, f"Falha sistémica de solvers: {e}")

    sucesso = prob.status in ["optimal", "optimal_inaccurate"]
    pesos = np.clip(w.value, 0, 1) if sucesso and w.value is not None else np.zeros(n)
    if np.sum(pesos) > 0: 
        pesos /= np.sum(pesos)
    
    # 2ª ETAPA: Heurística de Cardinalidade
    if n_ativos_max and n_ativos_max < n and sucesso:
        indices_top = _selecionar_melhores_ativos(pesos, n_ativos_max)
        n_filtrado = len(indices_top)
        cov_filtrada = cov_matrix[np.ix_(indices_top, indices_top)]
        
        w_filt = cp.Variable(n_filtrado)
        prob_filt = cp.Problem(cp.Minimize(cp.quad_form(w_filt, cov_filtrada)), 
                               [cp.sum(w_filt) == 1, w_filt >= 0, w_filt <= peso_maximo])
        try:
            _resolver_problema(prob_filt)
            sucesso_filt = prob_filt.status in ["optimal", "optimal_inaccurate"]
            
            if sucesso_filt and w_filt.value is not None:
                pesos_finais = np.zeros(n)
                pesos_limpos = np.clip(w_filt.value, 0, 1)
                if np.sum(pesos_limpos) > 0: 
                    pesos_limpos /= np.sum(pesos_limpos)
                
                for i, idx in enumerate(indices_top):
                    pesos_finais[idx] = pesos_limpos[i]
                
                return ResultadoOtimizacao(
                    pesos=pesos_finais, retorno_esperado=calcular_retorno_portfolio(pesos_finais, ret_medio),
                    volatilidade=calcular_volatilidade_portfolio(pesos_finais, cov_matrix),
                    sharpe=calcular_sharpe(pesos_finais, ret_medio, cov_matrix, taxa_livre_risco),
                    tickers=tickers, sucesso=True, mensagem="Convergência Global OSQP (2 Etapas)"
                )
        except Exception as e:
            logger.warning(f"Filtro heurístico falhou: {e}. A retornar resultado da 1ª etapa.")

    msg = "Convergência Global OSQP" if sucesso else str(prob.status)
    return ResultadoOtimizacao(
        pesos=pesos, retorno_esperado=calcular_retorno_portfolio(pesos, ret_medio),
        volatilidade=calcular_volatilidade_portfolio(pesos, cov_matrix),
        sharpe=calcular_sharpe(pesos, ret_medio, cov_matrix, taxa_livre_risco),
        tickers=tickers, sucesso=sucesso, mensagem=msg
    )


def otimizar_max_retorno(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                         taxa_livre_risco: float = None, peso_maximo: float = 0.20,
                         vol_maxima: float = 0.40, n_ativos_max: Optional[int] = None) -> ResultadoOtimizacao:
    """Maximiza retorno sujeito a um teto restrito de volatilidade."""
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
        
    n = len(retornos_medios)
    tickers = retornos_medios.index.tolist()
    ret_medio = retornos_medios.values
    cov_matrix = _preparar_matriz_covariancia(matriz_cov)

    w = cp.Variable(n)
    risco = cp.quad_form(w, cov_matrix)
    restricoes = [cp.sum(w) == 1, w >= 0, w <= peso_maximo, risco <= vol_maxima**2]
    prob = cp.Problem(cp.Maximize(ret_medio.T @ w), restricoes)
    
    try:
        _resolver_problema(prob)
    except Exception as e:
        # Fallback de viabilidade: Se a restrição de volatilidade for excessivamente baixa
        logger.warning(f"Otimização max_retorno inviável ou falhou ({e}). Fallback para min_volatilidade.")
        return otimizar_min_volatilidade(retornos_medios, matriz_cov, taxa_livre_risco, peso_maximo, n_ativos_max)

    sucesso = prob.status in ["optimal", "optimal_inaccurate"]
    pesos = np.clip(w.value, 0, 1) if sucesso and w.value is not None else np.zeros(n)
    if np.sum(pesos) > 0: 
        pesos /= np.sum(pesos)
    
    # 2ª ETAPA: Heurística de Cardinalidade
    if n_ativos_max and n_ativos_max < n and sucesso:
        indices_top = _selecionar_melhores_ativos(pesos, n_ativos_max)
        n_filtrado = len(indices_top)
        ret_filtrado = ret_medio[indices_top]
        cov_filtrada = cov_matrix[np.ix_(indices_top, indices_top)]
        
        w_filt = cp.Variable(n_filtrado)
        restricoes_filt = [
            cp.sum(w_filt) == 1, 
            w_filt >= 0, 
            w_filt <= peso_maximo, 
            cp.quad_form(w_filt, cov_filtrada) <= vol_maxima**2
        ]
        prob_filt = cp.Problem(cp.Maximize(ret_filtrado.T @ w_filt), restricoes_filt)
        
        try:
            _resolver_problema(prob_filt)
            if prob_filt.status in ["optimal", "optimal_inaccurate"] and w_filt.value is not None:
                pesos_finais = np.zeros(n)
                pesos_limpos = np.clip(w_filt.value, 0, 1)
                if np.sum(pesos_limpos) > 0: 
                    pesos_limpos /= np.sum(pesos_limpos)
                
                for i, idx in enumerate(indices_top):
                    pesos_finais[idx] = pesos_limpos[i]
                
                return ResultadoOtimizacao(
                    pesos=pesos_finais, retorno_esperado=calcular_retorno_portfolio(pesos_finais, ret_medio),
                    volatilidade=calcular_volatilidade_portfolio(pesos_finais, cov_matrix),
                    sharpe=calcular_sharpe(pesos_finais, ret_medio, cov_matrix, taxa_livre_risco),
                    tickers=tickers, sucesso=True, mensagem="Convergência Global OSQP (2 Etapas)"
                )
        except Exception:
             logger.warning("Filtro heurístico falhou na 2ª etapa. A retornar resultado da 1ª etapa.")

    msg = "Convergência Global OSQP" if sucesso else str(prob.status)
    return ResultadoOtimizacao(
        pesos=pesos, retorno_esperado=calcular_retorno_portfolio(pesos, ret_medio),
        volatilidade=calcular_volatilidade_portfolio(pesos, cov_matrix),
        sharpe=calcular_sharpe(pesos, ret_medio, cov_matrix, taxa_livre_risco),
        tickers=tickers, sucesso=sucesso, mensagem=msg
    )


def otimizar_max_sharpe(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                        taxa_livre_risco: float = None, peso_maximo: float = 0.20,
                        n_ativos_max: Optional[int] = None) -> ResultadoOtimizacao:
    """
    Maximiza o rácio de Sharpe varrendo a fronteira eficiente estritamente no espaço viável.
    A maximização direta do Sharpe é um problema fracionário; esta abordagem iterativa 
    é quantitativamente mais estável e garantidamente convexa.
    """
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
        
    n = len(retornos_medios)
    tickers = retornos_medios.index.tolist()
    ret_medio = retornos_medios.values
    cov_matrix = _preparar_matriz_covariancia(matriz_cov)
    
    def _calcular_limites_retorno(mu: np.ndarray, cov: np.ndarray, p_max: float) -> Tuple[float, float]:
        """Calcula matematicamente o limite inferior e superior de retorno possível no hiperplano."""
        w_tmp = cp.Variable(len(mu))
        try:
            p_max_ret = cp.Problem(cp.Maximize(mu.T @ w_tmp), [cp.sum(w_tmp) == 1, w_tmp >= 0, w_tmp <= p_max])
            _resolver_problema(p_max_ret)
            r_max = float(p_max_ret.value)
        except Exception:
            r_max = float(np.max(mu))

        try:
            p_min_vol = cp.Problem(cp.Minimize(cp.quad_form(w_tmp, cov)), [cp.sum(w_tmp) == 1, w_tmp >= 0, w_tmp <= p_max])
            _resolver_problema(p_min_vol)
            r_min = float(mu.T @ w_tmp.value)
        except Exception:
            r_min = float(np.min(mu))
        
        if r_max is None or r_min is None or r_min >= r_max or np.isnan(r_min) or np.isnan(r_max):
            return float(np.min(mu)), float(np.max(mu))
        return r_min, r_max

    ret_min, ret_max = _calcular_limites_retorno(ret_medio, cov_matrix, peso_maximo)
    # Define os vetores de pesquisa
    target_returns = np.linspace(ret_min, ret_max, 50)
    
    w = cp.Variable(n)
    param_retorno = cp.Parameter()
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov_matrix)), 
                      [cp.sum(w) == 1, w >= 0, w <= peso_maximo, ret_medio.T @ w >= param_retorno])
    
    best_sharpe = -np.inf
    best_pesos = np.zeros(n)
    sucesso_encontrado = False
    
    for target in target_returns:
        param_retorno.value = target
        try:
            _resolver_problema(prob, warm_start=True)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                p = np.clip(w.value, 0, 1)
                if np.sum(p) > 0: 
                    p /= np.sum(p)
                vol = np.sqrt(p.T @ cov_matrix @ p)
                ret = p.T @ ret_medio
                sharpe = (ret - taxa_livre_risco) / vol if vol > 0 else 0
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_pesos = p.copy()
                    sucesso_encontrado = True
        except Exception:
            continue

    if not sucesso_encontrado:
        logger.warning("Falha a maximizar Sharpe ao longo da fronteira. Fallback analítico para min_vol.")
        return otimizar_min_volatilidade(retornos_medios, matriz_cov, taxa_livre_risco, peso_maximo, n_ativos_max)

    # 2ª ETAPA: Heurística de Cardinalidade Dinâmica
    if n_ativos_max and n_ativos_max < n:
        indices_top = _selecionar_melhores_ativos(best_pesos, n_ativos_max)
        n_filtrado = len(indices_top)
        ret_filtrado = ret_medio[indices_top]
        cov_filtrada = cov_matrix[np.ix_(indices_top, indices_top)]
        
        # CORREÇÃO CRÍTICA: É absolutamente necessário recalcular o espaço topológico viável do novo subconjunto.
        # Se usarmos os limites de retorno globais, o solver vai rejeitar por impossibilidade matemática.
        r_min_f, r_max_f = _calcular_limites_retorno(ret_filtrado, cov_filtrada, peso_maximo)
        target_returns_filt = np.linspace(r_min_f, r_max_f, 50)
        
        w_filt = cp.Variable(n_filtrado)
        param_ret_filt = cp.Parameter()
        prob_filt = cp.Problem(cp.Minimize(cp.quad_form(w_filt, cov_filtrada)), 
                               [cp.sum(w_filt) == 1, w_filt >= 0, w_filt <= peso_maximo, ret_filtrado.T @ w_filt >= param_ret_filt])
        
        best_sharpe_filt = -np.inf
        best_pesos_filt = None
        sucesso_filt = False
        
        for target in target_returns_filt:
            param_ret_filt.value = target
            try:
                _resolver_problema(prob_filt, warm_start=True)
                if prob_filt.status in ["optimal", "optimal_inaccurate"] and w_filt.value is not None:
                    p = np.clip(w_filt.value, 0, 1)
                    if np.sum(p) > 0: 
                        p /= np.sum(p)
                    vol = np.sqrt(p.T @ cov_filtrada @ p)
                    ret = p.T @ ret_filtrado
                    sharpe = (ret - taxa_livre_risco) / vol if vol > 0 else 0
                    
                    if sharpe > best_sharpe_filt:
                        best_sharpe_filt = sharpe
                        best_pesos_filt = p.copy()
                        sucesso_filt = True
            except Exception:
                continue

        if sucesso_filt and best_pesos_filt is not None:
            pesos_finais = np.zeros(n)
            for i, idx in enumerate(indices_top):
                pesos_finais[idx] = best_pesos_filt[i]
            
            return ResultadoOtimizacao(
                pesos=pesos_finais, retorno_esperado=calcular_retorno_portfolio(pesos_finais, ret_medio),
                volatilidade=calcular_volatilidade_portfolio(pesos_finais, cov_matrix),
                sharpe=calcular_sharpe(pesos_finais, ret_medio, cov_matrix, taxa_livre_risco),
                tickers=tickers, sucesso=True, mensagem="Max Sharpe Heurístico (2 Etapas Resolvidas)"
            )
        else:
            logger.warning("Filtro heurístico falhou na varredura sub-dimensional. A retornar resultado da 1ª etapa.")
            
    return ResultadoOtimizacao(
        pesos=best_pesos, retorno_esperado=calcular_retorno_portfolio(best_pesos, ret_medio),
        volatilidade=calcular_volatilidade_portfolio(best_pesos, cov_matrix),
        sharpe=calcular_sharpe(best_pesos, ret_medio, cov_matrix, taxa_livre_risco),
        tickers=tickers, sucesso=True, mensagem="Max Sharpe Global Encontrado"
    )

def gerar_fronteira_eficiente(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                               taxa_livre_risco: float = None, n_pontos: int = 50,
                               peso_maximo: float = 0.20) -> pd.DataFrame:
    """
    Constrói a fronteira eficiente dinamicamente através de warm-start no solver convexo.
    Isto reduz exponencialmente o tempo de processamento das iterações consecutivas.
    """
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
                               
    n = len(retornos_medios)
    ret_medio = retornos_medios.values
    cov_matrix = _preparar_matriz_covariancia(matriz_cov)
    
    w = cp.Variable(n)
    
    try:
        p_min = cp.Problem(cp.Minimize(cp.quad_form(w, cov_matrix)), [cp.sum(w) == 1, w >= 0, w <= peso_maximo])
        _resolver_problema(p_min)
        ret_min = float(ret_medio.T @ w.value)
    except Exception:
        ret_min = float(np.min(ret_medio))

    try:
        p_max = cp.Problem(cp.Maximize(ret_medio.T @ w), [cp.sum(w) == 1, w >= 0, w <= peso_maximo])
        _resolver_problema(p_max)
        ret_max = float(p_max.value)
    except Exception:
        ret_max = float(np.max(ret_medio))
        
    if ret_min is None or ret_max is None or np.isnan(ret_min) or np.isnan(ret_max) or ret_min >= ret_max:
        ret_min, ret_max = float(np.min(ret_medio)), float(np.max(ret_medio))

    retornos_alvo = np.linspace(ret_min, ret_max, n_pontos)
    fronteira = []
    
    param_retorno = cp.Parameter()
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov_matrix)),
        [cp.sum(w) == 1, w >= 0, w <= peso_maximo, ret_medio.T @ w >= param_retorno]
    )
    
    for ret_alvo in retornos_alvo:
        param_retorno.value = ret_alvo
        try:
            _resolver_problema(prob, warm_start=True)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                p = np.clip(w.value, 0, 1)
                if np.sum(p) > 0: 
                    p /= np.sum(p)
                vol = np.sqrt(p.T @ cov_matrix @ p)
                ret = p.T @ ret_medio
                sharpe = (ret - taxa_livre_risco) / vol if vol > 0 else 0
                
                fronteira.append({'retorno': float(ret), 'volatilidade': float(vol), 'sharpe': float(sharpe)})
        except Exception:
            continue
            
    return pd.DataFrame(fronteira)


def otimizar_por_perfil(perfil: str, retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                        taxa_livre_risco: float = None, peso_maximo: float = 0.20,
                        n_ativos_max: Optional[int] = None) -> ResultadoOtimizacao:
    """Orquestrador base que delega a lógica dependendo do risco do perfil selecionado."""
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC

    perfil_config = risk_profiles.get_perfil(perfil)

    if perfil == "Conservador":
        return otimizar_min_volatilidade(retornos_medios, matriz_cov, taxa_livre_risco, peso_maximo, n_ativos_max)
    elif perfil == "Agressivo":
        return otimizar_max_retorno(retornos_medios, matriz_cov, taxa_livre_risco, peso_maximo, perfil_config.volatilidade_maxima, n_ativos_max)
    return otimizar_max_sharpe(retornos_medios, matriz_cov, taxa_livre_risco, peso_maximo, n_ativos_max)