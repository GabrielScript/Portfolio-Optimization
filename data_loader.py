"""
data_loader.py - ETL de dados históricos da B3 via yfinance
TCC: Otimização de Carteiras de Investimentos
Autor: Gabriel

Este módulo é responsável por:
- Baixar dados históricos de preços
- Calcular retornos logarítmicos
- Calcular matriz de covariância
- Calcular estatísticas descritivas
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import streamlit as st

# Período de análise: 5 anos
ANOS_HISTORICO = 5

@st.cache_data(ttl=3600)  # Cache por 1 hora
def baixar_dados_historicos(tickers: List[str], anos: int = ANOS_HISTORICO) -> pd.DataFrame:
    """
    Baixa dados históricos de preços ajustados via yfinance.
    
    Args:
        tickers: Lista de tickers (ex: ['PETR4.SA', 'VALE3.SA'])
        anos: Quantidade de anos de histórico
        
    Returns:
        DataFrame com preços ajustados de fechamento
    """
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=anos * 365)
    
    try:
        dados = yf.download(
            tickers,
            start=data_inicio.strftime('%Y-%m-%d'),
            end=data_fim.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True
        )
        
        # Se só tem um ticker, yfinance retorna diferente
        if len(tickers) == 1:
            precos = dados[['Close']].copy()
            precos.columns = tickers
        else:
            precos = dados['Close'].copy()
        
        # Remove colunas com muitos NaN (ativos com dados insuficientes)
        precos = precos.dropna(axis=1, thresh=int(len(precos) * 0.8))
        
        # Preenche NaN restantes com forward fill
        precos = precos.ffill().bfill()
        
        return precos
        
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

def calcular_retornos(precos: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos logarítmicos diários.
    
    Fórmula: r_t = ln(P_t / P_{t-1})
    
    Args:
        precos: DataFrame com preços de fechamento
        
    Returns:
        DataFrame com retornos logarítmicos
    """
    retornos = np.log(precos / precos.shift(1))
    return retornos.dropna()

def calcular_estatisticas(retornos: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calcula retorno médio anualizado e matriz de covariância.
    
    Baseado na Teoria de Markowitz:
    - Retorno esperado: média dos retornos * 252 (dias úteis)
    - Covariância: cov(r_i, r_j) * 252
    
    Args:
        retornos: DataFrame com retornos logarítmicos
        
    Returns:
        Tuple com (retornos_medios_anuais, matriz_covariancia_anual)
    """
    # 252 dias úteis por ano
    retornos_medios = retornos.mean() * 252
    matriz_cov = retornos.cov() * 252
    
    return retornos_medios, matriz_cov

def calcular_metricas_ativo(retornos: pd.DataFrame, taxa_livre_risco: float = 0.1325) -> pd.DataFrame:
    """
    Calcula métricas individuais para cada ativo.
    
    Args:
        retornos: DataFrame com retornos logarítmicos
        taxa_livre_risco: Taxa Selic anual
        
    Returns:
        DataFrame com métricas por ativo
    """
    retornos_anuais = retornos.mean() * 252
    volatilidade_anual = retornos.std() * np.sqrt(252)
    sharpe = (retornos_anuais - taxa_livre_risco) / volatilidade_anual
    
    metricas = pd.DataFrame({
        'Retorno Anual (%)': (retornos_anuais * 100).round(2),
        'Volatilidade (%)': (volatilidade_anual * 100).round(2),
        'Sharpe': sharpe.round(3)
    })
    
    return metricas.sort_values('Sharpe', ascending=False)

def carregar_dados_completos(tickers: List[str], anos: int = ANOS_HISTORICO) -> Optional[dict]:
    """
    Pipeline completo de carregamento e processamento de dados.
    
    Args:
        tickers: Lista de tickers para análise
        anos: Quantidade de anos de histórico
        
    Returns:
        Dict com todos os dados processados ou None se erro
    """
    with st.spinner('📊 Baixando dados históricos...'):
        precos = baixar_dados_historicos(tickers, anos)
        
    if precos.empty:
        return None
    
    # Atualiza lista de tickers válidos (alguns podem ter sido removidos)
    tickers_validos = precos.columns.tolist()
    
    with st.spinner('📈 Calculando retornos e estatísticas...'):
        retornos = calcular_retornos(precos)
        retornos_medios, matriz_cov = calcular_estatisticas(retornos)
        metricas = calcular_metricas_ativo(retornos)
    
    # Cálculo de período em diferentes unidades
    n_dias = len(retornos)
    n_meses = round(n_dias / 21)  # ~21 dias úteis por mês
    n_anos = round(n_dias / 252, 1)  # 252 dias úteis por ano
    
    return {
        'precos': precos,
        'retornos': retornos,
        'retornos_medios': retornos_medios,
        'matriz_cov': matriz_cov,
        'metricas': metricas,
        'tickers': tickers_validos,
        'n_ativos': len(tickers_validos),
        'periodo_inicio': precos.index[0].strftime('%m/%Y'),
        'periodo_fim': precos.index[-1].strftime('%m/%Y'),
        'n_observacoes': n_dias,
        'n_meses': n_meses,
        'n_anos': n_anos
    }
