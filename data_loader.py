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
import requests
import logging
import risk_profiles
from sklearn.covariance import LedoitWolf

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

        # Preenche buracos com forward fill. NÃO usar bfill: preencheria
        # o período pré-IPO com o primeiro preço, criando retornos zero
        # falsos que subestimam a volatilidade de ativos jovens.
        precos = precos.ffill()

        return precos
        
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def baixar_cambio_usdbrl(anos: int = ANOS_HISTORICO) -> pd.Series:
    """
    Baixa a série USD/BRL via yfinance para converter ativos US em reais.

    Misturar retornos em USD e BRL na mesma covariância embute o câmbio
    como fator de risco oculto; converter tudo para BRL torna os retornos
    comparáveis na moeda do investidor.
    """
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=anos * 365 + 30)
    try:
        fx = yf.download(
            "USDBRL=X",
            start=data_inicio.strftime('%Y-%m-%d'),
            end=data_fim.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True
        )
        serie = fx['Close']
        if isinstance(serie, pd.DataFrame):
            serie = serie.iloc[:, 0]
        return serie.ffill()
    except Exception as e:
        logging.warning(f"Falha ao baixar USDBRL: {e}")
        return pd.Series(dtype=float)


def converter_para_brl(precos: pd.DataFrame, tickers_usd: List[str],
                       cambio: pd.Series) -> pd.DataFrame:
    """
    Converte colunas cotadas em USD para BRL usando a série de câmbio.

    Args:
        precos: DataFrame de preços (mistura BRL e USD)
        tickers_usd: colunas que estão em USD
        cambio: série USD/BRL diária

    Returns:
        DataFrame com todos os preços em BRL.
    """
    if cambio.empty:
        return precos
    precos = precos.copy()
    fx = cambio.reindex(precos.index).ffill().bfill()
    cols = [t for t in tickers_usd if t in precos.columns]
    precos[cols] = precos[cols].mul(fx, axis=0)
    return precos


@st.cache_data(ttl=86400)  # Cache de 24h
def baixar_cdi_historico(anos: int = ANOS_HISTORICO) -> pd.Series:
    """
    Busca a série histórica do CDI diário via API do Banco Central (SGS série 12).

    Permite calcular o excesso de retorno do portfólio dia a dia, corrigindo
    a distorção de usar uma taxa Selic constante em backtests multi-anuais.
    Referência: https://dadosabertos.bcb.gov.br/dataset/12-taxa-de-juros---cdi

    Returns:
        pd.Series com a taxa CDI diária em decimal, indexada por data.
    """
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=anos * 365 + 60)

    try:
        url = (
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?"
            f"formato=json&"
            f"dataInicial={data_inicio.strftime('%d/%m/%Y')}&"
            f"dataFinal={data_fim.strftime('%d/%m/%Y')}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        dados = resp.json()

        datas = []
        taxas = []
        for d in dados:
            datas.append(pd.to_datetime(d['data'], dayfirst=True))
            # SGS série 12 retorna CDI diário em % a.d.
            # Ex: 0.0527 significa 0.0527% ao dia → 0.000527 em decimal
            valor = float(d['valor'].replace(',', '.'))
            taxas.append(valor / 100)

        serie = pd.Series(taxas, index=pd.DatetimeIndex(datas), name='cdi_diario')
        serie = serie.sort_index()

        logging.info(
            f"CDI histórico carregado: {len(serie)} obs "
            f"({serie.index[0]:%Y-%m} a {serie.index[-1]:%Y-%m})"
        )
        return serie

    except Exception as e:
        logging.warning(f"Falha ao buscar CDI histórico via BCB: {e}")
        return pd.Series(dtype=float)

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
    - Covariância: usando encolhimento Ledoit-Wolf para robustez * 252
    
    Args:
        retornos: DataFrame com retornos logarítmicos
        
    Returns:
        Tuple com (retornos_medios_anuais, matriz_covariancia_anual)
    """
    # 252 dias úteis por ano
    retornos_medios = retornos.mean() * 252
    
    # Aplicação de Shrinkage (Ledoit-Wolf) na Matriz de Covariância
    # Corrige a sensibilidade ao erro de estimação da matriz amostral
    try:
        lw = LedoitWolf()
        cov_estimada = lw.fit(retornos).covariance_
        matriz_cov = pd.DataFrame(cov_estimada, index=retornos.columns, columns=retornos.columns)
    except Exception as e:
        import logging
        logging.warning(f"Falha ao usar Ledoit-Wolf ({e}). Usando covariância amostral.")
        matriz_cov = retornos.cov()
        
    matriz_cov = matriz_cov * 252
    
    return retornos_medios, matriz_cov

def aplicar_shrinkage_mu(retornos: pd.DataFrame, retornos_medios: pd.Series,
                         matriz_cov: pd.DataFrame) -> pd.Series:
    """
    Shrinkage de James-Stein/Jorion (1986) no vetor de retornos esperados.

    A média histórica é o pior estimador de mu em finanças: erro de estimação
    enorme, e o otimizador Max Return amplifica exatamente esse erro
    (ativos com mu inflado por sorte recebem peso máximo). O estimador
    encolhe cada mu em direção à média global do universo:

        mu_shrunk = (1 - w) * mu + w * mu_barra

    com intensidade w estimada dos próprios dados (Jorion):

        w = (N + 2) / [(N + 2) + T * (mu - mu0)' Sigma^-1 (mu - mu0)]

    Quanto mais ruidosas as médias (T pequeno, dispersão alta), maior o w.

    Args:
        retornos: DataFrame de retornos diários (para extrair T)
        retornos_medios: mu anualizado por ativo
        matriz_cov: covariância anualizada (Ledoit-Wolf, já bem condicionada)

    Returns:
        pd.Series de mu anualizado encolhido.
    """
    T, N = retornos.shape
    if N < 3 or T < 30:
        return retornos_medios

    mu = retornos_medios.values
    mu0 = float(np.mean(mu))
    d = mu - mu0

    try:
        # Sigma^-1 d via solve (evita inversão explícita)
        sigma_inv_d = np.linalg.solve(matriz_cov.values, d)
        # T em anos: mu e Sigma estão anualizados, T_anos = T/252
        quad = (T / 252.0) * float(d @ sigma_inv_d)
        quad = max(quad, 0.0)
        w = (N + 2) / ((N + 2) + quad)
    except np.linalg.LinAlgError:
        logging.warning("Shrinkage de mu: matriz singular, usando w=0.5")
        w = 0.5

    w = float(np.clip(w, 0.0, 1.0))
    mu_shrunk = (1 - w) * mu + w * mu0
    logging.info(f"Shrinkage de mu (Jorion): w={w:.2f}, alvo={mu0*100:.1f}% a.a.")
    return pd.Series(mu_shrunk, index=retornos_medios.index)


def calcular_metricas_ativo(retornos: pd.DataFrame, taxa_livre_risco: float = None) -> pd.DataFrame:
    """
    Calcula métricas individuais para cada ativo.
    
    Args:
        retornos: DataFrame com retornos logarítmicos
        taxa_livre_risco: Taxa Selic anual. Se None, usa taxa dinâmica do BCB.
        
    Returns:
        DataFrame com métricas por ativo
    """
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
        
    retornos_anuais = retornos.mean() * 252
    volatilidade_anual = retornos.std() * np.sqrt(252)
    sharpe = (retornos_anuais - taxa_livre_risco) / volatilidade_anual
    
    metricas = pd.DataFrame({
        'Retorno Anual (%)': (retornos_anuais * 100).round(2),
        'Volatilidade (%)': (volatilidade_anual * 100).round(2),
        'Sharpe': sharpe.round(3)
    })
    
    return metricas.sort_values('Sharpe', ascending=False)

def filtrar_universo_growth(dados: dict, growth_scores: pd.DataFrame,
                            percentil_minimo: float = 40.0) -> dict:
    """
    Filtra o universo de ativos pelo Growth Score (hard gate).

    Análogo ao "quintile portfolio" (Palomar §6.4.4): rankeia por score
    fundamentalista e mantém só o topo — formalmente um portfólio robusto
    a erros de estimação de mu.

    Args:
        dados: dict retornado por carregar_dados_completos
        growth_scores: DataFrame indexado por ticker com coluna 'score'
        percentil_minimo: percentil de corte (ex: 40 = mantém os 60% melhores)

    Returns:
        Novo dict de dados com universo filtrado e estatísticas recalculadas.
    """
    scores = growth_scores['score'].reindex(dados['tickers'])
    corte = np.nanpercentile(scores.dropna(), percentil_minimo)
    aprovados = scores[scores >= corte].index.tolist()

    # Garante universo mínimo viável para otimização
    if len(aprovados) < 5:
        aprovados = scores.sort_values(ascending=False).head(5).index.tolist()

    precos = dados['precos'][aprovados]
    retornos = dados['retornos'][aprovados]
    retornos_medios, matriz_cov = calcular_estatisticas(retornos)

    novo = dict(dados)
    novo.update({
        'precos': precos,
        'retornos': retornos,
        'retornos_medios': retornos_medios,
        'matriz_cov': matriz_cov,
        'metricas': calcular_metricas_ativo(retornos),
        'tickers': aprovados,
        'n_ativos': len(aprovados),
        'corte_growth': corte,
    })
    return novo


def aplicar_tilt_growth(retornos_medios: pd.Series, matriz_cov: pd.DataFrame,
                        growth_scores: pd.DataFrame,
                        peso_tilt: float = 0.30) -> pd.Series:
    """
    Aplica tilt fundamentalista no retorno esperado (estilo Black-Litterman "lite").

    mu_ajustado_i = mu_i + peso_tilt * score_norm_i * sigma_i

    onde score_norm_i in [-1, +1] (score centrado na mediana do universo) e
    sigma_i é a volatilidade anualizada do ativo. Ou seja: a "view" growth
    desloca o retorno esperado em unidades da própria vol do ativo — um ativo
    no topo do ranking com peso_tilt=0.3 ganha +0.3 sigma de retorno esperado.
    Aditivo (não multiplicativo) para não inverter o sinal de mu negativos.

    Args:
        retornos_medios: mu histórico anualizado por ativo
        matriz_cov: covariância anualizada (para extrair sigma_i)
        growth_scores: DataFrame indexado por ticker com coluna 'score'
        peso_tilt: intensidade da view growth em unidades de vol (0 = desliga)

    Returns:
        pd.Series de retornos esperados ajustados.
    """
    if peso_tilt <= 0:
        return retornos_medios

    scores = growth_scores['score'].reindex(retornos_medios.index)
    mediana = scores.median()
    amplitude = (scores.max() - scores.min()) or 1.0
    score_norm = ((scores - mediana) / amplitude * 2).clip(-1, 1).fillna(0)

    sigma = pd.Series(np.sqrt(np.diag(matriz_cov)), index=matriz_cov.index)
    sigma = sigma.reindex(retornos_medios.index).fillna(sigma.mean())

    return retornos_medios + peso_tilt * score_norm * sigma


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
