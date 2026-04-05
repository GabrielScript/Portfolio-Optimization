"""
backtesting.py - Módulo de Backtesting e Métricas de Risco
TCC: Otimização de Carteiras de Investimentos
Autor: Gabriel Estrela Lopes

Este módulo implementa:
- Backtesting com janela rolante (Walk-Forward)
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Comparação com benchmark (Ibovespa)
- Métricas de performance
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
from scipy import stats
import risk_profiles
import optimizer
import data_loader

# Configuração de logging para debug
logger = logging.getLogger(__name__)


def _alinhar_cdi(retornos_serie: pd.Series, cdi_serie: pd.Series):
    """
    Alinha temporalmente a série de retornos do portfólio com a série CDI diária.
    Trata diferenças de timezone e gaps via forward-fill.
    
    Returns:
        Tuple (retornos_alinhados, cdi_alinhado) com mesmo índice.
    """
    ret = retornos_serie.copy()
    rf = cdi_serie.copy()
    
    # Remove timezone se presente (yfinance pode retornar tz-aware)
    if hasattr(ret.index, 'tz') and ret.index.tz is not None:
        ret.index = ret.index.tz_localize(None)
    if hasattr(rf.index, 'tz') and rf.index.tz is not None:
        rf.index = rf.index.tz_localize(None)
    
    # Forward-fill CDI para cobrir eventuais gaps de feriados
    rf_alinhado = rf.reindex(ret.index, method='ffill')
    mask = rf_alinhado.notna()
    
    return ret[mask], rf_alinhado[mask]


def _calcular_sharpe_sortino_variavel(retornos_carteira, serie_cdi_diario, retorno_anualizado, volatilidade, taxa_livre_risco):
    """
    Calcula Sharpe e Sortino usando CDI diário variável quando disponível.
    Fallback para taxa constante quando CDI não disponível.
    
    Fórmula correta: SR = E[r_p - r_f] / σ(r_p - r_f) × √252
    """
    if serie_cdi_diario is not None and not serie_cdi_diario.empty:
        ret_alinhado, rf_alinhado = _alinhar_cdi(retornos_carteira, serie_cdi_diario)
        if len(ret_alinhado) > 20:
            excesso = ret_alinhado - rf_alinhado
            
            # Sharpe com excesso de retorno diário
            sharpe = (excesso.mean() / excesso.std()) * np.sqrt(252) if excesso.std() > 0 else 0
            
            # Sortino: downside deviation do excesso de retorno
            excesso_neg = excesso[excesso < 0]
            if len(excesso_neg) > 0 and excesso_neg.std() > 0:
                sortino = (excesso.mean() * 252) / (excesso_neg.std() * np.sqrt(252))
            else:
                sortino = float('inf')
            
            return sharpe, sortino
    
    # Fallback: taxa constante (comportamento original)
    sharpe = (retorno_anualizado - taxa_livre_risco) / volatilidade if volatilidade > 0 else 0
    sortino = calcular_sortino(retornos_carteira.values, taxa_livre_risco)
    return sharpe, sortino


def calcular_var(retornos: np.ndarray, confianca: float = 0.95) -> float:
    """
    Calcula o Value at Risk (VaR) paramétrico.
    
    VaR representa a perda máxima esperada em um dado nível de confiança.
    
    Args:
        retornos: Array de retornos diários
        confianca: Nível de confiança (default 95%)
        
    Returns:
        VaR diário como valor positivo (perda)
    """
    # VaR paramétrico assumindo distribuição normal
    media = np.mean(retornos)
    desvio = np.std(retornos)
    
    # Percentil correspondente ao nível de confiança
    z_score = stats.norm.ppf(1 - confianca)
    
    var = -(media + z_score * desvio)
    return var


def calcular_var_historico(retornos: np.ndarray, confianca: float = 0.95) -> float:
    """
    Calcula o VaR histórico (não-paramétrico).
    
    Usa a distribuição real dos retornos, sem assumir normalidade.
    
    Args:
        retornos: Array de retornos diários
        confianca: Nível de confiança
        
    Returns:
        VaR histórico diário
    """
    percentil = (1 - confianca) * 100
    var = -np.percentile(retornos, percentil)
    return var


def calcular_var_cornish_fisher(retornos: np.ndarray, confianca: float = 0.95) -> float:
    """
    Calcula o Value at Risk (VaR) usando a expansão de Cornish-Fisher.
    
    Ajusta o z-score da distribuição normal considerando a assimetria (skewness)
    e a curtose (kurtosis) empíricas da série de retornos.
    Ideal para mercados financeiros, que frequentemente apresentam 'fat tails'.
    
    Args:
        retornos: Array de retornos diários
        confianca: Nível de confiança
        
    Returns:
        VaR diário ajustado (valor positivo)
    """
    media = np.mean(retornos)
    desvio = np.std(retornos)
    z = stats.norm.ppf(1 - confianca)
    
    # Assimetria e Curtose (excesso)
    s = stats.skew(retornos)
    k = stats.kurtosis(retornos)
    
    # Ajuste de Cornish-Fisher no z-score
    z_cf = (z + 
            (z**2 - 1) * s / 6 + 
            (z**3 - 3*z) * k / 24 - 
            (2*z**3 - 5*z) * (s**2) / 36)
            
    var_cf = -(media + z_cf * desvio)
    return var_cf


def calcular_cvar(retornos: np.ndarray, confianca: float = 0.95) -> float:
    """
    Calcula o Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR é a perda média esperada nos cenários que excedem o VaR.
    É uma métrica mais conservadora que o VaR.
    
    Args:
        retornos: Array de retornos diários
        confianca: Nível de confiança
        
    Returns:
        CVaR diário como valor positivo
    """
    var = calcular_var_historico(retornos, confianca)
    # Média dos retornos que são piores que o VaR
    retornos_abaixo_var = retornos[retornos < -var]
    
    if len(retornos_abaixo_var) == 0:
        return var
    
    cvar = -np.mean(retornos_abaixo_var)
    return cvar


def calcular_drawdown(precos: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Calcula o drawdown (queda do pico) ao longo do tempo.
    
    Args:
        precos: Série de preços ou valores da carteira
        
    Returns:
        Tuple: (série de drawdowns, drawdown máximo, duração máxima em dias)
    """
    # Pico acumulado (máximo até cada ponto)
    pico = precos.expanding().max()
    
    # Drawdown em cada ponto
    drawdown = (precos - pico) / pico
    
    # Drawdown máximo
    max_drawdown = drawdown.min()
    
    # Duração do maior drawdown
    is_drawdown = drawdown < 0
    duracao_max = 0
    duracao_atual = 0
    
    for dd in is_drawdown:
        if dd:
            duracao_atual += 1
            duracao_max = max(duracao_max, duracao_atual)
        else:
            duracao_atual = 0
    
    return drawdown, max_drawdown, duracao_max


def calcular_sortino(retornos: np.ndarray, taxa_livre_risco: float = None) -> float:
    """
    Calcula o Índice de Sortino.
    
    Similar ao Sharpe, mas considera apenas volatilidade negativa (downside).
    Mais adequado para avaliar risco de perda.
    
    Args:
        retornos: Array de retornos diários
        taxa_livre_risco: Taxa Selic anual
        
    Returns:
        Índice de Sortino anualizado
    """
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
        
    retorno_anual = np.mean(retornos) * 252
    
    # Apenas retornos negativos para downside deviation
    retornos_negativos = retornos[retornos < 0]
    
    if len(retornos_negativos) == 0:
        return np.inf
    
    downside_deviation = np.std(retornos_negativos) * np.sqrt(252)
    
    if downside_deviation == 0:
        return np.inf
    
    sortino = (retorno_anual - taxa_livre_risco) / downside_deviation
    return sortino


def backtesting_pesos_fixos(
    precos: pd.DataFrame,
    pesos: Dict[str, float],
    janela_rebalanceamento: int = 63,  # Trimestral (~63 dias úteis)
    capital_inicial: float = 100000,
    taxa_livre_risco: float = None,
    serie_cdi_diario: pd.Series = None
) -> Dict:
    """
    Realiza backtesting com rebalanceamento periódico usando pesos fixos.
    
    Args:
        precos: DataFrame com preços históricos
        pesos: Dicionário {ticker: peso}
        janela_rebalanceamento: Dias entre rebalanceamentos
        capital_inicial: Valor inicial investido
        serie_cdi_diario: Série CDI diária para Sharpe/Sortino variável no tempo
        
    Returns:
        Dict com resultados do backtesting
    """
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
        
    # Filtra apenas tickers que existem nos preços
    tickers_validos = [t for t in pesos.keys() if t in precos.columns]
    pesos_validos = {t: pesos[t] for t in tickers_validos}
    
    # Normaliza pesos
    soma_pesos = sum(pesos_validos.values())
    pesos_norm = {t: p / soma_pesos for t, p in pesos_validos.items()}
    
    # Calcula retornos diários
    retornos = precos[tickers_validos].pct_change().dropna()
    
    # Simula evolução da carteira
    valor_carteira = [capital_inicial]
    datas = [retornos.index[0]]
    
    valor_atual = capital_inicial
    dias_desde_rebalanceamento = 0
    pesos_atuais = pesos_norm.copy()
    
    for i, (data, ret_dia) in enumerate(retornos.iterrows()):
        # Calcula retorno ponderado do dia
        retorno_carteira = sum(
            pesos_atuais.get(t, 0) * ret_dia.get(t, 0) 
            for t in tickers_validos
        )
        
        valor_atual *= (1 + retorno_carteira)
        valor_carteira.append(valor_atual)
        datas.append(data)
        
        # Derivação do peso (drift)
        denom = 1 + retorno_carteira
        if denom > 0:
            pesos_atuais = {t: pesos_atuais.get(t, 0) * (1 + ret_dia.get(t, 0)) / denom for t in tickers_validos}
            
        dias_desde_rebalanceamento += 1
        
        # Rebalanceia periodicamente
        if dias_desde_rebalanceamento >= janela_rebalanceamento:
            pesos_atuais = pesos_norm.copy()
            dias_desde_rebalanceamento = 0
    
    # Cria série temporal
    serie_carteira = pd.Series(valor_carteira, index=datas)
    retornos_carteira = serie_carteira.pct_change().dropna()
    
    # Calcula métricas
    retorno_total = (serie_carteira.iloc[-1] / serie_carteira.iloc[0]) - 1
    retorno_anualizado = (1 + retorno_total) ** (252 / len(retornos_carteira)) - 1
    volatilidade = retornos_carteira.std() * np.sqrt(252)
    
    drawdown_serie, max_drawdown, duracao_dd = calcular_drawdown(serie_carteira)
    
    var_95 = calcular_var_cornish_fisher(retornos_carteira.values, 0.95)
    cvar_95 = calcular_cvar(retornos_carteira.values, 0.95)
    
    # Sharpe e Sortino com CDI diário variável (correção metodológica)
    sharpe, sortino = _calcular_sharpe_sortino_variavel(
        retornos_carteira, serie_cdi_diario,
        retorno_anualizado, volatilidade, taxa_livre_risco
    )
    
    return {
        'serie_carteira': serie_carteira,
        'retornos_carteira': retornos_carteira,
        'drawdown_serie': drawdown_serie,
        'capital_inicial': capital_inicial,
        'capital_final': serie_carteira.iloc[-1],
        'retorno_total': retorno_total,
        'retorno_anualizado': retorno_anualizado,
        'volatilidade': volatilidade,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'duracao_max_drawdown': duracao_dd,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'var_95_anual': var_95 * np.sqrt(252),
        'cvar_95_anual': cvar_95 * np.sqrt(252),
        'n_dias': len(retornos_carteira),
        'periodo_inicio': serie_carteira.index[0],
        'periodo_fim': serie_carteira.index[-1]
    }


def backtesting_walk_forward(
    precos: pd.DataFrame,
    perfil: str,
    janela_treino: int = 252 * 2,  # 2 anos de treino
    janela_teste: int = 63,        # 1 trimestre out-of-sample
    capital_inicial: float = 100000,
    taxa_livre_risco: float = None,
    n_ativos_max: Optional[int] = None,
    peso_maximo: float = 0.20,
    serie_cdi_diario: pd.Series = None
) -> Dict:
    """
    Realiza o VERDADEIRO backtesting Walk-Forward (Out-of-Sample).
    
    Metodologia:
    1. Otimiza a carteira em T usando janela_treino [T - janela_treino, T]
    2. Aplica esses pesos para os próximos dias (janela_teste) simulando a vida real
    3. Avança o tempo em janela_teste dias (T -> T + janela_teste)
    4. Repete o processo até o fim dos dados
    """
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC

    # Retornos simples para simulação de capital (compounding correto)
    retornos_simples = precos.pct_change().dropna()
    # Retornos logarítmicos para otimização (consistente com o dashboard principal)
    retornos_log = np.log(precos / precos.shift(1)).dropna()

    total_dias = len(retornos_simples)

    if total_dias <= janela_treino + janela_teste:
        raise ValueError("Dados insuficientes para Walk-Forward com estas janelas.")

    valor_carteira = [capital_inicial]
    datas = []
    valor_atual = capital_inicial

    # Prepara o primeiro dia para alinhar array
    datas.append(retornos_simples.index[janela_treino - 1])

    inicio_teste = janela_treino

    while inicio_teste < total_dias:
        fim_teste = min(inicio_teste + janela_teste, total_dias)

        # 1. Isola os dados de "treino" in-sample (log returns para otimização)
        retornos_treino = retornos_log.iloc[inicio_teste - janela_treino : inicio_teste]

        # 2. Otimiza apenas olhando pro passado para não ter look-ahead bias
        ret_medios, cov_matrix = data_loader.calcular_estatisticas(retornos_treino)
        
        try:
            res_opt = optimizer.otimizar_por_perfil(
                perfil=perfil,
                retornos_medios=ret_medios,
                matriz_cov=cov_matrix,
                taxa_livre_risco=taxa_livre_risco,
                peso_maximo=peso_maximo,
                n_ativos_max=n_ativos_max
            )
            if not res_opt.sucesso:
                pesos_opt = np.ones(len(precos.columns)) / len(precos.columns)
            else:
                pesos_opt = res_opt.pesos
        except Exception as e:
            logger.error(f"Erro no WF opt: {e}")
            pesos_opt = np.ones(len(precos.columns)) / len(precos.columns)
            
        pesos_dict = dict(zip(precos.columns, pesos_opt))
        
        # 3. Executa do dia inicio_teste até fim_teste usando pesos_dict (Out-Of-Sample)
        # Usa retornos SIMPLES para simulação de capital (compounding correto)
        retornos_teste = retornos_simples.iloc[inicio_teste : fim_teste]
        pesos_atuais = pesos_dict.copy()
        
        for data, ret_dia in retornos_teste.iterrows():
            # Retorno diário da carteira Out-of-Sample
            retorno_carteira = sum(pesos_atuais.get(t, 0) * ret_dia.get(t, 0) for t in precos.columns)
            valor_atual *= (1 + retorno_carteira)
            
            valor_carteira.append(valor_atual)
            datas.append(data)
            
            # Evolução orgânica (drift) dos pesos
            denom = 1 + retorno_carteira
            if denom > 0:
                pesos_atuais = {t: pesos_atuais.get(t, 0) * (1 + ret_dia.get(t, 0)) / denom for t in precos.columns}
        
        # Avança a janela
        inicio_teste = fim_teste
        
    serie_carteira = pd.Series(valor_carteira, index=datas)
    # Remove duplicidade de data inicial caso exista
    serie_carteira = serie_carteira.groupby(serie_carteira.index).first()
    retornos_carteira = serie_carteira.pct_change().dropna()

    retorno_total = (serie_carteira.iloc[-1] / serie_carteira.iloc[0]) - 1
    n_dias_bt = len(retornos_carteira)
    retorno_anualizado = (1 + retorno_total) ** (252 / max(n_dias_bt, 1)) - 1
    volatilidade = retornos_carteira.std() * np.sqrt(252)
    drawdown_serie, max_drawdown, duracao_dd = calcular_drawdown(serie_carteira)

    var_95 = calcular_var_cornish_fisher(retornos_carteira.values, 0.95)
    cvar_95 = calcular_cvar(retornos_carteira.values, 0.95)
    
    # Sharpe e Sortino com CDI diário variável (correção metodológica)
    sharpe, sortino = _calcular_sharpe_sortino_variavel(
        retornos_carteira, serie_cdi_diario,
        retorno_anualizado, volatilidade, taxa_livre_risco
    )

    return {
        'serie_carteira': serie_carteira,
        'retornos_carteira': retornos_carteira,
        'drawdown_serie': drawdown_serie,
        'capital_inicial': capital_inicial,
        'capital_final': serie_carteira.iloc[-1],
        'retorno_total': retorno_total,
        'retorno_anualizado': retorno_anualizado,
        'volatilidade': volatilidade,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'duracao_max_drawdown': duracao_dd,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'var_95_anual': var_95 * np.sqrt(252),
        'cvar_95_anual': cvar_95 * np.sqrt(252),
        'n_dias': n_dias_bt,
        'periodo_inicio': serie_carteira.index[0],
        'periodo_fim': serie_carteira.index[-1]
    }


def comparar_com_benchmark(
    serie_carteira: pd.Series,
    precos_benchmark: pd.Series,
    nome_benchmark: str = "IBOV",
    taxa_livre_risco: float = None
) -> Dict:
    """
    Compara performance da carteira com um benchmark.
    
    Args:
        serie_carteira: Série de valores da carteira
        precos_benchmark: Série de preços do benchmark
        nome_benchmark: Nome do benchmark
        
    Returns:
        Dict com métricas comparativas
    """
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
        
    # Alinha datas
    datas_comuns = serie_carteira.index.intersection(precos_benchmark.index)
    
    if len(datas_comuns) < 10:
        return {'erro': 'Dados insuficientes para comparação'}
    
    carteira = serie_carteira.loc[datas_comuns]
    benchmark = precos_benchmark.loc[datas_comuns]
    
    # Normaliza para base 100
    carteira_norm = carteira / carteira.iloc[0] * 100
    benchmark_norm = benchmark / benchmark.iloc[0] * 100
    
    # Retornos
    ret_carteira = carteira.pct_change().dropna()
    ret_benchmark = benchmark.pct_change().dropna()
    
    # Métricas do benchmark
    ret_total_benchmark = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
    ret_anual_benchmark = (1 + ret_total_benchmark) ** (252 / len(ret_benchmark)) - 1
    vol_benchmark = ret_benchmark.std() * np.sqrt(252)
    
    # Métricas da carteira
    ret_total_carteira = (carteira.iloc[-1] / carteira.iloc[0]) - 1
    ret_anual_carteira = (1 + ret_total_carteira) ** (252 / len(ret_carteira)) - 1
    
    # Alpha e Beta (CAPM)
    cov_matrix = np.cov(ret_carteira.values, ret_benchmark.values)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1
    alpha = ret_anual_carteira - (taxa_livre_risco + beta * (ret_anual_benchmark - taxa_livre_risco))
    
    # Correlação
    correlacao = ret_carteira.corr(ret_benchmark)
    
    # Tracking Error
    tracking_error = (ret_carteira - ret_benchmark).std() * np.sqrt(252)
    
    # Information Ratio
    excesso_retorno = ret_anual_carteira - ret_anual_benchmark
    information_ratio = excesso_retorno / tracking_error if tracking_error > 0 else 0
    
    return {
        'carteira_normalizada': carteira_norm,
        'benchmark_normalizado': benchmark_norm,
        'nome_benchmark': nome_benchmark,
        'retorno_carteira': ret_anual_carteira,
        'retorno_benchmark': ret_anual_benchmark,
        'volatilidade_benchmark': vol_benchmark,
        'alpha': alpha,
        'beta': beta,
        'correlacao': correlacao,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'excesso_retorno': excesso_retorno
    }


def calcular_metricas_risco_portfolio(
    retornos: pd.DataFrame,
    pesos: Dict[str, float],
    taxa_livre_risco: float = None
) -> Dict:
    """
    Calcula todas as métricas de risco para um portfólio.
    
    Args:
        retornos: DataFrame de retornos dos ativos
        pesos: Dicionário de pesos
        taxa_livre_risco: Taxa Selic
        
    Returns:
        Dict com todas as métricas de risco
    """
    if taxa_livre_risco is None:
        taxa_livre_risco = risk_profiles.TAXA_SELIC
        
    # Filtra tickers válidos
    tickers = [t for t in pesos.keys() if t in retornos.columns]
    pesos_arr = np.array([pesos[t] for t in tickers])
    pesos_arr = pesos_arr / pesos_arr.sum()  # Normaliza
    
    # Retornos do portfólio (converte log returns para simples antes de ponderar)
    retornos_simples = np.exp(retornos[tickers]) - 1
    ret_portfolio = (retornos_simples * pesos_arr).sum(axis=1)
    
    # Métricas básicas
    ret_anual = ret_portfolio.mean() * 252
    vol_anual = ret_portfolio.std() * np.sqrt(252)
    sharpe = (ret_anual - taxa_livre_risco) / vol_anual if vol_anual > 0 else 0
    
    # Métricas de risco
    var_95 = calcular_var_cornish_fisher(ret_portfolio.values, 0.95)
    var_99 = calcular_var_cornish_fisher(ret_portfolio.values, 0.99)
    cvar_95 = calcular_cvar(ret_portfolio.values, 0.95)
    sortino = calcular_sortino(ret_portfolio.values, taxa_livre_risco)
    
    # Curtose e assimetria (indica caudas pesadas)
    curtose = stats.kurtosis(ret_portfolio.values)
    assimetria = stats.skew(ret_portfolio.values)
    
    return {
        'retorno_anual': ret_anual,
        'volatilidade_anual': vol_anual,
        'sharpe': sharpe,
        'sortino': sortino,
        'var_95_diario': var_95,
        'var_99_diario': var_99,
        'var_95_anual': var_95 * np.sqrt(252),
        'cvar_95_diario': cvar_95,
        'cvar_95_anual': cvar_95 * np.sqrt(252),
        'curtose': curtose,
        'assimetria': assimetria,
        'interpretacao_var': f"Com 95% de confiança, a perda diária não excederá {var_95*100:.2f}% (Cornish-Fisher)",
        'interpretacao_cvar': f"Nos piores 5% dos dias, a perda média será de {cvar_95*100:.2f}%"
    }
