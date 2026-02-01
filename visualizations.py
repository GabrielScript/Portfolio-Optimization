"""
visualizations.py - Gráficos Plotly para o Dashboard
TCC: Otimização de Carteiras de Investimentos
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict

# Cores do tema escuro
CORES = {
    'fundo': '#0E1117',
    'card': '#1E2130',
    'texto': '#FAFAFA',
    'grid': '#2D3250',
    'conservador': '#00D4AA',
    'moderado': '#FFB74D',
    'agressivo': '#FF5252',
    'azul': '#667EEA',
    'roxo': '#764BA2'
}


def grafico_fronteira_eficiente(fronteira: pd.DataFrame, carteira_otima: dict, 
                                 perfil: str) -> go.Figure:
    """Gráfico da Fronteira Eficiente com carteira ótima destacada."""
    
    cor_perfil = CORES.get(perfil.lower(), CORES['moderado'])
    
    fig = go.Figure()
    
    # Fronteira eficiente
    fig.add_trace(go.Scatter(
        x=fronteira['volatilidade'] * 100,
        y=fronteira['retorno'] * 100,
        mode='lines',
        name='Fronteira Eficiente',
        line=dict(color=CORES['azul'], width=3),
        hovertemplate='Volatilidade: %{x:.1f}%<br>Retorno: %{y:.1f}%<extra></extra>'
    ))
    
    # Ponto ótimo
    fig.add_trace(go.Scatter(
        x=[carteira_otima['volatilidade'] * 100],
        y=[carteira_otima['retorno'] * 100],
        mode='markers',
        name=f'Carteira {perfil}',
        marker=dict(size=20, color=cor_perfil, symbol='star',
                   line=dict(width=2, color='white')),
        hovertemplate=f'<b>Carteira Ótima ({perfil})</b><br>' +
                      'Volatilidade: %{x:.2f}%<br>Retorno: %{y:.2f}%<extra></extra>'
    ))
    
    # Taxa Selic
    fig.add_hline(y=13.25, line_dash="dash", line_color="#888",
                  annotation_text="Selic 13.25%", annotation_position="right")
    
    fig.update_layout(
        title=dict(text='📈 Fronteira Eficiente de Markowitz', font=dict(size=20, color=CORES['texto'])),
        xaxis_title='Volatilidade (% a.a.)',
        yaxis_title='Retorno Esperado (% a.a.)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        height=500
    )
    
    return fig


def grafico_composicao_pizza(pesos: dict, orcamento: float) -> go.Figure:
    """Gráfico de pizza com composição da carteira."""
    
    # Filtra pesos > 1%
    pesos_filtrados = {k: v for k, v in pesos.items() if v > 0.01}
    
    labels = [t.replace('.SA', '') for t in pesos_filtrados.keys()]
    valores = list(pesos_filtrados.values())
    valores_reais = [v * orcamento for v in valores]
    
    cores = px.colors.qualitative.Set3[:len(labels)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=valores,
        hole=0.5,
        marker=dict(colors=cores, line=dict(color=CORES['fundo'], width=2)),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Peso: %{percent}<br>Valor: R$ %{customdata:,.2f}<extra></extra>',
        customdata=valores_reais
    )])
    
    fig.update_layout(
        title=dict(text='🥧 Composição da Carteira', font=dict(size=20, color=CORES['texto'])),
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        font=dict(color=CORES['texto']),
        showlegend=False,
        height=450,
        annotations=[dict(text=f'R$ {orcamento:,.0f}', x=0.5, y=0.5, 
                         font=dict(size=18, color=CORES['texto']), showarrow=False)]
    )
    
    return fig


def grafico_barras_alocacao(pesos: dict, orcamento: float, info_tickers: dict) -> go.Figure:
    """Gráfico de barras horizontal com alocação por ativo."""
    
    # Filtra e ordena
    pesos_filt = {k: v for k, v in pesos.items() if v > 0.005}
    pesos_ord = dict(sorted(pesos_filt.items(), key=lambda x: x[1], reverse=True))
    
    tickers = [t.replace('.SA', '') for t in pesos_ord.keys()]
    valores_pct = [v * 100 for v in pesos_ord.values()]
    valores_reais = [v * orcamento for v in pesos_ord.values()]
    nomes = [info_tickers.get(t, {}).get('nome', t) for t in pesos_ord.keys()]
    
    # Gradient de cores
    n = len(tickers)
    cores = [f'rgba(102, 126, 234, {0.4 + 0.6 * (n-i)/n})' for i in range(n)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=tickers,
        x=valores_pct,
        orientation='h',
        marker=dict(color=cores, line=dict(width=0)),
        text=[f'{v:.1f}% (R$ {r:,.0f})' for v, r in zip(valores_pct, valores_reais)],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>%{customdata}<br>Peso: %{x:.2f}%<extra></extra>',
        customdata=nomes
    ))
    
    # Linha de 20% (limite máximo)
    fig.add_vline(x=20, line_dash="dash", line_color=CORES['agressivo'],
                  annotation_text="Limite 20%")
    
    fig.update_layout(
        title=dict(text='📊 Alocação por Ativo', font=dict(size=20, color=CORES['texto'])),
        xaxis_title='Peso (%)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=max(400, len(tickers) * 35),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def grafico_matriz_correlacao(matriz_cov: pd.DataFrame) -> go.Figure:
    """Heatmap da matriz de correlação."""
    
    # Converte covariância em correlação
    std = np.sqrt(np.diag(matriz_cov))
    corr = matriz_cov / np.outer(std, std)
    corr = corr.clip(-1, 1)  # Garante valores válidos
    
    labels = [t.replace('.SA', '') for t in corr.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=8),
        hovertemplate='%{x} vs %{y}<br>Correlação: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='🔗 Matriz de Correlação', font=dict(size=20, color=CORES['texto'])),
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        font=dict(color=CORES['texto']),
        height=600,
        xaxis=dict(tickangle=45)
    )
    
    return fig


def grafico_evolucao_precos(precos: pd.DataFrame, tickers_selecionados: list) -> go.Figure:
    """Gráfico de evolução dos preços normalizados."""
    
    # Normaliza para base 100
    precos_norm = precos[tickers_selecionados] / precos[tickers_selecionados].iloc[0] * 100
    
    fig = go.Figure()
    
    cores = px.colors.qualitative.Plotly
    
    for i, col in enumerate(precos_norm.columns):
        fig.add_trace(go.Scatter(
            x=precos_norm.index,
            y=precos_norm[col],
            mode='lines',
            name=col.replace('.SA', ''),
            line=dict(width=2, color=cores[i % len(cores)]),
            hovertemplate='%{x}<br>Valor: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='📉 Evolução dos Preços (Base 100)', font=dict(size=20, color=CORES['texto'])),
        xaxis_title='Data',
        yaxis_title='Valor Normalizado',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=400,
        legend=dict(orientation='h', y=-0.2)
    )
    
    return fig


def grafico_setores(pesos: dict, info_tickers: dict) -> go.Figure:
    """Gráfico de pizza por setor."""
    
    setores = {}
    for ticker, peso in pesos.items():
        if peso > 0.005:
            setor = info_tickers.get(ticker, {}).get('setor', 'Outros')
            setores[setor] = setores.get(setor, 0) + peso
    
    fig = go.Figure(data=[go.Pie(
        labels=list(setores.keys()),
        values=list(setores.values()),
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set2)
    )])
    
    fig.update_layout(
        title=dict(text='🏢 Distribuição por Setor', font=dict(size=20, color=CORES['texto'])),
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        font=dict(color=CORES['texto']),
        height=400
    )
    
    return fig


def grafico_backtesting(serie_carteira: pd.Series, serie_benchmark: pd.Series = None,
                        nome_benchmark: str = "IBOV") -> go.Figure:
    """
    Gráfico de evolução do backtesting comparando carteira vs benchmark.
    
    Args:
        serie_carteira: Série temporal dos valores da carteira
        serie_benchmark: Série temporal do benchmark (opcional)
        nome_benchmark: Nome do benchmark
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    # Normaliza para base 100
    carteira_norm = serie_carteira / serie_carteira.iloc[0] * 100
    
    # Carteira otimizada
    fig.add_trace(go.Scatter(
        x=carteira_norm.index,
        y=carteira_norm.values,
        mode='lines',
        name='Carteira Otimizada',
        line=dict(color=CORES['azul'], width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='%{x}<br>Valor: %{y:.2f}<extra></extra>'
    ))
    
    # Benchmark se disponível
    if serie_benchmark is not None and len(serie_benchmark) > 0:
        # Alinha datas
        datas_comuns = carteira_norm.index.intersection(serie_benchmark.index)
        if len(datas_comuns) > 10:
            benchmark_norm = serie_benchmark.loc[datas_comuns]
            benchmark_norm = benchmark_norm / benchmark_norm.iloc[0] * 100
            
            fig.add_trace(go.Scatter(
                x=benchmark_norm.index,
                y=benchmark_norm.values,
                mode='lines',
                name=nome_benchmark,
                line=dict(color=CORES['agressivo'], width=2, dash='dash'),
                hovertemplate='%{x}<br>Valor: %{y:.2f}<extra></extra>'
            ))
    
    # Linha de base 100
    fig.add_hline(y=100, line_dash="dot", line_color="#888",
                  annotation_text="Base 100")
    
    fig.update_layout(
        title=dict(text='📈 Backtesting: Evolução da Carteira', font=dict(size=20, color=CORES['texto'])),
        xaxis_title='Data',
        yaxis_title='Valor (Base 100)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=450,
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    return fig


def grafico_drawdown(drawdown_serie: pd.Series) -> go.Figure:
    """
    Gráfico de drawdown ao longo do tempo.
    
    Args:
        drawdown_serie: Série temporal de drawdowns
        
    Returns:
        Figura Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown_serie.index,
        y=drawdown_serie.values * 100,
        mode='lines',
        name='Drawdown',
        line=dict(color=CORES['agressivo'], width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.3)',
        hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Linha de 0
    fig.add_hline(y=0, line_color="#888")
    
    # Destaca drawdown máximo
    max_dd_idx = drawdown_serie.idxmin()
    max_dd_val = drawdown_serie.min() * 100
    
    fig.add_annotation(
        x=max_dd_idx,
        y=max_dd_val,
        text=f"Máx: {max_dd_val:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor=CORES['agressivo'],
        font=dict(color=CORES['texto'])
    )
    
    fig.update_layout(
        title=dict(text='📉 Drawdown (Queda do Pico)', font=dict(size=20, color=CORES['texto'])),
        xaxis_title='Data',
        yaxis_title='Drawdown (%)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=350
    )
    
    return fig


def grafico_metricas_risco(metricas: dict) -> go.Figure:
    """
    Gráfico de barras com métricas de risco (VaR, CVaR, etc).
    
    Args:
        metricas: Dict com métricas de risco
        
    Returns:
        Figura Plotly
    """
    categorias = ['VaR 95%', 'CVaR 95%', 'Volatilidade']
    valores = [
        metricas.get('var_95_anual', 0) * 100,
        metricas.get('cvar_95_anual', 0) * 100,
        metricas.get('volatilidade_anual', 0) * 100
    ]
    
    cores = [CORES['moderado'], CORES['agressivo'], CORES['azul']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categorias,
        y=valores,
        marker=dict(color=cores),
        text=[f'{v:.1f}%' for v in valores],
        textposition='outside',
        hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='⚠️ Métricas de Risco Anualizado', font=dict(size=20, color=CORES['texto'])),
        yaxis_title='Percentual (%)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=350,
        showlegend=False
    )
    
    return fig


def grafico_comparativo_indices(sharpe: float, sortino: float, 
                                 alpha: float = None, beta: float = None) -> go.Figure:
    """
    Gráfico radar com índices de performance.
    
    Args:
        sharpe: Índice de Sharpe
        sortino: Índice de Sortino
        alpha: Alpha (vs benchmark)
        beta: Beta (vs benchmark)
        
    Returns:
        Figura Plotly
    """
    categorias = ['Sharpe', 'Sortino']
    valores = [sharpe, sortino]
    
    if alpha is not None:
        categorias.append('Alpha')
        valores.append(alpha * 100)  # Em percentual
    
    # Normaliza para visualização (escala 0-1 baseada em valores típicos)
    valores_norm = []
    for i, v in enumerate(valores):
        if categorias[i] in ['Sharpe', 'Sortino']:
            # Sharpe/Sortino bom: > 1, excelente: > 2
            valores_norm.append(min(v / 2, 1.5))
        else:
            # Alpha bom: > 0
            valores_norm.append(min((v + 10) / 20, 1.5))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=valores_norm + [valores_norm[0]],  # Fecha o polígono
        theta=categorias + [categorias[0]],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color=CORES['azul'], width=3),
        name='Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.5]),
            bgcolor=CORES['card']
        ),
        title=dict(text='🎯 Índices de Performance', font=dict(size=20, color=CORES['texto'])),
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        font=dict(color=CORES['texto']),
        height=400,
        showlegend=False
    )
    
    return fig

