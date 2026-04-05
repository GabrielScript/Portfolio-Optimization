"""
visualizations.py - Gráficos Plotly para o Dashboard (Rigor Quantitativo)
TCC: Otimização de Carteiras de Investimentos
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Paleta de cores adaptada para contraste institucional
CORES = {
    'fundo': '#0E1117',
    'card': '#1E2130',
    'texto': '#FAFAFA',
    'grid': '#2D3250',
    'conservador': '#00D4AA',
    'moderado': '#FFB74D',
    'agressivo': '#FF5252',
    'azul': '#667EEA',
    'roxo': '#764BA2',
    'alerta': '#FF3366'
}

def grafico_fronteira_eficiente(fronteira: pd.DataFrame, carteira_otima: dict, 
                                 perfil: str, taxa_livre_risco: float = 0.1075,
                                 benchmark_1n: dict = None) -> go.Figure:
    """
    Gráfico da Fronteira Eficiente.
    Corrigido para aceitar a taxa livre de risco dinamicamente e evitar distorções de escala.
    Inclui ponto de benchmark 1/N (carteira equiponderada) quando fornecido.
    """
    cor_perfil = CORES.get(perfil.lower(), CORES['moderado'])
    fig = go.Figure()
    
    if not fronteira.empty:
        fig.add_trace(go.Scatter(
            x=fronteira['volatilidade'] * 100,
            y=fronteira['retorno'] * 100,
            mode='lines',
            name='Fronteira Eficiente',
            line=dict(color=CORES['azul'], width=3),
            hovertemplate='Volatilidade: %{x:.2f}%<br>Retorno: %{y:.2f}%<extra></extra>'
        ))
    
    # Ponto ótimo destacado
    fig.add_trace(go.Scatter(
        x=[carteira_otima['volatilidade'] * 100],
        y=[carteira_otima['retorno'] * 100],
        mode='markers',
        name=f'Alocação: {perfil}',
        marker=dict(size=18, color=cor_perfil, symbol='diamond',
                   line=dict(width=2, color='white')),
        hovertemplate=f'<b>{perfil}</b><br>' +
                      'Risco (Vol): %{x:.2f}%<br>Retorno Esp.: %{y:.2f}%<extra></extra>'
    ))
    
    # Benchmark 1/N (carteira equiponderada ingénua)
    if benchmark_1n is not None:
        fig.add_trace(go.Scatter(
            x=[benchmark_1n['volatilidade'] * 100],
            y=[benchmark_1n['retorno'] * 100],
            mode='markers',
            name='Benchmark 1/N',
            marker=dict(size=14, color=CORES['moderado'], symbol='x-thin',
                       line=dict(width=3, color=CORES['moderado'])),
            hovertemplate='<b>1/N (Equiponderado)</b><br>' +
                          'Risco (Vol): %{x:.2f}%<br>Retorno Esp.: %{y:.2f}%<extra></extra>'
        ))
    
    # Linha base dinâmica (Taxa Livre de Risco real do momento)
    taxa_pct = taxa_livre_risco * 100
    fig.add_hline(y=taxa_pct, line_dash="dash", line_color="#888",
                  annotation_text=f"Taxa Sem Risco ({taxa_pct:.2f}%)", annotation_position="bottom right")
    
    fig.update_layout(
        title=dict(text='Fronteira Eficiente de Markowitz', font=dict(size=18, color=CORES['texto'])),
        xaxis_title='Risco / Volatilidade (% a.a.)',
        yaxis_title='Retorno Esperado (% a.a.)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0)'),
        height=500
    )
    
    return fig

def grafico_composicao_pizza(pesos: dict, orcamento: float) -> go.Figure:
    """Gráfico estático da última alocação recomendada."""
    # Filtro rigoroso: Otimizadores convexos evitam caudas infinitas, limpamos < 0.1%
    pesos_filtrados = {k: v for k, v in pesos.items() if v >= 0.001}
    
    if not pesos_filtrados: return go.Figure()

    labels = [t.replace('.SA', '') for t in pesos_filtrados.keys()]
    valores = list(pesos_filtrados.values())
    valores_reais = [v * orcamento for v in valores]
    cores = px.colors.qualitative.Pastel[:len(labels)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=valores,
        hole=0.6,
        marker=dict(colors=cores, line=dict(color=CORES['card'], width=1.5)),
        textinfo='percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Peso Relativo: %{percent}<br>Capital Alocado: R$ %{customdata:,.2f}<extra></extra>',
        customdata=valores_reais
    )])
    
    fig.update_layout(
        title=dict(text='Composicao da Carteira', font=dict(size=18, color=CORES['texto'])),
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        font=dict(color=CORES['texto']),
        showlegend=False,
        height=450,
        annotations=[dict(text=f'Exposição Total:<br>R$ {orcamento:,.0f}', x=0.5, y=0.5, 
                         font=dict(size=16, color=CORES['texto'], weight='bold'), showarrow=False)]
    )
    return fig

def grafico_barras_alocacao(pesos: dict, orcamento: float, info_tickers: dict) -> go.Figure:
    """Bar plot da composição transversal atual."""
    pesos_filt = {k: v for k, v in pesos.items() if v >= 0.001}
    pesos_ord = dict(sorted(pesos_filt.items(), key=lambda x: x[1], reverse=True))
    
    tickers = [t.replace('.SA', '') for t in pesos_ord.keys()]
    valores_pct = [v * 100 for v in pesos_ord.values()]
    valores_reais = [v * orcamento for v in pesos_ord.values()]
    nomes = [info_tickers.get(t, {}).get('nome', t) for t in pesos_ord.keys()]
    
    n = len(tickers)
    cores = [f'rgba(102, 126, 234, {0.3 + 0.7 * (n-i)/n})' for i in range(n)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=tickers,
        x=valores_pct,
        orientation='h',
        marker=dict(color=cores, line=dict(width=0)),
        text=[f'{v:.1f}%' for v in valores_pct],
        textposition='outside',
        hovertemplate='<b>%{y}</b> (%{customdata})<br>Alocação: %{x:.2f}%<extra></extra>',
        customdata=nomes
    ))
    
    fig.update_layout(
        title=dict(text='Alocacao por Ativo (%)', font=dict(size=18, color=CORES['texto'])),
        xaxis_title='Peso (%)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=max(400, len(tickers) * 35),
        yaxis=dict(autorange='reversed', title='')
    )
    return fig

def grafico_matriz_correlacao(matriz_cov: pd.DataFrame) -> go.Figure:
    """
    Heatmap institucional da estrutura de dependência estatística.
    Inclui salvaguarda contra NaN derivada de matrizes estabilizadas com baixa variância.
    """
    std = np.sqrt(np.diag(matriz_cov))
    # Proteção: Substitui zeros por 1 para a divisão para evitar runtime warnings/NaNs
    std_safe = np.where(std == 0, 1.0, std) 
    
    corr = matriz_cov.values / np.outer(std_safe, std_safe)
    # Limpeza para assets sem volatilidade real (casos limite de optimização)
    corr = np.where(np.outer(std, std) == 0, 0, corr)
    corr = np.clip(corr, -1.0, 1.0) 
    
    labels = [t.replace('.SA', '') for t in matriz_cov.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr, 2),
        texttemplate='%{text}',
        textfont=dict(size=10, color='white'),
        hovertemplate='%{x} vs %{y}<br>Matriz de Correlação: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Matriz de Correlacao', font=dict(size=18, color=CORES['texto'])),
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        height=650,
        xaxis=dict(tickangle=45, title=''),
        yaxis=dict(title='')
    )
    return fig

def grafico_evolucao_precos(precos: pd.DataFrame, tickers_selecionados: list) -> go.Figure:
    """Tracking visual de retorno acumulativo estático."""
    # Previne divisão por zeros no início da base
    precos_base = precos[tickers_selecionados].copy()
    precos_base = precos_base.replace(0, np.nan).bfill() 
    precos_norm = precos_base / precos_base.iloc[0] * 100
    
    fig = go.Figure()
    cores = px.colors.qualitative.Bold
    
    for i, col in enumerate(precos_norm.columns):
        fig.add_trace(go.Scatter(
            x=precos_norm.index,
            y=precos_norm[col],
            mode='lines',
            name=col.replace('.SA', ''),
            line=dict(width=1.5, color=cores[i % len(cores)]),
            hovertemplate='Data: %{x|%Y-%m-%d}<br>Índice de Retorno: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='Evolucao de Precos Normalizados (Base 100)', font=dict(size=18, color=CORES['texto'])),
        xaxis_title='',
        yaxis_title='Indice (Base 100)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=450,
        legend=dict(orientation='h', y=-0.15),
        hovermode='x unified'
    )
    return fig

def grafico_backtesting(serie_carteira: pd.Series, serie_benchmark: pd.Series = None,
                        nome_benchmark: str = "IBOV", serie_1n: pd.Series = None) -> go.Figure:
    """
    A prova visual empírica do motor Out-of-Sample.
    Inclui comparação com Ibovespa e carteira 1/N (equiponderada) conforme Q1 do TCC.
    """
    fig = go.Figure()

    carteira_norm = serie_carteira / serie_carteira.iloc[0] * 100

    fig.add_trace(go.Scatter(
        x=carteira_norm.index,
        y=carteira_norm.values,
        mode='lines',
        name='Carteira Otimizada',
        line=dict(color=CORES['azul'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.15)',
        hovertemplate='Otimizada: %{y:.2f}<extra></extra>'
    ))

    if serie_benchmark is not None and not serie_benchmark.empty:
        datas_comuns = carteira_norm.index.intersection(serie_benchmark.index)
        if len(datas_comuns) > 5:
            benchmark_norm = serie_benchmark.loc[datas_comuns]
            benchmark_norm = benchmark_norm / benchmark_norm.iloc[0] * 100

            fig.add_trace(go.Scatter(
                x=benchmark_norm.index,
                y=benchmark_norm.values,
                mode='lines',
                name=f'Mercado ({nome_benchmark})',
                line=dict(color=CORES['alerta'], width=1.5, dash='dashdot'),
                hovertemplate='Mercado: %{y:.2f}<extra></extra>'
            ))

    if serie_1n is not None:
        serie_1n_norm = serie_1n / serie_1n.iloc[0] * 100
        datas_comuns_1n = carteira_norm.index.intersection(serie_1n_norm.index)
        if len(datas_comuns_1n) > 5:
            serie_1n_plot = serie_1n_norm.loc[datas_comuns_1n]
            fig.add_trace(go.Scatter(
                x=serie_1n_plot.index,
                y=serie_1n_plot.values,
                mode='lines',
                name='Carteira 1/N',
                line=dict(color=CORES['moderado'], width=1.5, dash='dot'),
                hovertemplate='1/N: %{y:.2f}<extra></extra>'
            ))

    fig.add_hline(y=100, line_dash="solid", line_color="#444", line_width=1)

    fig.update_layout(
        title=dict(text='Curva de Capital (Backtesting)', font=dict(size=18, color=CORES['texto'])),
        xaxis_title='',
        yaxis_title='Capital Indexado (Base 100)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=450,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified'
    )
    return fig

def grafico_drawdown(drawdown_serie: pd.Series) -> go.Figure:
    """Grafico de drawdowns (quedas de pico a vale)."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown_serie.index,
        y=drawdown_serie.values * 100,
        mode='lines',
        name='Drawdown',
        line=dict(color=CORES['alerta'], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255, 51, 102, 0.25)',
        hovertemplate='%{x}<br>Drawdown Atual: %{y:.2f}%<extra></extra>'
    ))
    
    min_idx = drawdown_serie.idxmin()
    min_val = drawdown_serie.min() * 100
    
    fig.add_annotation(
        x=min_idx,
        y=min_val,
        text=f"Max Drawdown: {min_val:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor=CORES['texto'],
        ax=0,
        ay=-40,
        font=dict(color=CORES['texto'], size=12, weight='bold')
    )
    
    fig.update_layout(
        title=dict(text='Drawdowns (Quedas de Pico a Vale)', font=dict(size=18, color=CORES['texto'])),
        xaxis_title='',
        yaxis_title='Queda do Pico (%)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=350,
        yaxis=dict(autorange='reversed')
    )
    return fig

def grafico_pesos_historicos(df_pesos: pd.DataFrame) -> go.Figure:
    """
    (VISIONÁRIO): Gráfico da evolução dos pesos da carteira durante o Walk-Forward.
    df_pesos deve ter como índice a Data e colunas os Ativos.
    """
    fig = px.area(df_pesos, 
                  color_discrete_sequence=px.colors.qualitative.Prism,
                  title="🔄 Rebalanceamentos Estruturais (Evolução de Pesos OOS)")
                  
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        xaxis_title="",
        yaxis_title="Exposição Relativa (Alocação %)",
        legend_title="Ativo",
        height=400,
        hovermode="x unified"
    )
    # Formata y axis em percentagem
    fig.update_yaxes(tickformat=".0%")
    return fig

def grafico_metricas_risco(metricas: dict) -> go.Figure:
    """
    Bar plot que respeita a escala correta de métricas de Cauda (Cornish-Fisher) vs Gaussianas.
    A tua versão misturava taxas normalizadas.
    """
    categorias = ['Value at Risk (Cauda)', 'Cond. VaR (Stress Extremo)', 'Volatilidade Anual']
    
    # Busca a versão robusta primeiro, fallback para standard
    var_95 = metricas.get('var_95_anual', 0) * 100
    cvar_95 = metricas.get('cvar_95_anual', 0) * 100
    vol_anual = metricas.get('volatilidade_anual', 0) * 100
    
    valores = [var_95, cvar_95, vol_anual]
    cores = [CORES['moderado'], CORES['alerta'], CORES['azul']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=valores,
        y=categorias,
        orientation='h',
        marker=dict(color=cores, line=dict(width=0)),
        text=[f'{v:.2f}%' for v in valores],
        textposition='auto',
        hovertemplate='%{y}<br>Risco de Proteção: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='⚠️ Risco Não-Linear (Painel de Stress Test)', font=dict(size=18, color=CORES['texto'])),
        xaxis_title='Vulnerabilidade do Capital (%)',
        yaxis_title='',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=350,
        showlegend=False
    )
    return fig