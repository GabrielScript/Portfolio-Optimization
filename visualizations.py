"""
visualizations.py - Gráficos Plotly para o Dashboard (Rigor Quantitativo)
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
                                 perfil: str, taxa_livre_risco: float,
                                 benchmark_1n: dict = None) -> go.Figure:
    """
    Fronteira eficiente (sem cardinalidade), carteira escolhida e 1/N.
    ``taxa_livre_risco`` na convenção aritmética 252 x diária, a mesma de mu.
    A carteira com limite de nº de ativos pode ficar ABAIXO da fronteira.
    """
    cor_perfil = CORES['agressivo']
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

def grafico_curvas_capital(curvas: Dict[str, pd.Series], destaque: Optional[str] = None) -> go.Figure:
    """
    Curvas de capital em base 100, na janela comum a todas. Todas as curvas
    vêm do mesmo walk-forward (mesmas datas, mesmos custos), então a
    comparação é justa.
    """
    paleta = [CORES['azul'], CORES['moderado'], CORES['conservador'], CORES['roxo'],
              CORES['alerta'], '#AAAAAA']
    estilos = ['solid', 'dot', 'dash', 'dashdot', 'longdash', 'dot']
    fig = go.Figure()
    datas_comuns = None
    for serie in curvas.values():
        datas_comuns = serie.index if datas_comuns is None else datas_comuns.intersection(serie.index)
    for i, (nome, serie) in enumerate(curvas.items()):
        recorte = serie.loc[datas_comuns]
        base100 = recorte / recorte.iloc[0] * 100
        principal = nome == destaque
        fig.add_trace(go.Scatter(
            x=base100.index, y=base100.values, mode='lines', name=nome,
            line=dict(color=paleta[i % len(paleta)], width=2.8 if principal else 1.5,
                      dash='solid' if principal else estilos[i % len(estilos)]),
            hovertemplate=f'{nome}: ' + '%{y:.1f}<extra></extra>'
        ))

    fig.add_hline(y=100, line_dash="solid", line_color="#444", line_width=1)

    fig.update_layout(
        title=dict(text='Curva de Capital — Walk-Forward fora da amostra, líquido de custos',
                   font=dict(size=18, color=CORES['texto'])),
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

def grafico_contribuicao_risco(contrib: pd.Series, pesos: dict = None) -> go.Figure:
    """
    Contribuição relativa de risco por ativo (RC_i = w_i (Σw)_i / σ_p²).

    Compara alocação de capital vs contribuição de risco — revela concentração
    de risco que a pizza de alocação esconde (Palomar §6.5.3, risk parity).
    """
    tickers = [t.replace('.SA', '') for t in contrib.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers,
        y=contrib.values * 100,
        name='Contribuição de Risco',
        marker=dict(color=CORES['alerta']),
        text=[f'{v*100:.1f}%' for v in contrib.values],
        textposition='auto',
        hovertemplate='%{x}<br>Contribuição de Risco: %{y:.1f}%<extra></extra>'
    ))

    if pesos is not None:
        pesos_alinhados = [pesos.get(t, 0) * 100 for t in contrib.index]
        fig.add_trace(go.Bar(
            x=tickers,
            y=pesos_alinhados,
            name='Alocação de Capital',
            marker=dict(color=CORES['azul']),
            text=[f'{v:.1f}%' for v in pesos_alinhados],
            textposition='auto',
            hovertemplate='%{x}<br>Alocação: %{y:.1f}%<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text='🎯 Capital vs Risco: Onde o Risco Realmente Mora',
                   font=dict(size=18, color=CORES['texto'])),
        xaxis_title='',
        yaxis_title='Participação (%)',
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=400,
        barmode='group',
        legend=dict(orientation='h', y=1.1)
    )
    return fig


def grafico_metricas_rolling(df_rolling: pd.DataFrame, janela: int = 63) -> go.Figure:
    """
    Volatilidade e Sharpe em janela móvel (Palomar §6.3.4).

    Expõe mudanças de regime de risco ao longo do tempo — a métrica cheia
    do período esconde que vol e correlações explodem juntas em stress.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f'Volatilidade Anualizada Rolling ({janela}d)',
                        f'Sharpe Rolling ({janela}d)')
    )

    fig.add_trace(go.Scatter(
        x=df_rolling.index, y=df_rolling['vol_rolling'] * 100,
        mode='lines', name='Vol Rolling',
        line=dict(color=CORES['alerta'], width=1.8),
        hovertemplate='%{x}<br>Vol: %{y:.1f}%<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_rolling.index, y=df_rolling['sharpe_rolling'],
        mode='lines', name='Sharpe Rolling',
        line=dict(color=CORES['azul'], width=1.8),
        hovertemplate='%{x}<br>Sharpe: %{y:.2f}<extra></extra>'
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash='dot', line_color=CORES['grid'], row=2, col=1)

    fig.update_layout(
        title=dict(text='📈 Regime de Risco ao Longo do Tempo',
                   font=dict(size=18, color=CORES['texto'])),
        template='plotly_dark',
        paper_bgcolor=CORES['fundo'],
        plot_bgcolor=CORES['card'],
        font=dict(color=CORES['texto']),
        height=500,
        showlegend=False
    )
    fig.update_yaxes(title_text='Vol (%)', row=1, col=1)
    fig.update_yaxes(title_text='Sharpe', row=2, col=1)
    return fig