"""
visualizations.py - The capital curves chart of the backtest sections (Plotly).

Colors come from the Streamlit theme (``.streamlit/config.toml``, the redesign of 2026-11-06); the old dashboard's
frontier, pie, bar and drawdown charts and their black-and-gold palette left with the page they drew for.
"""

from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go


def grafico_curvas_capital(curvas: Dict[str, pd.Series], destaque: Optional[str] = None,
                           titulo: str = 'Curva de Capital — Walk-Forward fora da amostra, líquido de custos') -> go.Figure:
    """
    Curvas de capital em base 100, na janela comum a todas. Todas as curvas
    vêm do mesmo walk-forward (mesmas datas, mesmos custos), então a
    comparação é justa.

    Colors come from the Streamlit theme's colorway (``chartCategoricalColors`` of the light or dark section,
    the user's design request of 2026-11-06): the highlighted curve goes first, so it takes the first color, and the
    ground stays clear so the page's appearance shows through.
    """
    estilos = ['dot', 'dash', 'dashdot', 'longdash', 'dot']
    fig = go.Figure()
    datas_comuns = None
    for serie in curvas.values():
        datas_comuns = serie.index if datas_comuns is None else datas_comuns.intersection(serie.index)
    ordem = sorted(curvas, key=lambda nome: nome != destaque)          # stable: the others keep their order
    for i, nome in enumerate(ordem):
        recorte = curvas[nome].loc[datas_comuns]
        base100 = recorte / recorte.iloc[0] * 100
        principal = nome == destaque
        fig.add_trace(go.Scatter(
            x=base100.index, y=base100.values, mode='lines', name=nome,
            line=dict(width=2.8 if principal else 1.5, dash='solid' if principal else estilos[i % len(estilos)]),
            hovertemplate=f'{nome}: ' + '%{y:.1f}<extra></extra>'
        ))

    fig.add_hline(y=100, line_dash="solid", line_color="rgba(128,128,128,0.45)", line_width=1)

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=18)),
        xaxis_title='',
        yaxis_title='Capital Indexado (Base 100)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified'
    )
    return fig
