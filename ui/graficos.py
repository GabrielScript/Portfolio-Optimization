"""
graficos.py - The live accounts' curves and return tables in Brazilian format (the finish review of 2026-11-06).

``st.line_chart`` starts every axis at zero (R$ 10,000 to R$ 10,067 drew a flat line), writes "10,000" and English
dates, and shows drawdowns as raw fractions. These Altair charts keep Streamlit's theme colors (no color range is set)
and add a pt-BR locale, a zoomed R$ axis, dd/mm dates and drawdowns in percent. UI strings are Portuguese.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from ui import formato

# Vega's d3 locale for Brazil: decimal comma, dot for thousands, R$ before the amount, Portuguese month names
LOCALE = {
    "number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["R$ ", ""]},
    "time": {"dateTime": "%A, %e de %B de %Y. %X", "date": "%d/%m/%Y", "time": "%H:%M:%S",
             "periods": ["AM", "PM"],
             "days": ["Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"],
             "shortDays": ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"],
             "months": ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro",
                        "Outubro", "Novembro", "Dezembro"],
             "shortMonths": ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]},
}


def _longo(curvas: pd.DataFrame, valor: str) -> pd.DataFrame:
    return (curvas.rename_axis("Data").reset_index()
            .melt(id_vars="Data", var_name="Curva", value_name=valor).dropna(subset=[valor]))


def _linhas(curvas: pd.DataFrame, valor: str, titulo: str, formato: str, zero: bool) -> alt.Chart:
    return (alt.Chart(_longo(curvas, valor))
            .mark_line(strokeWidth=2)
            .encode(x=alt.X("Data:T", title=None, axis=alt.Axis(format="%d/%m", labelAngle=0,
                                                                 tickCount={"interval": "day", "step": 1})),
                    y=alt.Y(f"{valor}:Q", title=titulo, scale=alt.Scale(zero=zero), axis=alt.Axis(format=formato)),
                    color=alt.Color("Curva:N", title=None, legend=alt.Legend(orient="bottom")),
                    tooltip=[alt.Tooltip("Data:T", format="%d/%m/%Y"), "Curva:N",
                             alt.Tooltip(f"{valor}:Q", format=formato)])
            .properties(height=320)
            .configure(locale=LOCALE))


def curvas_em_reais(curvas: pd.DataFrame) -> alt.Chart:
    """Dates x accounts, in R$: the axis zooms on the curves (a R$ 67 gain is visible on R$ 10,000)."""
    return _linhas(curvas, "Valor", "Patrimônio (R$)", "$,.0f", zero=False)


def quedas_em_percentual(quedas: pd.DataFrame) -> alt.Chart:
    """Dates x accounts, the fall from the peak as a fraction (``core.simulador.quedas``), read in percent."""
    return _linhas(quedas, "Queda", "Queda desde o pico (%)", ".1%", zero=True)


def tom_por_dia(tom: pd.Series) -> alt.Chart:
    """The news tone of each day (-1 to +1) in signed pt-BR numbers and dd/mm dates (st.line_chart wrote "-0.14" and
    English dates)."""
    dados = tom.rename_axis("Data").reset_index(name="Tom")
    return (alt.Chart(dados)
            .mark_line(strokeWidth=2, point=True)
            .encode(x=alt.X("Data:T", title=None, axis=alt.Axis(format="%d/%m", labelAngle=0)),
                    y=alt.Y("Tom:Q", title="Tom do mercado (−1 a +1)", axis=alt.Axis(format="+.2f")),
                    tooltip=[alt.Tooltip("Data:T", format="%d/%m/%Y"), alt.Tooltip("Tom:Q", format="+.2f")])
            .properties(height=260)
            .configure(locale=LOCALE))


def percentuais_em_texto(tabela: pd.DataFrame) -> pd.DataFrame:
    """A table of fractions as pt-BR percent strings, with a dash where the account has no history for the window
    (the dataframe printed "None" there)."""
    return tabela.apply(lambda coluna: coluna.map(formato.percentual))


def colunas_de_texto(tabela: pd.DataFrame) -> dict:
    """Right-aligned text columns, so the pt-BR percent strings line up like numbers."""
    import streamlit as st

    return {coluna: st.column_config.TextColumn(alignment="right") for coluna in tabela.columns}


def barras_por_setor(pesos: pd.Series) -> alt.Chart:
    """The Composite's weight by sector, in percent, with whole sector names."""
    dados = pesos.sort_values(ascending=False).rename_axis("Setor").reset_index(name="Peso")
    return (alt.Chart(dados)
            .mark_bar(cornerRadiusEnd=3)
            .encode(x=alt.X("Peso:Q", title="Peso na Composite (%)", axis=alt.Axis(format=".0%")),
                    y=alt.Y("Setor:N", title=None, sort="-x", axis=alt.Axis(labelLimit=320)),
                    tooltip=["Setor:N", alt.Tooltip("Peso:Q", format=".1%")])
            .properties(height=alt.Step(26))
            .configure(locale=LOCALE))
