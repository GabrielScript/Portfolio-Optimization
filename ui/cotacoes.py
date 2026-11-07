"""
cotacoes.py - The Composite's quotes now (the user, 2026-11-05: the current quotes table left the page with the buy
plan on 2026-11-04; it comes back on its own): B3's last trade of the session (~15 min late, ``core.cotacao_b3``), or
the official close once the session ended, next to the previous close and the day's change, with a refresh button.

Information only (DECISOES 124): no order or quantity here. UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import aparencia, formato
from ui.filtros import filtrar

TITULO = "## Cotações de agora: as ações da Composite"
PREGOES_PARA_FECHAMENTO = 10      # the window the last and the previous close come from


def tabela_de_cotacoes(cotacoes: pd.DataFrame, fechamentos: pd.DataFrame, info: dict) -> pd.DataFrame:
    """One row per ticker of ``cotacoes`` (``preco``, ``horario``): name and sector from ``info``, the quote, the close
    of the session before the quote's and the change between them. ``fechamentos``: dates x tickers. The source column
    left on 2026-11-06 (the user)."""
    linhas = []
    for ticker, cotacao in cotacoes.iterrows():
        sessao = pd.Timestamp(cotacao["horario"]).tz_localize(None).normalize() if pd.notna(cotacao["horario"]) else None
        validos = fechamentos[ticker].dropna() if ticker in fechamentos.columns else pd.Series(dtype=float)
        antes = validos.loc[validos.index < sessao] if sessao is not None else validos.iloc[:0]
        anterior = float(antes.iloc[-1]) if len(antes) else float("nan")
        preco = float(cotacao["preco"])
        linhas.append({"Ticker": ticker.removesuffix(".SA"), "Empresa": info.get(ticker, {}).get("nome", ""),
                       "Setor": info.get(ticker, {}).get("setor", ""), "Cotação (R$)": preco,
                       "Fechamento anterior (R$)": anterior, "Variação no dia": preco / anterior - 1,
                       "Horário": pd.Timestamp(cotacao["horario"]) if sessao is not None else pd.NaT})
    return pd.DataFrame(linhas)


def secao(tickers: tuple[str, ...], fechamentos: pd.DataFrame, carregar, descartar, info: dict, faixa=None,
          nativo: str = "dark") -> None:
    """The section: ``carregar(tickers, janela)`` and ``descartar(tickers, janela)`` are ``ui.servicos``'s cached quote
    loader and its per-portfolio clear; ``faixa``, an ``st.empty`` higher on the page, gets the quote band."""
    st.markdown(TITULO)
    if not tickers:
        st.caption("Sem a Composite registrada.")
        return
    janela = fechamentos.tail(PREGOES_PARA_FECHAMENTO)
    if st.button("Atualizar cotações", icon=":material/refresh:", key="atualizar_cotacoes"):
        descartar(tickers, janela)
    with st.spinner("Buscando as cotações do dia…"):
        cotacoes = carregar(tickers, janela)
    tabela = tabela_de_cotacoes(cotacoes, janela, info)
    if faixa is not None and (band := aparencia.fita(tabela, nativo)):
        faixa.html(band)
    estilo, config = formato.br(filtrar(tabela, "cotacoes"), {"Cotação (R$)": formato.decimal,
                                                              "Fechamento anterior (R$)": formato.decimal,
                                                              "Variação no dia": formato.percentual})
    st.dataframe(estilo, width="stretch", hide_index=True,
                 column_config={**config, "Horário": st.column_config.DatetimeColumn(format="DD/MM HH:mm")},
                 alt="Cotação de agora, fechamento anterior e variação no dia de cada ação da Composite")
    st.caption("Último negócio na B3 com atraso de cerca de 15 minutos; depois do pregão, o fechamento oficial. "
               "Variação contra o fechamento do pregão anterior.")
