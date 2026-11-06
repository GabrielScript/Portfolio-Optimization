"""
formulas.py - The 28 formulations tested against the Composite on the B3 (DECISOES 265 and 268): the 25 of Palomar's
book in two batches and the 3 optimizers fed the Composite's alpha, each read once on the same frozen inputs. Built by
``scripts/livro_tabela.py b3`` on the research branch (``feat/lote3-composto``) and copied to
``docs/dados/formulas-testadas-b3.parquet``. UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.filtros import filtrar

TITULO = "## As 28 fórmulas testadas contra a Composite"
GRUPOS = {0: "Referência", 1: "Livro, lote 1", 2: "Livro, lote 2", 3: "Alimentadas pelo alfa da Composite"}
NOMES = {"Composto": "Composite"}
PERCENTUAIS = ("CAGR (pior fase)", "CAGR (melhor fase)", "Vol", "Queda máx.", "P[> Composite]", "P[> Conservadora]",
               "P[> BOVA11]")
LEGENDA = ("Cada fórmula rodou no mesmo motor, com o mesmo universo da B3 sem viés de sobrevivência, em três fases de "
           "início (0, 21 e 42 pregões), e foi lida uma única vez. Para passar no portão, a probabilidade de bater a "
           "Composite, a Conservadora e o BOVA11 precisava ser de 95% ou mais em todas as fases, com DSR de 0,95 ou mais; "
           "a tabela mostra a pior fase de cada medida. Nenhuma passou: o sinal da Composite paga, o otimizador não "
           "acrescenta. Antes de custos e IR; passado não garante futuro.")


def tabela(dados: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Grupo": dados["lote"].map(GRUPOS), "Fórmula": dados["braco"].replace(NOMES),
        "CAGR (pior fase)": dados["cagr_min"], "CAGR (melhor fase)": dados["cagr_max"], "Vol": dados["vol"],
        "Queda máx.": dados["queda"], "P[> Composite]": dados["p_composto"],
        "P[> Conservadora]": dados["p_conservadora"], "P[> BOVA11]": dados["p_referencia"], "DSR": dados["dsr"],
        "Giro (vezes ao ano)": dados["giro"],
        "Passou no portão": [("—" if lote == 0 else ("sim" if passa else "não"))
                             for lote, passa in zip(dados["lote"], dados["passa"])]})


def secao(dados: pd.DataFrame | None) -> None:
    st.markdown(TITULO)
    if dados is None:
        st.caption("Sem a tabela das fórmulas testadas nesta versão do app.")
        return
    quadro = tabela(dados)
    configuracao = {coluna: st.column_config.NumberColumn(format="percent") for coluna in PERCENTUAIS}
    configuracao |= {"DSR": st.column_config.NumberColumn(format="%.2f"),
                     "Giro (vezes ao ano)": st.column_config.NumberColumn(format="%.2f")}
    st.dataframe(filtrar(quadro, "formulas_testadas"), width="stretch", hide_index=True, column_config=configuracao,
                 alt="As 28 fórmulas testadas, com CAGR, risco, probabilidade de bater cada referência e o portão")
    st.caption(LEGENDA)
