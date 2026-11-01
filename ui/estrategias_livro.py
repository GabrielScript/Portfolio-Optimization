"""
estrategias_livro.py - Screen of the strategies from the portfolio optimization books, tested on the B3 and on US stocks
(docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md and 2026-11-01-lote2-livro-design.md). Past results, read from the derived
files in ``docs/dados``. UI strings are Portuguese.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import streamlit as st

PASTA = Path(__file__).resolve().parents[1] / "docs" / "dados"
MERCADOS = {"b3": ("B3", "BOVA11"), "us": ("Ações americanas", "SPY")}
LOTES = {0: "Referência", 1: "Primeiro", 2: "Segundo"}
PERCENTUAL = st.column_config.NumberColumn(format="percent")
TRES_CASAS = st.column_config.NumberColumn(format="%.3f")


def carregar(pasta: Path = PASTA) -> dict[str, pd.DataFrame]:
    """The markets whose table exists."""
    return {mercado: pd.read_parquet(pasta / f"livro-tabela-{mercado}.parquet") for mercado in MERCADOS
            if (pasta / f"livro-tabela-{mercado}.parquet").exists()}


def _veredito(tabelas: Mapping[str, pd.DataFrame]) -> str:
    candidatas = [f"{braco} ({MERCADOS[mercado][0]})" for mercado, tabela in tabelas.items()
                  for braco in tabela.loc[tabela["passa"].astype(bool), "braco"]]
    if not candidatas:
        return ("Nenhuma estratégia passou no portão até aqui. Nada aqui vira decisão de dinheiro real, e o Composto e a Conservadora "
                "continuam como estão.")
    return (f"Passou: {', '.join(candidatas)}. Passar faz da estratégia apenas candidata a um registro ao vivo; nenhuma decisão de "
            "dinheiro real sai desta tabela.")


def _para_a_tela(tabela: pd.DataFrame, referencia: str) -> pd.DataFrame:
    ordenada = tabela.sort_values("cagr_min", ascending=False)
    return pd.DataFrame({"Estratégia": ordenada["braco"], "Lote": ordenada["lote"].map(LOTES),
                         "Retorno a.a., pior fase": ordenada["cagr_min"], "Retorno a.a., melhor fase": ordenada["cagr_max"],
                         "Vol": ordenada["vol"], "Queda máx.": ordenada["queda"], "Giro anual": ordenada["giro"], "Recusas": ordenada["recusas"],
                         "P[> Composto]": ordenada["p_composto"], "P[> Conservadora]": ordenada["p_conservadora"],
                         f"P[> {referencia}]": ordenada["p_referencia"], "DSR": ordenada["dsr"], "Passou": ordenada["passa"].astype(bool)})


def _mercado(mercado: str, tabela: pd.DataFrame) -> None:
    nome, referencia = MERCADOS[mercado]
    st.markdown(f"### {nome}")
    colunas = {"Retorno a.a., pior fase": PERCENTUAL, "Retorno a.a., melhor fase": PERCENTUAL, "Vol": PERCENTUAL, "Queda máx.": PERCENTUAL,
               "Giro anual": st.column_config.NumberColumn(format="%.1f"), "P[> Composto]": TRES_CASAS, "P[> Conservadora]": TRES_CASAS,
               f"P[> {referencia}]": TRES_CASAS, "DSR": TRES_CASAS}
    st.dataframe(_para_a_tela(tabela, referencia), width="stretch", hide_index=True, column_config=colunas,
                 alt=f"Retorno, risco, giro e portão de cada estratégia dos livros em {nome}, da melhor para a pior na pior fase")


def secao(tabelas: Mapping[str, pd.DataFrame]) -> None:
    st.markdown("## Estratégias dos livros de otimização")
    if not tabelas:
        st.info("O teste ainda não terminou: sem resultado para mostrar.")
        return
    st.warning(_veredito(tabelas))
    st.caption("Cada linha é uma estratégia com as regras do app: 20% por ação, 30% por setor e 40% de volatilidade, exceto \"Kelly-exato-sem-teto\", "
               "que tira de propósito o teto de volatilidade para medir o que ele custa. Três fases, com datas de rebalanceio deslocadas. O portão pede "
               "P contra o Composto, a Conservadora e a referência, e DSR, todos de pelo menos 0,95 nas três fases; aqui aparece o menor de cada um. "
               "Rebalanceio a cada 3 meses, antes de IR; B3 em reais, ações americanas em dólares. Recusa é a data em que a carteira quebrou uma regra "
               "e a anterior ficou. Resultado passado, sem promessa de futuro.")
    for mercado, tabela in tabelas.items():
        _mercado(mercado, tabela)
    for mercado, (nome, _) in MERCADOS.items():
        if mercado not in tabelas:
            st.info(f"{nome}: o teste ainda não terminou.")
