"""
eua.py - Screen of the Alepha strategies on US stocks (``scripts/us_avaliar.py``). Results measured in the past, read from
the derived files in ``docs/dados``, so it shows the same locally and in the deployed app. UI strings are Portuguese.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

PERCENTUAL = st.column_config.NumberColumn(format="percent")
COLUNAS_DO_RESUMO = {"CAGR": st.column_config.NumberColumn("Retorno a.a. (R$)", format="percent"),
                     "Vol": PERCENTUAL, "Queda máx.": PERCENTUAL,
                     "R$ 1 vira": st.column_config.NumberColumn(format="R$ %.2f"),
                     "Desde": st.column_config.DateColumn(format="MM/YYYY")}
COLUNAS_DO_PORTAO = {"P[> SPY] mínimo": st.column_config.NumberColumn(format="%.3f"),
                     "DSR mínimo": st.column_config.NumberColumn(format="%.3f")}
LINHAS_DO_GRAFICO = ("Composto", "Conservadora", "SPY", "B3 Composto", "B3 Conservadora", "BOVA11", "CDI",
                     "15 fundos BR, melhores 12 meses")


@dataclass(frozen=True)
class ResultadoEUA:
    curvas: pd.DataFrame        # R$ 1 at the first date, in reais
    resumo: pd.DataFrame
    portao: pd.DataFrame
    carteiras: pd.DataFrame
    cobertura: dict


def _milhar(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def secao(resultado: ResultadoEUA | None) -> None:
    st.markdown("## Ações americanas")
    if resultado is None:
        st.info("O teste das estratégias nas ações americanas ainda não rodou: sem resultado para mostrar.")
        return
    cobertura = resultado.cobertura
    st.warning(f"Base incompleta nas empresas que saíram da bolsa: {_milhar(cobertura['dead_from_tiingo'])} de "
               f"{_milhar(cobertura['mortos_candidatos'])} nomes mortos já vieram do Tiingo, os maiores primeiro. Sem "
               "todos, o resultado tende a sair melhor do que seria. Nenhuma decisão de dinheiro real sai desta tabela.")
    st.caption(f"{_milhar(cobertura['companies'])} empresas, {cobertura['first_month'][:7]} a "
               f"{cobertura['last_month'][:7]}; preços do stockanalysis e do Tiingo, balanços do EDGAR, setores e "
               "fatores do Ken French, macro do FRED, câmbio PTAX. Rebalanceio a cada 3 meses, 10 bps por lado. "
               "Resultado passado, sem promessa de futuro.")
    st.markdown("### Em reais, nas mesmas datas")
    tabela = resultado.resumo.sort_values("CAGR", ascending=False)
    st.dataframe(tabela, width="stretch", hide_index=True, column_config=COLUNAS_DO_RESUMO,
                 alt="Retorno anual em reais, volatilidade, queda máxima e valor final de cada estratégia, fundo e ETF")
    st.caption("\"Retrospecto\" são os melhores do período escolhidos depois de vê-lo: ninguém poderia tê-los escolhido "
               "no início. Os três fundos citados começam quando passam a existir na janela.")
    linhas = [nome for nome in LINHAS_DO_GRAFICO if nome in resultado.curvas.columns]
    st.line_chart(resultado.curvas[linhas], y_label="R$ 1 aplicado no início")
    st.markdown("### Passou no teste?")
    st.caption(f"Mesmo portão das estratégias na B3: P[> SPY] e DSR de pelo menos 0,95 nas três fases, com N = "
               f"{cobertura['n_tentativas']}. Passar faz da estratégia candidata a um registro ao vivo, nada mais.")
    st.dataframe(resultado.portao, width="stretch", hide_index=True, column_config=COLUNAS_DO_PORTAO,
                 alt="Menor P contra o SPY e menor DSR de cada estratégia nas três fases, e se passou no portão")
    mes = pd.Timestamp(resultado.carteiras["mes"].iloc[0])
    st.markdown(f"### Quem as regras escolheriam em {mes:%m/%Y}")
    st.caption("Informação, não recomendação: mostra o que cada regra lê hoje.")
    st.dataframe(resultado.carteiras.drop(columns="mes"), width="stretch", hide_index=True,
                 column_config={"peso": PERCENTUAL}, alt="As 15 ações do Composto e da Conservadora nos Estados Unidos")
