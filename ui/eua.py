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
TODOS_OS_SETORES = "Todos os setores"
COLUNAS_DOS_INDICADORES = {"Valor de mercado": st.column_config.NumberColumn("Valor de mercado (US$)", format="compact"),
                           "Retorno 12m": PERCENTUAL, "Vol anual": PERCENTUAL, "Beta": st.column_config.NumberColumn(format="%.2f"),
                           "Queda máx. (3 anos)": PERCENTUAL, "P/L": st.column_config.NumberColumn(format="%.1f"),
                           "P/VP": st.column_config.NumberColumn(format="%.1f"), "ROE": PERCENTUAL, "Margem líquida": PERCENTUAL,
                           "Dív. líq./EBITDA": st.column_config.NumberColumn(format="%.1f"), "DY 12m": PERCENTUAL,
                           "Balanço de": st.column_config.DateColumn(format="MM/YYYY")}
COLUNAS_DA_MACRO = {"Valor": st.column_config.NumberColumn(format="%.2f"), "Mês": st.column_config.DateColumn(format="MM/YYYY")}
COLUNAS_DO_HRP = {"nome": st.column_config.TextColumn("Carteira"), "Retorno": st.column_config.NumberColumn("Retorno (R$)", format="percent"),
                  "Retorno US$": PERCENTUAL, "Vol": PERCENTUAL, "Queda máx.": PERCENTUAL, "Menor fase": PERCENTUAL, "Maior fase": PERCENTUAL,
                  "CAGR inteiro": st.column_config.NumberColumn("Retorno a.a. desde 11/2019", format="percent"),
                  "Nomes efetivos": st.column_config.NumberColumn(format="%.0f"), "Maior peso": PERCENTUAL,
                  "Giro a.a.": st.column_config.NumberColumn(format="%.1f")}
LINHAS_DO_GRAFICO = ("Composto", "Conservadora", "SPY", "B3 Composto", "B3 Conservadora", "BOVA11", "CDI",
                     "15 fundos BR, melhores 12 meses")


@dataclass(frozen=True)
class ResultadoEUA:
    curvas: pd.DataFrame        # R$ 1 at the first date, in reais
    resumo: pd.DataFrame
    portao: pd.DataFrame
    carteiras: pd.DataFrame
    cobertura: dict
    indicadores: pd.DataFrame | None = None     # one row per stock of the last universe (``scripts/us_indicadores.py``)
    macro: pd.DataFrame | None = None
    anos: pd.DataFrame | None = None            # by year in reais, with BOVA11, the CDI and the Selic
    hrp: pd.DataFrame | None = None             # HRP's variants since 2026 (``scripts/hrp_us.py``)


def _milhar(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _anos(tabela: pd.DataFrame) -> None:
    st.markdown("### Ano a ano, em reais")
    st.caption("A base de preços das ações americanas vai de 11/2016 a hoje, e não chega a 2010. Os 36 primeiros meses "
               "treinam as regras, então o resultado das estratégias começa em 11/2019 e 2020 é o primeiro ano inteiro; "
               "os anos anteriores ficam vazios em vez de estimados. As estratégias pagam 10 bps por lado, antes de IR. BOVA11, CDI e Selic vão até a mesma data; CDI e Selic são as taxas "
               "diárias do Banco Central acumuladas, antes de IR. Só a primeira data de rebalanceio.")
    st.dataframe(tabela, width="stretch", hide_index=True,
                 column_config={nome: PERCENTUAL for nome in tabela.columns[1:]},
                 alt="Retorno de cada ano em reais: estratégias americanas, BOVA11, CDI e Selic")


def _hrp(tabela: pd.DataFrame) -> None:
    st.markdown("### HRP e variantes desde 2026, em reais")
    st.caption("Todas as linhas nas mesmas datas, do último fechamento de 2025 ao último mês da base americana. As estratégias americanas pagam 10 bps "
               "por lado; tudo antes de IR. Leitura descritiva sobre a base incompleta: nove meses e cerca de três rebalanceios não decidem nada. "
               "\"Menor fase\" e \"Maior fase\" mostram quanto o resultado muda só por deslocar a data do rebalanceio em 1 ou 2 meses.")
    st.dataframe(tabela, width="stretch", hide_index=True, column_config=COLUNAS_DO_HRP,
                 alt="Retorno desde 2026 em reais de cada variante de HRP, do Composto, da Conservadora, do BOVA11 e do CDI")


def _indicadores(tabela: pd.DataFrame) -> None:
    st.markdown("### Indicadores por ação")
    st.caption(f"As {_milhar(len(tabela))} ações do universo atual, das maiores às menores. Múltiplos dos balanços do EDGAR, "
               "como conhecidos hoje; risco de retornos mensais, com beta contra o mercado do Ken French. Vazio é falta de "
               "dado: valor de mercado que não bate com o site ou com o volume negociado fica de fora. Informação, não "
               "recomendação.")
    setor = st.selectbox("Setor", [TODOS_OS_SETORES, *sorted(tabela["Setor"].dropna().unique())], key="eua_setor")
    vistas = tabela if setor == TODOS_OS_SETORES else tabela[tabela["Setor"] == setor]
    st.dataframe(vistas, width="stretch", hide_index=True, column_config=COLUNAS_DOS_INDICADORES,
                 alt="Valor de mercado, retorno, risco e múltiplos de cada ação americana do universo")


def _macro(tabela: pd.DataFrame) -> None:
    st.markdown("### Macro dos Estados Unidos")
    st.caption("Última leitura de cada série do FRED e os fatores do Ken French em 12 meses.")
    st.dataframe(tabela, width="stretch", hide_index=True, column_config=COLUNAS_DA_MACRO,
                 alt="Última leitura de juros, inflação, desemprego, volatilidade, crédito e fatores nos Estados Unidos")


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
    if resultado.anos is not None:
        _anos(resultado.anos)
    if resultado.hrp is not None:
        _hrp(resultado.hrp)
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
    if resultado.indicadores is not None:
        _indicadores(resultado.indicadores)
    if resultado.macro is not None:
        _macro(resultado.macro)
