"""
simulador.py - Screen of the fictitious account (``core.simulador``). UI strings are Portuguese, like the rest of the
app. It shows a measured past and promises nothing: two weeks of returns are noise.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.simulador import CAPITAL_ACOES, CAPITAL_OPCOES, INICIO, Painel, quedas, retornos_recentes

CARTEIRA_PRINCIPAL = "Composto"
PERCENTUAL = st.column_config.NumberColumn(format="percent")
COLUNAS_DAS_POSICOES = {"Qtd.": st.column_config.NumberColumn(format="%d"),
                        "Fechamento na troca (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
                        "Valor (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
                        "Desde a troca": PERCENTUAL, "Peso": PERCENTUAL}


def _reais(valor: float, casas: int = 2) -> str:
    return "R$ " + f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _sinal(valor: float) -> str:
    return f"{valor:+.2%}".replace(".", ",")


def secao(painel: Painel | None) -> None:
    st.markdown("## Simulador")
    if painel is None:
        st.info(f"A conta fictícia começa no fechamento de {INICIO:%d/%m/%Y}: {_reais(CAPITAL_ACOES, 0)} no Composto e "
                f"{_reais(CAPITAL_OPCOES, 0)} no CDI, com BOVA11, IVVB11 e CDI ao lado. A partir do dia seguinte, esta "
                "seção mostra o resultado.")
        return
    st.caption(f"Conta fictícia, sem dinheiro real: {_reais(CAPITAL_ACOES, 0)} seguem o último Composto registrado no "
               f"paper-v3 e {_reais(CAPITAL_OPCOES, 0)} (opções) ficam no CDI até o teste delas passar. A Conservadora é a "
               "linha de controle, com o mesmo capital. Custos incluídos; IR zero enquanto as vendas do mês ficam abaixo "
               "de R$ 20 mil. Duas semanas de retorno são ruído.")
    if not painel.com_cdi:
        st.warning("Série do CDI indisponível agora: o caixa e a linha do CDI aparecem sem rendimento.")
    curvas = painel.curvas
    retornos = retornos_recentes(curvas)
    principal = retornos.loc[CARTEIRA_PRINCIPAL]
    colunas = st.columns(4)
    colunas[0].metric(f"Patrimônio do {CARTEIRA_PRINCIPAL}", _reais(curvas[CARTEIRA_PRINCIPAL].iloc[-1]),
                      _sinal(principal["Desde o início"]))
    for coluna, periodo in zip(colunas[1:], ("Dia", "Semana", "Mês"), strict=True):
        coluna.metric(periodo, _sinal(principal[periodo]))
    st.dataframe(retornos, width="stretch", column_config=dict.fromkeys(retornos.columns, PERCENTUAL),
                 alt="Retorno do dia, da semana, do mês e desde o início, e queda máxima de cada curva")
    grafico, queda = st.columns(2)
    grafico.line_chart(curvas, y_label="R$")
    queda.line_chart(quedas(curvas), y_label="Queda desde o pico")
    for carteira, tabela in painel.posicoes.items():
        st.markdown(f"**Posições: {carteira}**")
        st.dataframe(tabela, width="stretch", column_config=COLUNAS_DAS_POSICOES,
                     alt=f"Ações, caixa e reserva de opções da carteira {carteira} no último fechamento")
    st.caption(f"Último fechamento: {pd.Timestamp(curvas.index[-1]):%d/%m/%Y}.")
