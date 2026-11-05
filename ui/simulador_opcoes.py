"""
simulador_opcoes.py - Screen of the fictitious options account (``core.simulador_opcoes``, DECISOES 259). UI strings
are Portuguese, like the rest of the app. It shows a measured past and promises nothing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.simulador import quedas, retornos_recentes
from core.simulador_opcoes import CAPITAL, INICIO, ORCAMENTO, PainelOpcoes
from ui.filtros import filtrar

TITULO = "## Conta de opções (fictícia)"
REAIS = st.column_config.NumberColumn(format="R$ %.2f")
PERCENTUAL = st.column_config.NumberColumn(format="percent")
REGRA = (f"{ORCAMENTO:.0%} da conta por mês, no primeiro pregão: metade numa put perto de 0,95 × o preço da ação mais "
         "esticada perto da máxima de 1 ano (sua ideia A), metade numa call perto de 1,05 × o preço da vencedora mais "
         "volátil (sua ideia B), cerca de 1 mês até o vencimento, compradas pelo preço de venda em lotes de 100 e "
         "levadas até o vencimento. Só compra de opções: a perda máxima é o prêmio pago. O resto fica na Selic. "
         "Custos e IR de 15% sobre o lucro incluídos.")


def _reais(valor: float) -> str:
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def secao(painel: PainelOpcoes | None) -> None:
    st.markdown(TITULO)
    if painel is None:
        st.info(f"Começa no fechamento de {INICIO:%d/%m/%Y}: {_reais(CAPITAL)} fictícios só para opções, separados "
                f"das contas de ações. {REGRA}")
        return
    st.caption(f"Conta fictícia de {_reais(CAPITAL)} desde {INICIO:%d/%m/%Y}. {REGRA} Comprar opções fora do dinheiro "
               "costuma perder na média: a leitura é depois de 12 ciclos, contra a Selic (DECISOES 259).")
    curvas = painel.curvas
    retornos = retornos_recentes(curvas)
    colunas = st.columns(2)
    colunas[0].metric("Patrimônio da conta de opções", _reais(curvas["Opções"].iloc[-1]),
                      f"{retornos.loc['Opções', 'Desde o início']:+.2%}".replace(".", ","))
    colunas[1].metric("Mesmo valor na Selic", _reais(curvas["Selic"].iloc[-1]))
    st.dataframe(filtrar(retornos.rename_axis("Curva"), "retornos_opcoes"), width="stretch",
                 column_config=dict.fromkeys(retornos.columns, PERCENTUAL),
                 alt="Retorno do dia, da semana, do mês e desde o início, e queda máxima da conta de opções e da Selic")
    grafico, queda = st.columns(2)
    grafico.line_chart(curvas, y_label="R$")
    queda.line_chart(quedas(curvas), y_label="Queda desde o pico")
    st.markdown("**Opções em aberto**")
    if painel.abertas.empty:
        st.caption("Nenhuma opção em aberto: tudo na Selic até o próximo ciclo.")
    else:
        st.dataframe(filtrar(painel.abertas, "opcoes_abertas"), width="stretch", hide_index=True,
                     column_config={"Prêmio (R$)": REAIS, "Valor (R$)": REAIS, "Strike": REAIS,
                                    "Qtd.": st.column_config.NumberColumn(format="%d")},
                     alt="Opções compradas que ainda não venceram, com o valor na última marcação")
    st.markdown("**Ordens de cada ciclo**")
    ordens = painel.ordens.assign(Data=pd.to_datetime(painel.ordens["Data"]).dt.strftime("%d/%m/%Y"))
    st.dataframe(filtrar(ordens, "opcoes_ordens"), width="stretch", hide_index=True,
                 column_config={"Prêmio (R$)": REAIS, "Custo (R$)": REAIS, "Strike": REAIS,
                                "Qtd.": st.column_config.NumberColumn(format="%d")},
                 alt="Cada ciclo: a opção comprada por ideia, ou o motivo de não ter comprado")
    st.caption(f"Último fechamento: {pd.Timestamp(curvas.index[-1]):%d/%m/%Y}.")
