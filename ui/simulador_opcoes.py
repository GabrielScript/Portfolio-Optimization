"""
simulador_opcoes.py - Screen of the fictitious options account (``core.simulador_opcoes``, DECISOES 259). UI strings
are Portuguese, like the rest of the app. It shows a measured past and promises nothing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.simulador import quedas, retornos_recentes
from core.simulador_opcoes import CAPITAL, DEEP, OPCOES, ORCAMENTO, Conta, PainelOpcoes
from ui import formato, graficos
from ui.filtros import filtrar

TITULO = "## Conta de opções (fictícia)"
TITULO_DEEP = "## Conta de opções deep OTM (fictícia)"
QUANTIDADE = {"Qtd.": st.column_config.NumberColumn(format="%d")}
REGRA = (f"{ORCAMENTO:.0%} da conta por mês, no primeiro pregão: metade numa put perto de 0,95 × o preço da ação mais "
         "esticada perto da máxima de 1 ano (sua ideia A), metade numa call perto de 1,05 × o preço da vencedora mais "
         "volátil (sua ideia B), cerca de 1 mês até o vencimento, compradas pelo preço de venda em lotes de 100 e "
         "levadas até o vencimento. Só compra de opções: a perda máxima é o prêmio pago. O resto fica na Selic. "
         "Custos e Imposto de Renda de 15% sobre o lucro incluídos.")
REGRA_DEEP = (f"Mesmas ações, datas e verba da conta de opções acima ({ORCAMENTO:.0%} da conta por mês, metade por "
              "ideia), só que bem fora do dinheiro: a put perto de 0,85 × o preço da ação (entre 0,80 e 0,90) e a call "
              "perto de 1,15 × (entre 1,10 e 1,20). Em cada faixa compra a opção mais negociada no dia, pelo preço de "
              "venda, nunca mais do que o volume negociado nela naquele pregão, e leva até o vencimento. A perda máxima "
              "é o prêmio pago; o resto fica na Selic. Custos e Imposto de Renda de 15% sobre o lucro incluídos.")
AVISO = ("Comprar opções fora do dinheiro costuma perder na média: a leitura é depois de 12 ciclos, contra a Selic "
         "(DECISOES 259).")
AVISO_DEEP = ("Opções muito fora do dinheiro vencem sem valor na maioria dos meses: a conta tende a sangrar e depender "
              "de poucos ganhos grandes. A leitura é depois de 12 ciclos (fim de outubro de 2027), contra a Selic e "
              "contra a conta de opções acima.")


def _reais(valor: float) -> str:
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def secao(painel: PainelOpcoes | None) -> None:
    """The first options account ("Opções", DECISOES 259)."""
    _conta(painel, OPCOES, TITULO, REGRA, AVISO, "opcoes")


def secao_deep(painel: PainelOpcoes | None) -> None:
    """The deep out-of-the-money options account (spec ``2026-11-08-opcoes-deep-otm-design.md``)."""
    _conta(painel, DEEP, TITULO_DEEP, REGRA_DEEP, AVISO_DEEP, "opcoes_deep")


def _conta(painel: PainelOpcoes | None, conta: Conta, titulo: str, regra: str, aviso: str, chave: str) -> None:
    st.markdown(titulo)
    if painel is None:
        st.caption(f"Começa no fechamento de {conta.inicio:%d/%m/%Y}: {_reais(CAPITAL)} fictícios só para opções, "
                   f"separados das contas de ações. {regra}")
        return
    st.caption(f"Conta fictícia de {_reais(CAPITAL)} desde {conta.inicio:%d/%m/%Y}. {regra} {aviso}")
    curvas = painel.curvas
    retornos = retornos_recentes(curvas)
    colunas = st.columns(2)
    colunas[0].metric(f"Patrimônio da conta de {conta.nome[0].lower()}{conta.nome[1:]}",
                      _reais(curvas[conta.nome].iloc[-1]),
                      f"{retornos.loc[conta.nome, 'Desde o início']:+.2%}".replace(".", ","))
    colunas[1].metric("Mesmo valor na Selic", _reais(curvas["Selic"].iloc[-1]))
    texto = graficos.percentuais_em_texto(retornos)
    st.dataframe(filtrar(texto.rename_axis("Curva"), f"retornos_{chave}"), width="stretch",
                 column_config=graficos.colunas_de_texto(texto),
                 alt="Retorno do dia, da semana, do mês e desde o início, e queda máxima da conta de opções e da Selic")
    grafico, queda = st.columns(2)
    grafico.altair_chart(graficos.curvas_em_reais(curvas))
    queda.altair_chart(graficos.quedas_em_percentual(quedas(curvas)))
    st.markdown("**Opções em aberto**")
    if painel.abertas.empty:
        st.caption("Nenhuma opção em aberto: tudo na Selic até o próximo ciclo.")
    else:
        estilo, config = formato.br(filtrar(painel.abertas, f"{chave}_abertas"),
                                    dict.fromkeys(("Prêmio (R$)", "Valor (R$)", "Strike"), formato.decimal))
        st.dataframe(estilo, width="stretch", hide_index=True, column_config={**config, **QUANTIDADE},
                     alt="Opções compradas que ainda não venceram, com o valor na última marcação")
    st.markdown("**Ordens de cada ciclo**")
    ordens = painel.ordens.assign(Data=pd.to_datetime(painel.ordens["Data"]).dt.strftime("%d/%m/%Y"))
    estilo, config = formato.br(filtrar(ordens, f"{chave}_ordens"),
                                dict.fromkeys(("Prêmio (R$)", "Custo (R$)", "Strike"), formato.decimal))
    st.dataframe(estilo, width="stretch", hide_index=True, column_config={**config, **QUANTIDADE},
                 alt="Cada ciclo: a opção comprada por ideia, ou o motivo de não ter comprado")
    st.caption(f"Último fechamento: {pd.Timestamp(curvas.index[-1]):%d/%m/%Y}.")
