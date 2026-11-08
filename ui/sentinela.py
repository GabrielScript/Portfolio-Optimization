"""
sentinela.py - Screen of the Sentinela (``core.sentinela``, DECISOES 262-272 and the build's): the DeepSeek watch
agent over the Composite's names. UI strings are Portuguese, like the rest of the app. It shows a measured past and
promises nothing; the agent acts only in a fictitious account.
"""

from __future__ import annotations

import streamlit as st

from core.sentinela import conta
from core.sentinela.tela import PainelSentinela
from core.simulador import quedas, retornos_recentes
from ui import formato, graficos
from ui.filtros import filtrar

TITULO = "## Sentinela (agente DeepSeek, conta fictícia)"
REGRA = ("A cada hora, das 8h às 19h nos pregões, o código lê as fontes oficiais (CVM, inclusive os documentos do dia, "
         "preços, volume e aluguel de ações da B3, BCB, IBGE e Tesouro). Só quando uma ação da Composite tem item "
         "novo o DeepSeek lê o pacote dela em esforço baixo e diz se algo a compromete gravemente; cada citação é "
         "conferida, letra por letra, contra o texto da fonte. Um caso grave vai ao esforço máximo, que só age com "
         "confiança de 70% ou mais e só na conta fictícia \"Composite + Sentinela\" (R$ 10.000), vendendo no fechamento. "
         "A Composite sem o agente corre ao lado, com o mesmo dinheiro, para medir o que ele acrescenta ou custa.")
LEITURA = ("Leitura única no primeiro pregão a partir de 08/01/2027: se as decisões de 70% ou mais acertarem menos de "
           "55% das vezes, ou se o Brier delas for pior que o da taxa-base, o agente perde a autonomia e passa a só "
           "mandar alertas.")


def _reais(valor: float) -> str:
    return "R$ " + f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def dolares(valor: float) -> str:
    return "US$ " + formato.decimal(valor)


def descartadas_em_texto(fracao: float | None) -> str:
    return formato.percentual(fracao, casas=1)


def secao(painel: PainelSentinela | None) -> None:
    st.markdown(TITULO)
    if painel is None:
        st.caption(f"Ainda não rodou. {REGRA}")
        return
    st.caption(f"{REGRA} {LEITURA}")
    colunas = st.columns(3)
    colunas[0].metric("Chamadas ao DeepSeek", f"{painel.chamadas}", help="Varreduras e decisões desde o início")
    colunas[1].metric("Custo dos modelos", dolares(painel.custo_usd))
    descartadas = descartadas_em_texto(painel.descartadas)
    colunas[2].metric("Respostas descartadas na checagem", descartadas, help="Meta abaixo de 5%; acima de 10% o agente para")
    if painel.curvas is not None:
        curvas = painel.curvas
        retornos = retornos_recentes(curvas)
        metricas = st.columns(2)
        for coluna, nome in zip(metricas, (conta.CONTA, conta.BASE)):
            coluna.metric(nome, _reais(curvas[nome].iloc[-1]),
                          f"{retornos.loc[nome, 'Desde o início']:+.2%}".replace(".", ","))
        texto = graficos.percentuais_em_texto(retornos)
        st.dataframe(filtrar(texto.rename_axis("Curva"), "retornos_sentinela"), width="stretch",
                     column_config=graficos.colunas_de_texto(texto),
                     alt="Retornos e queda máxima da Composite com e sem a Sentinela e da Selic")
        grafico, queda = st.columns(2)
        grafico.altair_chart(graficos.curvas_em_reais(curvas))
        queda.altair_chart(graficos.quedas_em_percentual(quedas(curvas)))
    else:
        st.caption(f"A conta fictícia começa no primeiro fechamento com o agente no ar ({_reais(conta.CAPITAL)}).")
    st.markdown("**Composto 12/3 + Sentinela (conta fictícia)**")
    if painel.curvas_12_3 is not None:
        curvas = painel.curvas_12_3
        retornos = retornos_recentes(curvas)
        metricas = st.columns(2)
        for coluna, nome in zip(metricas, (conta.CONTA_12_3, conta.BASE_12_3)):
            coluna.metric(nome, _reais(curvas[nome].iloc[-1]),
                          f"{retornos.loc[nome, 'Desde o início']:+.2%}".replace(".", ","))
        grafico, queda = st.columns(2)
        grafico.altair_chart(graficos.curvas_em_reais(curvas))
        queda.altair_chart(graficos.quedas_em_percentual(quedas(curvas)))
    else:
        st.caption(f"Começa no fechamento de {conta.INICIO_12_3:%d/%m/%Y} com {_reais(conta.CAPITAL)}: as 12 ações e os "
                   "3 BDRs do 12/3, com as mesmas vendas da Sentinela nas ações que ela tiver; os BDRs pagam 15% de "
                   "Imposto de Renda sobre o ganho.")
    st.markdown("**Alertas (grave e atenção)**")
    if painel.alertas.empty:
        st.caption("Nenhum alerta até agora.")
    else:
        st.dataframe(filtrar(painel.alertas, "alertas_sentinela"), width="stretch", hide_index=True,
                     alt="Alertas da Sentinela, do mais novo ao mais antigo, com o resumo conferido")
    st.markdown("**Ordens do agente (conta fictícia)**")
    if painel.acoes.empty:
        st.caption("Nenhuma ordem até agora.")
    else:
        estilo, config = formato.br(filtrar(painel.acoes, "ordens_sentinela"), {"Confiança": formato.percentual})
        st.dataframe(estilo, width="stretch", hide_index=True, column_config=config,
                     alt="Cada decisão de vender ou reduzir, com a confiança e se foi aceita, recusada ou executada")
    autonomia = painel.autonomia
    if autonomia.n:
        st.caption(f"Pontuação até agora ({autonomia.n} decisão(ões) com resultado de 20 pregões): Brier "
                   f"{autonomia.brier:.3f} contra {autonomia.brier_base:.3f} da taxa-base.".replace(".", ","))
    st.caption(f"Última rodada: {painel.ultima_rodada:%d/%m/%Y %H:%M}.")
