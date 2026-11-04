"""
simulador.py - Screen of the fictitious account (``core.simulador``). UI strings are Portuguese, like the rest of the
app. It shows a measured past and promises nothing: two weeks of returns are noise.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.simulador import CAPITAL_ACOES, INICIO, Painel, quedas, retornos_recentes
from ui.filtros import filtrar

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


SINAIS = {"favoravel": "🟢 compra", "neutra": "🟡 fora", "evidencia_insuficiente": "🟡 fora (evidência insuficiente)",
          "desfavoravel": "🔴 fora"}
SEM_PARECER = "🟡 fora (sem parecer)"
ORDEM_DOS_SINAIS = {"🟢": 0, "🟡": 1, "🔴": 2}


def tabela_do_parecer(registro: dict) -> pd.DataFrame:
    """Each name the DeepSeek opinion judged (both rule portfolios' since 2026-11-04) (``deepseek/pareceres/*.json``) with its light, green first."""
    acoes = {acao["ticker"]: acao for acao in registro["parecer"]["acoes"]}
    linhas = [(ticker, SINAIS.get(acoes[ticker]["postura"], SEM_PARECER) if ticker in acoes else SEM_PARECER,
               acoes[ticker]["confianca"] if ticker in acoes else None) for ticker in registro["universo"]]
    linhas.sort(key=lambda linha: ORDEM_DOS_SINAIS[linha[1][0]])
    return pd.DataFrame(linhas, columns=["Ticker", "Parecer do DeepSeek", "Confiança"])


def parecer_do_deepseek(registro: dict | None) -> None:
    """The DeepSeek portfolio's last opinion: which names it holds (green) and which it leaves out."""
    st.markdown("**Carteira do DeepSeek: último parecer**")
    if registro is None:
        st.caption("O DeepSeek ainda não deu parecer sobre as ações da Composite e da Conservadora.")
        return
    sobre = (f"as {len(registro['universo'])} ações da Composite e da Conservadora (cada uma uma vez)"
             if "base_da_conservadora" in registro else "a Composite")
    st.caption(f"Parecer de {pd.Timestamp(registro['quando']):%d/%m/%Y %H:%M} sobre {sobre} de "
               f"{pd.Timestamp(registro['base_do_composto']):%d/%m/%Y} ({registro['modelo']}, esforço "
               f"{registro['esforco']}): a carteira compra só as verdes, em pesos iguais (até 20% por ação), e o resto "
               "fica na Selic. A carteira Controle compra a lista inteira em pesos iguais, sem o DeepSeek: a diferença "
               "entre as duas é o que o filtro do modelo acrescenta ou custa. Notícias e posts só contaram quando a CVM, o Banco Central, a B3 ou a ANBIMA confirmaram. "
               "Opinião de um modelo de linguagem, sem teste de que acerte: só o tempo mede.")
    st.dataframe(filtrar(tabela_do_parecer(registro), "parecer_deepseek"), width="stretch", hide_index=True,
                 column_config={"Confiança": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                                             format="percent")},
                 alt="Sinal do DeepSeek para cada ação julgada: verde compra, amarelo e vermelho ficam de fora")


def secao(painel: Painel | None) -> None:
    st.markdown("## Simulador")
    if painel is None:
        st.info(f"Começa no fechamento de {INICIO:%d/%m/%Y}: {_reais(CAPITAL_ACOES, 0)} em ações em cada carteira "
                "(Composite, Conservadora, DeepSeek e Controle).")
        return
    st.caption(f"Conta fictícia: {_reais(CAPITAL_ACOES, 0)} em ações em cada carteira desde {INICIO:%d/%m/%Y}, com "
               "custos. BOVA11 e Selic são só referência.")
    if not painel.com_cdi:
        st.warning("Série da Selic indisponível agora: o caixa e a linha da Selic aparecem sem rendimento.")
    curvas = painel.curvas
    retornos = retornos_recentes(curvas)
    principal = retornos.loc[CARTEIRA_PRINCIPAL]
    colunas = st.columns(4)
    colunas[0].metric(f"Patrimônio do {CARTEIRA_PRINCIPAL}", _reais(curvas[CARTEIRA_PRINCIPAL].iloc[-1]),
                      _sinal(principal["Desde o início"]))
    for coluna, periodo in zip(colunas[1:], ("Dia", "Semana", "Mês"), strict=True):
        coluna.metric(periodo, _sinal(principal[periodo]))
    st.dataframe(filtrar(retornos.rename_axis("Carteira"), "retornos_simulador"), width="stretch",
                 column_config=dict.fromkeys(retornos.columns, PERCENTUAL),
                 alt="Retorno do dia, da semana, do mês e desde o início, e queda máxima de cada curva")
    grafico, queda = st.columns(2)
    grafico.line_chart(curvas, y_label="R$")
    queda.line_chart(quedas(curvas), y_label="Queda desde o pico")
    for carteira, tabela in painel.posicoes.items():
        st.markdown(f"**Posições: {carteira}**")
        st.dataframe(filtrar(tabela.rename_axis("Ativo"), f"posicoes_{carteira}"), width="stretch",
                     column_config=COLUNAS_DAS_POSICOES,
                     alt=f"Ações, caixa e reserva de opções da carteira {carteira} no último fechamento")
    st.caption(f"Último fechamento: {pd.Timestamp(curvas.index[-1]):%d/%m/%Y}.")
