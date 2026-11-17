"""
conta_mt5.py - Screen of the MT5 demo book (``core.conta_mt5``, spec ``2026-11-17-conta-mt5-no-app-design.md``). UI
strings are Portuguese, like the rest of the app. Fictitious money on XP's demo account, not the real buy.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.conta_mt5 import INICIAL, PainelMT5, lado_a_lado
from ui import formato, graficos
from ui.filtros import filtrar

TITULO = "## Conta demo no MetaTrader 5 (XP)"
EXPLICACAO = ("Conta demo da XP no MetaTrader 5 com R$ 100.000,00 fictícios desde 13/11/2026, seguindo a Composite: "
              "as ordens saem no call de fechamento e o valor é marcado às 21:00 pelos fechamentos do próprio "
              "MetaTrader 5. Mostra o que a execução real (lotes, fracionário, preço limite) faz com a carteira. "
              "Dinheiro fictício: não é a compra real.")
AVISO_CAIXA = ("A primeira compra gastou mais que os R$ 100.000,00 da conta. Enquanto o caixa estiver negativo, o "
               "executor recusa compras novas.")
QUANTIDADE = {"Qtd.": st.column_config.NumberColumn(format="%d")}


def _reais(valor: float) -> str:
    texto = "R$ " + f"{abs(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return "-" + texto if valor < 0 else texto


def _pct(valor: float) -> str:
    return f"{valor:+.2%}".replace(".", ",")


def secao(painel: PainelMT5 | None, composto: pd.Series | None) -> None:
    """The book's value, day, cash and the Composite simulator over the same days; ``composto``: the simulator's
    Composite account curve (None without the simulator)."""
    st.markdown(TITULO)
    if painel is None:
        st.caption(f"Ainda sem valor publicado. {EXPLICACAO}")
        return
    st.caption(EXPLICACAO)
    linha = painel.linha
    curvas = lado_a_lado(painel.curva, composto)
    colunas = st.columns(4)
    colunas[0].metric("Patrimônio no MetaTrader 5", _reais(linha.patrimonio), f"{_pct(linha.desde_inicio)} desde o início")
    colunas[1].metric("Resultado do dia", _reais(linha.dia_reais), _pct(linha.dia))
    colunas[2].metric("Caixa", _reais(painel.caixa), f"{_pct(painel.caixa / linha.patrimonio)} do patrimônio",
                      delta_color="off", delta_arrow="off")
    if "Composite (simulador)" in curvas:
        simulado = float(curvas["Composite (simulador)"].iloc[-1])
        colunas[3].metric("Composite no simulador", _reais(simulado), f"{_pct(simulado / INICIAL - 1)} no mesmo período",
                          help="A conta do simulador nos mesmos dias, partindo dos mesmos R$ 100.000,00: sem lotes, "
                               "sem preço limite e sem custo de execução.")
    else:
        colunas[3].metric("Composite no simulador", "—", "sem o simulador", delta_color="off", delta_arrow="off")
    if painel.caixa < 0:
        st.warning(AVISO_CAIXA, title="Caixa negativo", icon=":material/warning:")
    st.altair_chart(graficos.curvas_em_reais(curvas),
                    alt="Patrimônio da conta demo no MetaTrader 5 e da Composite no simulador, em reais, dia a dia")
    st.markdown("**Posições**")
    if painel.posicoes.empty:
        st.caption("Nenhuma ação na conta.")
    else:
        estilo, config = formato.br(filtrar(painel.posicoes, "mt5_posicoes"),
                                    dict.fromkeys(("Preço médio (R$)", "Custo (R$)"), formato.decimal))
        st.dataframe(estilo, width="stretch", hide_index=True, column_config={**config, **QUANTIDADE},
                     alt="Ações da conta demo: quantidade, preço médio de compra e custo")
    st.caption(f"Último fechamento marcado: {pd.Timestamp(painel.curva.index[-1]):%d/%m/%Y}.")
