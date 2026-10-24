"""
alepha.py - Screen blocks of the Alepha plan: the Conservadora registered in paper-v2 as the real-money plan
(the user's decision of 2026-10-23, DECISOES 117), from the current positions (``core.rebalanceio``).

Advice boundary (prediction-market-risk-review): backtest numbers appear as measured past, with the live proof still
to come; nothing is promised in R$. UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from core.impostos import ISENCAO_DE_VENDAS
from core.periodos import medidas_por_periodo, retornos_por_ano
from core.rebalanceio import SITUACAO_PULAR, MesFiscal, Rebalanceio

AVISO_CVM = ("Alepha é o nome de uma ferramenta pessoal de estudo e decisão: não é um fundo de investimento "
             "oferecido a terceiros nem recomendação de investimento (fundos: Resolução CVM 175/2022; análise de "
             "valores mobiliários: Resolução CVM 20/2021).")

COLUNAS_ORDENS = {
    "Cotação (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
    "Qtd.": st.column_config.NumberColumn(format="%d", help="Ações a comprar ou vender (lote padrão + fracionário)"),
    "Lote padrão": st.column_config.NumberColumn(format="%d", help="Múltiplos de 100 ações"),
    "Fracionário": st.column_config.NumberColumn(format="%d", help="1 a 99 ações, ticker com F (ex.: PETR4F)"),
    "Valor (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
    "Custo (R$)": st.column_config.NumberColumn(format="R$ %.2f", help="Emolumentos da B3 + meio spread + impacto"),
    "Ganho realizado (R$)": st.column_config.NumberColumn(
        format="R$ %.2f", help="Venda: valor − custo − quantidade × preço médio"),
    "IR marginal (R$)": st.column_config.NumberColumn(
        format="R$ %.2f", help="IR do mês com esta venda menos sem ela: passar de R$ 20 mil de vendas no mês tributa "
                               "o ganho do mês inteiro"),
    "Ganho esperado (R$)": st.column_config.NumberColumn(
        format="R$ %.4f", help="Redução de risco do reajuste (zona sem negociação, só informativa)"),
    "Peso final": st.column_config.NumberColumn(format="percent", help="Depois das ordens ÷ patrimônio"),
}


def aviso_conservadora(data: date) -> str:
    return (f"Alepha: a Conservadora (15 ações de baixa volatilidade, dividendos e momentum, pesos iguais, até 4 "
            f"por setor), registrada no paper-v2 em {data:%d/%m/%Y}. No backtest sem viés de sobrevivência (2010–2026, "
            "3 fases), 16% a 17% ao ano contra 7% do BOVA11, com volatilidade de 18% contra 23% e queda máxima de −40% "
            "contra −50%; depois de custos e IR, com R$ 10 mil, P[> BOVA11] de 0,98 a 0,995. Passado não garante "
            "futuro: a prova ao vivo é o paper-v2 (leitura em 126 pregões, ~27/04/2027; decisão em 252, ~25/10/2027). "
            "Só dinheiro de 5 anos ou mais; a carteira muda a cada 63 pregões, quando o paper-v2 registra a nova.")


def aviso_capital_pequeno(patrimonio: float, minimo: float, um_de_cada: float, nomes: int, esperando: list[str],
                          quantidade_minima: int, peso_maximo: float, nucleo: str) -> str:
    """The small-capital plan (DECISOES 124): what it does and how far it is from the tested portfolio."""
    texto = (f"Capital pequeno: com R$ {patrimonio:,.2f} (abaixo dos R$ {minimo:,.2f} que os {nomes} nomes com "
             f"{quantidade_minima} ações e pesos perto de iguais pedem), o plano mantém o {nucleo} que você já tem e "
             "usa o dinheiro novo para comprar as ações da Conservadora uma a uma no fracionário, primeiro as mais "
             f"abaixo do peso igual, cada ação com até {peso_maximo:.0%} de tudo o que você tem ({nucleo} incluído). "
             f"A cada aporte o plano completa os nomes que faltam: 1 ação de cada um dos {nomes} custa hoje cerca de "
             f"R$ {um_de_cada:,.2f}. No fracionário o spread costuma ser maior que no lote padrão: o custo mostrado é "
             "um piso.")
    if esperando:
        texto += f" Esperam mais capital: {', '.join(esperando)}."
    return texto


def editor_de_posicoes() -> pd.DataFrame:
    """The table of the positions the user already holds (empty = buy from zero)."""
    vazia = pd.DataFrame({"ticker": pd.Series(dtype="str"), "quantidade": pd.Series(dtype="Int64"),
                          "preco_medio": pd.Series(dtype="float")})
    with st.expander("Minhas posições (para rebalancear a partir delas)"):
        return st.data_editor(vazia, num_rows="dynamic", hide_index=True, key="posicoes", width="stretch",
                              column_config={
                                  "ticker": st.column_config.TextColumn("Ticker", help="Ex.: PETR4", required=True),
                                  "quantidade": st.column_config.NumberColumn("Quantidade", min_value=0, step=1,
                                                                              format="%d"),
                                  "preco_medio": st.column_config.NumberColumn(
                                      "Preço médio (R$)", min_value=0.0, format="R$ %.2f",
                                      help="O da nota de corretagem (custo médio para o IR)")})


def mes_fiscal() -> MesFiscal:
    """What the user already sold this month outside this plan: the exemption adds all of the person's stock
    sales."""
    with st.expander("IR do mês (vendas de ações já feitas fora deste plano)"):
        c1, c2, c3 = st.columns(3)
        vendas = c1.number_input("Vendas de ações no mês (R$)", min_value=0.0, value=0.0, step=100.0,
                                 key="vendas_do_mes")
        resultado = c2.number_input("Ganho ou perda dessas vendas (R$)", value=0.0, step=100.0, key="resultado_do_mes")
        prejuizo = c3.number_input("Prejuízo acumulado a compensar (R$)", min_value=0.0, value=0.0, step=100.0,
                                   key="prejuizo_acumulado")
    return MesFiscal(float(vendas), float(resultado), float(prejuizo))


def tabela_rebalanceio(plano: Rebalanceio, zona: Rebalanceio | None, patrimonio: float, mes_da_liquidez: str | None,
                       custo_bps: float) -> None:
    """The orders (buy, sell, keep), the totals and the notes on cost, IR and the zone."""
    ordens = plano.ordens
    tabela = pd.DataFrame({
        "Ticker": [ticker.removesuffix(".SA") for ticker in ordens.index],
        "Ordem": ordens["lado"].replace("", "manter").to_numpy(),
        "Qtd.": ordens["delta"].abs().to_numpy(), "Lote padrão": ordens["lote_padrao"].to_numpy(),
        "Fracionário": ordens["fracionario"].to_numpy(), "Cotação (R$)": ordens["cotacao"].to_numpy(),
        "Valor (R$)": ordens["valor"].to_numpy(), "Custo (R$)": ordens["custo"].to_numpy(),
        "Ganho realizado (R$)": ordens["ganho_realizado"].to_numpy(), "IR marginal (R$)": ordens["ir_marginal"].to_numpy(),
        "Peso final": (ordens["alvo"] * ordens["cotacao"] / patrimonio).to_numpy()})
    if zona is not None:
        tabela["Ganho esperado (R$)"] = zona.ordens["ganho_esperado"].reindex(ordens.index).to_numpy()
        tabela["Zona"] = ["pularia" if situacao == SITUACAO_PULAR else ""
                          for situacao in zona.ordens["situacao"].reindex(ordens.index).fillna("")]
    st.dataframe(tabela, width="stretch", hide_index=True, column_config=COLUNAS_ORDENS,
                 alt="Plano do Alepha: ordens de compra e venda a partir das posições atuais")
    vendas = ordens["delta"] < 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patrimônio", f"R$ {patrimonio:,.2f}", border=True)
    c2.metric("Compras / vendas", f"R$ {ordens.loc[~vendas, 'valor'].sum():,.2f}",
              f"vendas R$ {ordens.loc[vendas, 'valor'].sum():,.2f}", delta_color="off", delta_arrow="off", border=True)
    c3.metric("Custo estimado", f"R$ {ordens['custo'].sum():,.2f}", border=True)
    c4.metric("IR do mês (DARF)", f"R$ {plano.ir_do_mes:,.2f}", f"caixa final R$ {plano.caixa_final:,.2f}",
              delta_color="off", delta_arrow="off", border=True)
    excesso = plano.vendas_de_acoes_no_mes - ISENCAO_DE_VENDAS
    if excesso > 0 and plano.ir_do_mes > 0:
        st.warning(f"As vendas de ações do mês somam R\\$ {plano.vendas_de_acoes_no_mes:,.2f}, acima de R\\$ 20 mil: o "
                   f"ganho do mês inteiro paga 15%. Adiar R\\$ {excesso:,.2f} de vendas para o mês seguinte mantém a "
                   "isenção (as compras que dependem delas esperam junto).")
    fonte = (f"liquidez de {mes_da_liquidez} do lago (emolumentos de 0,030% + meio spread + impacto pela raiz do "
             f"tamanho, o modelo da prova final); papel sem medida: {custo_bps:.0f} bps" if mes_da_liquidez else
             f"{custo_bps:.0f} bps por lado (sem a foto de liquidez do lago)")
    st.caption(f"Custo: {fonte}. IR: 15% do ganho líquido do mês; vendas de ações até R$ 20 mil no mês são isentas "
               "(Lei 11.033/2004, art. 3º, I); ETF não tem isenção. Lote padrão = múltiplos de 100; o resto no "
               "fracionário (ticker terminado em F). Confira o preço no book antes de enviar a ordem.")
    if zona is not None:
        st.caption("Zona sem negociação: só informativa. O teste pré-registrado de 23/10/2026 "
                   "(docs/zona-2026-10-23.txt) não passou: pular os reajustes cortou o custo, mas o retorno caiu numa "
                   "das 3 fases com R\\$ 10 mil e em todas com R\\$ 100 mil. O plano manda todas as ordens.")


def _rotulo_do_ano(ano: int, datas: pd.DatetimeIndex) -> str:
    do_ano = datas[datas.year == ano]
    if ano == datas[0].year and do_ano[0].month > 1:
        return f"{ano} (desde {do_ano[0]:%d/%m})"
    if ano == datas[-1].year and do_ano[-1].month < 12:
        return f"{ano} (até {do_ano[-1]:%d/%m})"
    return str(ano)


def resultados_da_conservadora(quadro: pd.DataFrame | None) -> None:
    """The frozen backtest of the Conservadora against BOVA11 (DECISOES 122): whole period, last 5 and 3 years, and
    each calendar year, the bad ones included. Phase 0; the whole period's CAGR range covers the 3 phases."""
    st.markdown("### A Conservadora no backtest: por período e por ano")
    if quadro is None:
        st.caption("Sem as curvas do backtest (docs/dados/conservadora-curvas-2026-10-23.parquet).")
        return
    fases = {fase: parte.set_index("date")[["Conservadora", "BOVA11"]] for fase, parte in quadro.groupby("fase")}
    curvas = fases[min(fases)]
    medidas = medidas_por_periodo(curvas)
    tabela = pd.DataFrame({
        "Período": medidas.index,
        "De": [f"{data:%d/%m/%Y}" for data in medidas[("Início", "")]],
        "Até": [f"{data:%d/%m/%Y}" for data in medidas[("Fim", "")]],
        **{f"{medida} {nome}": medidas[(nome, coluna)].astype(float).to_numpy()
           for medida, coluna in (("Retorno a.a.", "CAGR"), ("Vol", "Vol anual"), ("Queda máx.", "MDD"))
           for nome in ("Conservadora", "BOVA11")}})
    st.dataframe(tabela, width="stretch", hide_index=True,
                 column_config={coluna: st.column_config.NumberColumn(format="percent")
                                for coluna in tabela.columns[3:]},
                 alt="Conservadora contra BOVA11 no período inteiro e nos últimos 5 e 3 anos")
    anos = retornos_por_ano(curvas)
    por_ano = pd.DataFrame({"Ano": [_rotulo_do_ano(ano, curvas.index) for ano in anos.index],
                            "Conservadora": anos["Conservadora"].to_numpy(), "BOVA11": anos["BOVA11"].to_numpy(),
                            "Diferença": (anos["Conservadora"] - anos["BOVA11"]).to_numpy()})
    c1, c2 = st.columns([2, 3])
    c1.dataframe(por_ano, width="stretch", hide_index=True,
                 column_config={coluna: st.column_config.NumberColumn(format="percent")
                                for coluna in ("Conservadora", "BOVA11", "Diferença")},
                 alt="Retorno de cada ano civil da Conservadora e do BOVA11")
    c2.bar_chart(por_ano.set_index("Ano")[["Conservadora", "BOVA11"]], stack=False, y_label="Retorno no ano",
                 color=["#2e7d32", "#9e9e9e"])
    inteiro = [float(medidas_por_periodo(parte, anos=()).iloc[0][("Conservadora", "CAGR")]) for parte in fases.values()]
    st.caption(f"Backtest sem viés de sobrevivência no lago, antes de custos e IR (motor do pré-registro, entradas "
               f"congeladas de 22/10/2026, preços até {curvas.index[-1]:%d/%m/%Y}); fase 0 da tabela. No período "
               f"inteiro, as 3 fases de rebalanceio dão {min(inteiro):.1%} a {max(inteiro):.1%} ao ano. Os anos ruins "
               "ficam na tabela: a vantagem do período inteiro vem quase toda de 2010–2022; nos últimos 3 anos o "
               "BOVA11 rendeu mais, com volatilidade parecida. Passado não garante futuro; a prova ao vivo é o "
               "paper-v2 (decisão em 252 pregões, ~25/10/2027).")
