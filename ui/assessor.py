"""
assessor.py - Screen block of the DeepSeek adviser (``core.assessor``, DECISOES 136): one button, one opinion on the
Alepha plan's stocks, each claim shown with the evidence it quotes, plus what the check dropped and what it cost.
Information only: nothing here reaches the weights or the orders. UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from core.assessor import Evidencia, Resultado

ROTULO_DA_POSTURA = {"favoravel": "🟢 favorável", "neutra": "🟡 neutra", "desfavoravel": "🔴 desfavorável",
                     "evidencia_insuficiente": "⚪ evidência insuficiente"}
CHAVE_DO_PARECER = "parecer_do_assessor"
_MARKDOWN = set("\\`*_{}[]()<>#+-.!|~$:")


def sem_markdown(texto: str) -> str:
    """``texto`` (the model's words, quotes of web pages) shown as plain text inside Markdown: every Markdown control
    character escaped, so injected text cannot become a link, an image or a format."""
    return "".join(f"\\{letra}" if letra in _MARKDOWN else letra for letra in texto)


def _virgula(numero: float, casas: int) -> str:
    return f"{numero:.{casas}f}".replace(".", ",")


def _milhar(numero: int) -> str:
    return f"{numero:,}".replace(",", ".")


def _afirmacoes(lista: list[dict]) -> str:
    return " · ".join(f"{afirmacao['afirmacao']} [{afirmacao['evidencia']}]" for afirmacao in lista)


def _fonte(afirmacao: dict, evidencias: dict[str, Evidencia]) -> str:
    evidencia = evidencias.get(afirmacao["evidencia"])
    origem = f"{evidencia.classe}, {evidencia.data}" if evidencia else "?"
    return (f"- {sem_markdown(afirmacao['afirmacao'])} — *{sem_markdown(afirmacao['evidencia'])} "
            f"({sem_markdown(origem)}): “{sem_markdown(afirmacao['citacao'])}”*")


def secao(indisponivel: str | None, universo: list[str], gerar: Callable[[], Resultado]) -> None:
    """The adviser's block; ``indisponivel`` is the reason it cannot run here (no key, no registered plan)."""
    st.markdown("### Assessor do Alepha (DeepSeek)")
    if indisponivel:
        st.caption(f"Assessor indisponível nesta versão do app: {indisponivel}.")
        return
    st.caption("O DeepSeek V4.1 Flash lê as ações do plano, o Banco Central, os fatos da CVM e as manchetes, pode "
               "buscar notícias, posts e páginas, e dá um parecer imparcial para o investidor arrojado. Cada afirmação "
               "cita uma evidência; o app confere a citação e os números e descarta o que não confere. Antes da "
               "postura, três razões macro do Banco Central contra a tese.")
    if st.button("Gerar parecer", key="gerar_parecer", icon=":material/psychology:"):
        with st.spinner("O assessor está lendo as evidências (até alguns minutos)…"):
            st.session_state[CHAVE_DO_PARECER] = gerar()
    resultado = st.session_state.get(CHAVE_DO_PARECER)
    if resultado is not None:
        _mostrar(resultado, universo)


def _mostrar(resultado: Resultado, universo: list[str]) -> None:
    uso = resultado.uso
    st.caption(f"{uso.requisicoes} pedido(s) ao DeepSeek, {_milhar(uso.entrada)} tokens de entrada (cache "
               f"{uso.taxa_de_cache:.0%}), {_milhar(uso.saida)} de saída ({_milhar(uso.raciocinio)} de raciocínio), "
               f"{len(resultado.chamadas)} ferramenta(s); custo estimado US$ {_virgula(uso.custo_usd, 4)}.")
    if resultado.validacao is None:
        st.error(f"Parecer indisponível: {sem_markdown(resultado.erro or '')}.")
        return
    parecer, evidencias = resultado.validacao.parecer, resultado.evidencias
    if parecer["resumo"]:
        st.info(sem_markdown(parecer["resumo"]))
    acoes = {acao["ticker"]: acao for acao in parecer["acoes"]}
    ordem = parecer["ranking"] + [ticker for ticker in acoes if ticker not in parecer["ranking"]]
    tabela = pd.DataFrame({"Posição": range(1, len(ordem) + 1), "Ticker": ordem,
                           "Postura": [ROTULO_DA_POSTURA[acoes[t]["postura"]] for t in ordem],
                           "Confiança": [acoes[t]["confianca"] for t in ordem],
                           "Tese": [_afirmacoes(acoes[t]["tese"]) for t in ordem],
                           "Contra a tese (Banco Central)": [_afirmacoes(acoes[t]["contra_tese_macro"]) for t in ordem],
                           "Riscos": [_afirmacoes(acoes[t]["riscos"]) for t in ordem]})
    st.dataframe(tabela, width="stretch", hide_index=True,
                 column_config={"Confiança": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                                             format="percent")},
                 alt="Parecer do assessor: postura, confiança, tese, contra-tese e riscos de cada ação do plano")
    sem_parecer = [t.removesuffix(".SA") for t in universo if t.removesuffix(".SA") not in acoes]
    if sem_parecer:
        st.caption(f"Sem parecer: {', '.join(sem_parecer)}.")
    for ticker in ordem:
        with st.expander(f"{ticker}: {ROTULO_DA_POSTURA[acoes[ticker]['postura']]} — evidências"):
            for titulo, campo in (("Tese", "tese"), ("Contra a tese (Banco Central)", "contra_tese_macro"),
                                  ("Riscos", "riscos")):
                if acoes[ticker][campo]:
                    st.markdown(f"**{titulo}**\n" + "\n".join(_fonte(a, evidencias) for a in acoes[ticker][campo]))
    if parecer["cenario_macro"]:
        st.markdown("**Cenário macro**\n" + "\n".join(_fonte(a, evidencias) for a in parecer["cenario_macro"]))
    if parecer["limitacoes"]:
        st.caption("Limitações apontadas pelo assessor: " + sem_markdown(" ".join(parecer["limitacoes"])))
    descartes = resultado.validacao.descartes
    if descartes:
        with st.expander(f"{len(descartes)} afirmação descartada pela checagem" if len(descartes) == 1 else
                         f"{len(descartes)} afirmações descartadas pela checagem"):
            st.markdown("\n".join(f"- {sem_markdown(motivo)}" for motivo in descartes))
    st.caption("Informativo: não muda os pesos nem as ordens do Alepha, e não é recomendação de investimento. "
               "Opinião de um modelo de linguagem sobre evidências conferidas, sem teste de que acerte.")
