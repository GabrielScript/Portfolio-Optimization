"""
app.py - Dashboard Streamlit (camada fina sobre o pacote ``core``): o Aleph Quant.
Otimizador de carteiras de ações da B3 para o investidor arrojado.
Autor: Gabriel Estrela Lopes

The page (the user's UI lists of 2026-11-03 and 2026-11-04): what Aleph Quant is; the survivorship-free backtest of the
Composite and the Conservadora; the simulator and the DeepSeek portfolio; the market panel; the DeepSeek adviser; and
the Composite next to the Markowitz efficient frontier. The buy plan from the held positions left the page on
2026-11-04 (the positions and the month's tax typed in were gone on the next visit); its engine stays in ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

from ui.marca import ICONE_PADRAO, arquivo_da_logo, arquivo_do_icone  # noqa: E402  (stdlib only: safe before set_page_config)

LOGO, ICONE = arquivo_da_logo(), arquivo_do_icone()
st.set_page_config(page_title="Aleph Quant · Alocação quantitativa B3", page_icon=str(ICONE or ICONE_PADRAO),
                   layout="wide")
if LOGO is not None:
    st.logo(str(LOGO), icon_image=str(ICONE), size="large")

from assets import BENCHMARK_ACOES, SETORES, get_all_tickers, get_ticker_info, get_tickers_by_setor  # noqa: E402
from ativos_eua import AVISO_EUA, get_ticker_info_eua, get_tickers_eua  # noqa: E402
from ativos_tela import info_curada, info_da_tela  # noqa: E402
from core.config import Objetivo, taxa_anual_para_aritmetica  # noqa: E402
from core.estimation import estimar_momentos, retornos_simples  # noqa: E402
from core import assessor, precos_do_app  # noqa: E402
from core.optimization import ErroOtimizacao, metricas_carteira, preparar_covariancia  # noqa: E402
from core.risco import LIMITES  # noqa: E402
from ui import alepha, painel, parecer, servicos  # noqa: E402
from ui import assessor as ui_assessor  # noqa: E402
from ui import eua as ui_eua  # noqa: E402
from ui import simulador as ui_simulador  # noqa: E402
from visualizations import grafico_fronteira_eficiente  # noqa: E402

# The US stocks screen (ui/eua.py, scraped US data) is on standby since the user's decision of 2026-11-03
# (DECISOES 240): B3 only. Code and data stay; the B3-listed BDRs and ETFs of ativos_eua.py stay in the sidebar.
MOSTRAR_ACOES_EUA = False
# The frontier (information only): 5 years of prices, Jorion-shrunk mu and Ledoit-Wolf covariance; its cap per asset
# goes up to 25% and its cap per sector up to 50% (user, 2026-11-04). The simulator keeps LIMITES.
ANOS_DA_FRONTEIRA = 5
TETO_POR_ATIVO_DA_FRONTEIRA = 25
TETO_POR_SETOR_DA_FRONTEIRA = 50

SOBRE = """\
**O Aleph Quant é uma ferramenta de decisão para investir dinheiro real em ações, ETFs e BDRs da B3.** Ele monta \
carteiras com regras fixas, testadas no passado sem viés de sobrevivência (as empresas que saíram da bolsa \
continuam na conta), e acompanha cada uma ao vivo.

- **Composite** (a carteira principal): as 15 ações mais bem colocadas numa nota que junta baixa volatilidade, \
dividendos, momento, valor (preço barato frente ao lucro) e crescimento do lucro. Pesos iguais, no máximo 4 por \
setor, troca a cada 3 meses.
- **Conservadora**: a mesma regra com uma nota mais simples (baixa volatilidade, dividendos e momento). Serve de \
comparação e de base.
- **Ativos**: as ações da B3 e os BDRs e ETFs listados aqui, com preços, balanços e fatos relevantes da CVM, do \
Banco Central e da B3.

Abaixo: o resultado de cada estratégia no passado, a conta simulada que segue as carteiras desde 03/11/2026, o \
painel do mercado, o parecer do assessor DeepSeek e a fronteira eficiente. Informação para decidir, não promessa: \
passado não garante futuro."""


@dataclass(frozen=True)
class Entradas:
    tickers: tuple[str, ...]     # the frontier's universe
    peso_maximo: float
    teto_setor: float


def ler_sidebar() -> Entradas:
    with st.sidebar:
        st.markdown("## Aleph Quant")
        st.markdown("### Fronteira eficiente")
        # a popover opens and closes on its own button; the multiselect's list did not close on a second click
        # (the user, 2026-11-04). No sector picked: all the stocks.
        with st.popover("Setores", icon=":material/category:", width="stretch"):
            setores = st.pills("Setores (nenhum = todas as ações)", SETORES, selection_mode="multi", key="setores")
        if setores:
            st.caption(f"{len(setores)} setor(es): {', '.join(setores)}")
        eua = st.checkbox(f"Incluir os EUA listados na B3 ({len(get_tickers_eua())}: 30 BDRs + IVVB11 + NASD11)",
                          value=True, key="eua", help=AVISO_EUA)
        peso_max = st.slider("Peso máximo por ativo (%)", 5, TETO_POR_ATIVO_DA_FRONTEIRA, 20,
                             key="peso_maximo") / 100
        teto_setor = st.slider("Teto por setor (%)", 20, TETO_POR_SETOR_DA_FRONTEIRA,
                               round(LIMITES.teto_setor * 100), 5, key="teto_setor",
                               help="Soma máxima dos pesos das ações de um mesmo setor na fronteira. As carteiras de "
                                    "verdade (Composite e Conservadora) seguem no máximo 4 ações por setor.") / 100
    acoes = get_tickers_by_setor(setores) if setores else get_all_tickers()
    if eua:
        acoes = [*acoes, *get_tickers_eua()]
    return Entradas(tuple(acoes), peso_max, teto_setor)


def _porcento(valor: float) -> str:
    return f"{valor:.1%}".replace(".", ",")


def _info_com_eua() -> dict:
    """Nome e setor das 181 e da perna dos EUA (``ativos_eua.py``, fora do caminho do paper)."""
    return {**get_ticker_info(), **get_ticker_info_eua()}


def main() -> None:
    # native theme in .streamlit/config.toml (the user's design request of 2026-11-04); the gold title is the theme's
    # the user's eagle above the name, both centered (the user, 2026-11-04)
    with st.container(horizontal_alignment="center", gap=None):
        if ICONE is not None:
            st.image(str(ICONE), width=120, alt="Logo do Aleph Quant: águia dourada e azul com seta de alta na asa")
        st.title(":yellow[Aleph Quant]", text_alignment="center")
        st.caption("Alocação quantitativa em ações, BDRs e ETFs da B3 · Gabriel Estrela Lopes", text_alignment="center")
    entradas = ler_sidebar()
    registrado = servicos.carregar_plano_registrado()
    st.markdown("## O que é o Aleph Quant")
    st.markdown(SOBRE)
    st.markdown("---")
    alepha.resultados_da_conservadora(servicos.carregar_curvas_conservadora(), servicos.carregar_juros())
    st.markdown("---")
    ui_simulador.secao(servicos.carregar_simulador(date.today(), precos_do_app.versao_dos_dados()))
    ui_simulador.parecer_do_deepseek(servicos.carregar_parecer_deepseek())
    st.markdown("---")
    if MOSTRAR_ACOES_EUA:
        ui_eua.secao(servicos.carregar_eua())
        st.markdown("---")
    secao_mercado()
    st.markdown("---")
    secao_assessor()
    st.markdown("---")
    secao_fronteira(entradas, registrado)


def secao_mercado() -> None:
    """The market panel (information only, DECISOES 124): news, the dollar and every stock, BDR and ETF of the
    app's daily prices (``ativos_tela``)."""
    hoje = date.today()
    diario, fatos, fundamentos = servicos.carregar_painel()
    curados = info_curada(get_ticker_info(), get_ticker_info_eua())
    info = {**curados, **info_da_tela(servicos.carregar_ativos_da_tela(), curados)}
    tickers = tuple(dict.fromkeys((*get_all_tickers(), *info, BENCHMARK_ACOES)))
    with st.spinner("Carregando o painel do mercado…"):
        precos = servicos.carregar_precos_da_tabela(tickers, hoje, precos_do_app.versao_dos_dados())
        desempenho = servicos.carregar_desempenho(tickers, hoje, precos_do_app.versao_dos_dados(), BENCHMARK_ACOES)
        painel.secao_mercado(hoje, diario, fatos, fundamentos, *servicos.carregar_manchetes(),
                             servicos.carregar_focus(hoje), servicos.carregar_indicadores_bcb(hoje),
                             precos, info, BENCHMARK_ACOES, desempenho, servicos.carregar_ata_copom(hoje))


def secao_assessor() -> None:
    """The DeepSeek adviser on the registered Aleph Quant plan (information only, DECISOES 136)."""
    registrado = servicos.carregar_plano_registrado()
    chave = servicos.chave_da_api("deepseek", "DEEPSEEK_API_KEY")
    indisponivel = ("sem a Composite registrada" if registrado is None else
                    "falta a chave do DeepSeek nos segredos do app ([deepseek] api_key)" if chave is None else None)
    universo = [] if registrado is None else list(registrado[1].index)
    ui_assessor.secao(indisponivel, universo, lambda profundidade: _parecer_do_plano(registrado, chave, profundidade))


def _parecer_do_plano(registrado: tuple[date, pd.Series], chave: str,
                      profundidade: assessor.Profundidade) -> assessor.Resultado:
    hoje = date.today()
    data, alvo = registrado
    pacote, fontes = parecer.pacote_e_fontes(alvo, data, chave, hoje)
    texto = assessor.mensagem_de_sistema(pacote, assessor.CARTEIRA_COMPOSTA)
    return servicos.gerar_parecer(texto, tuple(alvo.index), hoje, tuple(pacote), fontes, chave, profundidade.esforco,
                                  profundidade.partes)


def secao_fronteira(entradas: Entradas, registrado: tuple[date, pd.Series] | None) -> None:
    """The Composite's own estimated point next to the Markowitz frontier of the sidebar's universe (the user's request
    of 2026-11-03). The point is never moved onto the curve: the frontier treats estimated returns as certain, the
    equal-weight Composite uses none, so it sits below. Checked on 2026-11-04 with real data: 15 names, shrunk
    estimate 17.6% a year at 20.6% volatility, right of a capped frontier spanning 11.9-18.4% volatility; the caption
    reads it against the Selic and the equal-weight universe (the user asked for a plainer, less negative text)."""
    st.markdown("## Composite na fronteira eficiente")
    if registrado is None:
        st.caption("Sem a Composite registrada.")
        return
    _, alvo = registrado
    hoje = date.today()
    universo = tuple(dict.fromkeys((*entradas.tickers, *alvo.index)))
    with st.spinner("Calculando a fronteira eficiente…"):
        precos = servicos.carregar_precos(universo, ANOS_DA_FRONTEIRA, hoje, precos_do_app.versao_dos_dados()).precos
        rf = taxa_anual_para_aritmetica(servicos.carregar_selic(hoje).valor)
    colunas = [ticker for ticker in universo if ticker in precos.columns]
    nomes = [ticker for ticker in alvo.index if ticker in colunas]
    info = _info_com_eua()
    setores = {ticker: info[ticker]["setor"] for ticker in colunas if ticker in info}
    setores |= servicos.setores_do_lago(tuple(t for t in nomes if t not in setores), servicos.PASTA_MARCOS_V3)
    setores = {ticker: setores.get(ticker) or ticker.removesuffix(".SA") for ticker in colunas}
    momentos = estimar_momentos(retornos_simples(precos[colunas]), shrink_mu=True)
    mu, cov = momentos.mu[colunas], momentos.cov.loc[colunas, colunas]
    params = servicos.ParametrosOtimizacao(Objetivo.MIN_VOL.value, entradas.peso_maximo, len(colunas), None, 2.0,
                                           entradas.teto_setor, tuple(sorted(setores.items())))
    try:
        fronteira = servicos.calcular_fronteira(mu, cov, params, rf)
    except (ErroOtimizacao, ValueError) as erro:
        st.error(f"Fronteira inviável: {erro} Inclua mais setores ou ativos: o teto por setor é de no máximo "
                 f"{LIMITES.teto_setor:.0%}.")
        return
    pesos = pd.Series(1.0 / len(nomes), index=nomes).reindex(mu.index, fill_value=0.0)
    retorno, vol, _ = metricas_carteira(pesos.to_numpy(), mu.to_numpy(), preparar_covariancia(cov), rf)
    st.plotly_chart(grafico_fronteira_eficiente(fronteira, {"retorno": retorno, "volatilidade": vol},
                                                "Composite (pesos iguais)", rf), width="stretch")
    todas = pd.Series(1.0 / len(colunas), index=colunas)
    media, _, _ = metricas_carteira(todas.to_numpy(), mu.to_numpy(), preparar_covariancia(cov), rf)
    st.caption(
        f"Como ler: cada ponto da curva é a melhor carteira possível olhando para trás, escolhida já sabendo o retorno "
        f"de cada uma das {len(colunas)} ações nos últimos {ANOS_DA_FRONTEIRA} anos (até {precos.index[-1]:%d/%m/%Y}), "
        f"com no máximo {entradas.peso_maximo:.0%} por ação e {entradas.teto_setor:.0%} por setor. Nenhuma carteira "
        "real fica em cima dela, porque na hora da compra ninguém conhece o retorno que vem. "
        f"A Composite (pesos iguais nas suas {len(nomes)} ações) tem retorno estimado de {_porcento(retorno)} ao ano "
        f"com risco de {_porcento(vol)}: acima da Selic ({_porcento(rf)}) e da média das ações da fronteira "
        f"({_porcento(media)}). Ela fica mais à direita porque concentra em {len(nomes)} ações escolhidas pelos "
        "sinais em vez de espalhar por todas; se essa escolha funciona fora da amostra é o que o backtest sem viés "
        "acima mede.")
    por_setor = pd.Series(1.0 / len(nomes), index=nomes).groupby(lambda ticker: setores[ticker]).sum()
    st.bar_chart(por_setor.sort_values().rename("Peso na Composite"), horizontal=True, x_label="Peso na Composite")


if __name__ == "__main__":
    main()
