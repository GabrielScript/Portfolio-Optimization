"""
app.py - Dashboard Streamlit (camada fina sobre o pacote ``core``): o Aleph Neumann.
Otimizador de carteiras de ações da B3 para o investidor arrojado.
Autor: Gabriel Estrela Lopes

The page (the user's UI list of 2026-11-03): the real-money plan, the Composite registered in paper-v3 (DECISOES 242),
from the current positions; the survivorship-free backtest of the Composite and the Conservadora; the simulator and the
DeepSeek portfolio; the market panel; the DeepSeek adviser; and the Composite next to the Markowitz efficient frontier.
The study screens (the sidebar's optimized portfolio, the biased walk-forward, BOVA11 + satellite) left the page; their
code stays in ``core`` for the scripts and the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from ui.marca import ICONE_PADRAO, arquivo_da_logo, arquivo_do_icone  # noqa: E402  (stdlib only: safe before set_page_config)

LOGO, ICONE = arquivo_da_logo(), arquivo_do_icone()
st.set_page_config(page_title="Aleph Neumann · Alocação quantitativa B3", page_icon=str(ICONE or ICONE_PADRAO),
                   layout="wide")
if LOGO is not None:
    st.logo(str(LOGO), icon_image=str(ICONE), size="large")

from assets import (BENCHMARK_ACOES, BENCHMARKS, SETORES, get_all_tickers, get_ticker_info,  # noqa: E402
                    get_tickers_by_setor, get_tickers_liquidos)
from ativos_eua import AVISO_EUA, get_ticker_info_eua, get_tickers_eua  # noqa: E402
from ativos_tela import info_curada, info_da_tela  # noqa: E402
from core.config import Objetivo, limites_por_setor, taxa_anual_para_aritmetica  # noqa: E402
from core.estimation import estimar_momentos, retornos_simples  # noqa: E402
from core.execucao import Custos  # noqa: E402
from core import assessor, precos_do_app  # noqa: E402
from core.optimization import ErroOtimizacao, metricas_carteira, preparar_covariancia  # noqa: E402
from core.rebalanceio import (RegrasRebalanceio, ZonaSemNegociacao, capital_minimo, custo_com_liquidez,  # noqa: E402
                              pesos_finais, plano_com_nucleo, plano_de_rebalanceio, posicoes_da_tabela)
from core.risco import LIMITES, Violacao, checar_carteira  # noqa: E402
from ui import alepha, painel, parecer, secoes, servicos  # noqa: E402
from ui import assessor as ui_assessor  # noqa: E402
from ui import eua as ui_eua  # noqa: E402
from ui import simulador as ui_simulador  # noqa: E402
from visualizations import grafico_fronteira_eficiente  # noqa: E402

# The screen's ETFs (the references and IVVB11), which have no R$ 20 thousand exemption on sales.
ETFS_DA_TELA = frozenset((*BENCHMARKS, "IVVB11.SA"))
# The final proof's costs (scripts/avaliar_execucao.py); a name outside the lake's liquidity snapshot pays CUSTO_BPS.
CUSTOS_DA_B3 = Custos(emolumentos=0.0003, impacto=1.0)
CUSTO_BPS = 10.0                 # fixed since the user's UI list of 2026-11-03: the default, no longer a control
NUCLEO_DO_PEQUENO_CAPITAL = frozenset({BENCHMARK_ACOES})     # the ETF kept below the minimum capital (DECISOES 124)
QUANTIDADE_MINIMA_POR_ACAO = 2   # each stock bought has at least 2 shares (the user's choice, odd-lot market)
# The US stocks screen (ui/eua.py, scraped US data) is on standby since the user's decision of 2026-11-03
# (DECISOES 240): B3 only. Code and data stay; the B3-listed BDRs and ETFs of ativos_eua.py stay in the sidebar.
MOSTRAR_ACOES_EUA = False
PREGOES_PARA_FECHAMENTO = 10     # the window the last close comes from when the day has no trade yet
# The frontier (information only): 5 years of lake prices, Jorion-shrunk mu and Ledoit-Wolf covariance (the old
# study's defaults); its cap per asset goes up to 25% (user, 2026-11-04). The plan and the simulator keep LIMITES.
ANOS_DA_FRONTEIRA = 5
TETO_POR_ATIVO_DA_FRONTEIRA = 25


@dataclass(frozen=True)
class Entradas:
    orcamento: float
    tickers: tuple[str, ...]     # the frontier's universe
    peso_maximo: float
    teto_setor: float


def ler_sidebar() -> Entradas:
    with st.sidebar:
        st.markdown("## Aleph Neumann")
        orcamento = st.number_input("Valor a investir (R$)", 10.0, 10_000_000.0, 10_000.0, 10.0,
                                    key="orcamento")
        st.markdown("### Fronteira eficiente")
        setores = st.multiselect("Setores (vazio = todos os ativos ou a lista líquida)", SETORES, key="setores",
                                 placeholder="Escolha setores")
        todos = st.checkbox(f"Todos os ativos ({len(get_all_tickers())})", value=True, key="todos",
                            disabled=bool(setores), help="Desmarcado: só a lista líquida (mais rápido). Com setores "
                                                         "escolhidos, valem os setores.")
        eua = st.checkbox(f"Incluir os EUA listados na B3 ({len(get_tickers_eua())}: 30 BDRs + IVVB11 + NASD11)",
                          value=True, key="eua", help=AVISO_EUA)
        peso_max = st.slider("Peso máximo por ativo (%)", 5, TETO_POR_ATIVO_DA_FRONTEIRA, 20,
                             key="peso_maximo") / 100
        teto_setor = st.slider("Teto por setor (%)", 20, round(LIMITES.teto_setor * 100),
                               round(LIMITES.teto_setor * 100), 5, key="teto_setor",
                               help="Soma máxima dos pesos das ações de um mesmo setor na fronteira.") / 100
    if setores:
        acoes = get_tickers_by_setor(setores)
    elif todos:
        acoes = get_all_tickers()
    else:
        acoes = get_tickers_liquidos()
    if eua:
        acoes = [*acoes, *get_tickers_eua()]
    return Entradas(orcamento, tuple(acoes), peso_max, teto_setor)


def _info_com_eua() -> dict:
    """Nome e setor das 181 e da perna dos EUA (``ativos_eua.py``, fora do caminho do paper)."""
    return {**get_ticker_info(), **get_ticker_info_eua()}


def _descrever(violacoes: list[Violacao]) -> str:
    return "; ".join(f"{v.regra.value} em {v.alvo.replace('.SA', '')}: {v.valor:.2%} (limite {v.limite:.0%})"
                     for v in violacoes) + '.'


def main() -> None:
    # native theme in .streamlit/config.toml (the user's design request of 2026-11-04); the gold title is the theme's
    st.title(":yellow[Aleph Neumann]")
    st.caption("Alocação quantitativa em ações, BDRs e ETFs da B3 · Composite no lago sem viés de sobrevivência · "
               "Gabriel Estrela Lopes")
    entradas = ler_sidebar()
    registrado = servicos.carregar_plano_registrado()
    secao_conservadora(entradas, registrado)
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
        painel.secao_mercado(hoje, diario, fatos, fundamentos, *servicos.carregar_manchetes(),
                             servicos.carregar_focus(hoje), servicos.carregar_indicadores_bcb(hoje),
                             precos, info, BENCHMARK_ACOES)


def secao_assessor() -> None:
    """The DeepSeek adviser on the registered Aleph Neumann plan (information only, DECISOES 136)."""
    registrado = servicos.carregar_plano_registrado()
    chave = servicos.chave_da_api("deepseek", "DEEPSEEK_API_KEY")
    indisponivel = ("sem carteira registrada no paper-v3" if registrado is None else
                    "falta a chave do DeepSeek nos segredos do app ([deepseek] api_key)" if chave is None else None)
    universo = [] if registrado is None else list(registrado[1].index)
    ui_assessor.secao(indisponivel, universo, lambda: _parecer_do_plano(registrado, chave))


def _parecer_do_plano(registrado: tuple[date, pd.Series], chave: str) -> assessor.Resultado:
    hoje = date.today()
    data, alvo = registrado
    pacote, fontes = parecer.pacote_e_fontes(alvo, data, chave, hoje)
    texto = assessor.mensagem_de_sistema(pacote, assessor.CARTEIRA_COMPOSTA)
    return servicos.gerar_parecer(texto, tuple(alvo.index), hoje, tuple(pacote), fontes, chave)


def secao_conservadora(entradas: Entradas, registrado: tuple[date, pd.Series] | None) -> None:
    st.markdown("### Plano de compra: Aleph Neumann (Composite, dinheiro real)")
    if registrado is None:
        st.error("Sem carteira registrada no paper-v3 (paper-v3/composto.csv): o plano do Aleph Neumann usa a de lá.")
        return
    _, alvo = registrado
    posicoes = {ticker + ".SA": posicao for ticker, posicao in posicoes_da_tabela(alepha.editor_de_posicoes()).items()}
    mes = alepha.mes_fiscal()
    tickers = tuple(dict.fromkeys((*alvo.index, *posicoes)))
    historico = servicos.carregar_precos(tickers, 1, date.today(), precos_do_app.versao_dos_dados()).precos
    janela = historico.tail(PREGOES_PARA_FECHAMENTO)
    if st.button("Atualizar cotações", icon=":material/refresh:", key="atualizar_cotacoes_alepha"):
        servicos.descartar_cotacoes(tickers, janela)
    with st.spinner("Buscando as cotações do dia…"):
        cotacoes = servicos.carregar_cotacoes(tickers, janela)
    sem_cotacao = [ticker.removesuffix(".SA") for ticker in tickers if not cotacoes["preco"].get(ticker, math.nan) > 0]
    if sem_cotacao:
        st.error(f"Plano indisponível, sem cotação de {', '.join(sem_cotacao)}. Confira o ticker ou atualize as "
                 "cotações.")
        return
    setores = servicos.setores_do_lago(tuple(alvo.index), servicos.PASTA_MARCOS_V3)
    foto = servicos.carregar_liquidez()
    precos = cotacoes["preco"]
    patrimonio = sum(posicao.quantidade * float(precos[ticker]) for ticker, posicao in posicoes.items())
    patrimonio += entradas.orcamento
    minimo = capital_minimo(alvo, precos, QUANTIDADE_MINIMA_POR_ACAO, CUSTO_BPS)
    pequeno = patrimonio < minimo     # the user's decision of 2026-10-23 (DECISOES 124): start now, any amount
    quantidade_minima = 1 if pequeno else QUANTIDADE_MINIMA_POR_ACAO
    regras = RegrasRebalanceio(LIMITES.peso_maximo_acao, CUSTO_BPS, quantidade_minima,
                               limites_por_setor(setores, LIMITES.teto_setor))
    custo = custo_com_liquidez(foto, CUSTOS_DA_B3, CUSTO_BPS)
    argumentos = (alvo, posicoes, precos, entradas.orcamento, regras, custo)
    try:
        plano = (plano_com_nucleo(*argumentos, nucleo=NUCLEO_DO_PEQUENO_CAPITAL, mes=mes, etfs=ETFS_DA_TELA)
                 if pequeno else plano_de_rebalanceio(*argumentos, mes=mes, etfs=ETFS_DA_TELA))
    except ValueError as erro:
        st.error(f"Plano indisponível: {erro}")
        return
    try:  # information only: when the zone's redo fails, the plan shows without its columns
        zona = None if foto is None or pequeno else plano_de_rebalanceio(
            *argumentos, mes=mes, etfs=ETFS_DA_TELA,
            zona=ZonaSemNegociacao(foto.set_index("security")["vol_diaria"].rename(lambda papel: f"{papel}.SA")))
    except ValueError:
        zona = None
    violacoes = checar_carteira(pesos_finais(plano.ordens, patrimonio, NUCLEO_DO_PEQUENO_CAPITAL), setores)
    if violacoes:
        st.error("Plano recusado pelas regras de risco fixas: " + _descrever(violacoes))
        return
    if pequeno:
        st.info(alepha.aviso_capital_pequeno(
            patrimonio, minimo, float(precos.reindex(alvo.index).sum()), len(alvo),
            [ticker.removesuffix(".SA") for ticker in alvo.index if plano.ordens.at[ticker, "alvo"] == 0],
            QUANTIDADE_MINIMA_POR_ACAO, LIMITES.peso_maximo_acao, BENCHMARK_ACOES.removesuffix(".SA")))
    alepha.tabela_rebalanceio(plano, zona, patrimonio)
    if pequeno:
        st.warning(f"Com R\\$ {patrimonio:,.2f} a carteira fica longe da carteira testada ({len(alvo)} nomes com pesos "
                   f"iguais, R\\$ 10 mil na prova final): menos nomes e pesos desiguais, mais risco de cada empresa.")


def secao_fronteira(entradas: Entradas, registrado: tuple[date, pd.Series] | None) -> None:
    """The Composite's own estimated point next to the Markowitz frontier of the sidebar's universe (the user's request
    of 2026-11-03). The point is never moved onto the curve: the frontier treats estimated returns as certain, the
    equal-weight Composite uses none, so it sits below."""
    st.markdown("## Composite na fronteira eficiente")
    if registrado is None:
        st.caption("Sem a Composite registrada no paper-v3.")
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
    texto = (f"Fronteira com {len(colunas)} ativos e {ANOS_DA_FRONTEIRA} anos de preços do lago, até "
             f"{precos.index[-1]:%d/%m/%Y}. A Composite fica onde as estimativas a colocam: a fronteira trata o "
             "retorno do passado como certo e, na prática, rende menos do que promete.")
    curva = fronteira.sort_values("volatilidade")
    if not curva.empty and curva["volatilidade"].min() <= vol <= curva["volatilidade"].max():
        teto = float(np.interp(vol, curva["volatilidade"], curva["retorno"]))
        texto += (f" Com o mesmo risco ({vol:.1%} a.a.), a fronteira estima {teto:.1%} ao ano, contra {retorno:.1%} "
                  "da Composite.")
    st.caption(texto)
    por_setor = pd.Series(1.0 / len(nomes), index=nomes).groupby(lambda ticker: setores[ticker]).sum()
    st.bar_chart(por_setor.sort_values().rename("Peso na Composite"), horizontal=True, x_label="Peso na Composite")


if __name__ == "__main__":
    main()
