"""
app.py - Dashboard Streamlit (camada fina sobre o pacote ``core``): o Aleph Quant.
Otimizador de carteiras de ações da B3 para o investidor arrojado.
Autor: Gabriel Estrela Lopes

The page (the user's UI lists of 2026-11-03, 2026-11-04 and 2026-11-06): what Aleph Quant is; the Composite's quotes
and its weight by sector; the survivorship-free backtest of the Composite and the Conservadora; the simulator and the
DeepSeek portfolio; the market panel; and the DeepSeek adviser. The buy plan from the held positions left the page on
2026-11-04 (the positions and the month's tax typed in were gone on the next visit); its engine stays in ``core``. The
Composite next to the Markowitz efficient frontier left on 2026-11-06 with its sidebar block; ``core.optimization``
keeps the frontier.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

from ui.marca import ICONE_PADRAO, arquivo_da_logo, arquivo_do_icone  # noqa: E402  (stdlib only: safe before set_page_config)

LOGO, ICONE = arquivo_da_logo(), arquivo_do_icone()
st.set_page_config(page_title="Aleph Quant · Alocação quantitativa B3", page_icon=str(ICONE or ICONE_PADRAO),
                   layout="wide")
if LOGO is not None:     # the eagle and the gold name, top-left: the page has no sidebar (the user, 2026-11-06)
    st.logo(str(LOGO), icon_image=str(ICONE), size="large")

from assets import BENCHMARK_ACOES, get_all_tickers, get_ticker_info  # noqa: E402
from ativos_eua import get_ticker_info_eua  # noqa: E402
from ativos_tela import info_curada, info_da_tela  # noqa: E402
from core import assessor, precos_do_app  # noqa: E402
from ui import alepha, comparacao, cotacoes, painel, parecer, servicos  # noqa: E402
from ui import assessor as ui_assessor  # noqa: E402
from ui import eua as ui_eua  # noqa: E402
from ui import simulador as ui_simulador  # noqa: E402
from ui import formulas as ui_formulas  # noqa: E402
from ui import sentinela as ui_sentinela  # noqa: E402
from ui import simulador_opcoes as ui_opcoes  # noqa: E402

# The US stocks screen (ui/eua.py, scraped US data) is on standby since the user's decision of 2026-11-03
# (DECISOES 240): B3 only. Code and data stay.
MOSTRAR_ACOES_EUA = False

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

Abaixo: as ações da Composite e seus pesos por setor, o resultado de cada estratégia no passado, a conta simulada que \
segue as carteiras desde 03/11/2026, o painel do mercado e o parecer do assessor DeepSeek. Informação para decidir, não \
promessa: passado não garante futuro."""


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
    registrado = servicos.carregar_plano_registrado()
    st.markdown("## O que é o Aleph Quant")
    st.markdown(SOBRE)
    st.markdown("---")
    secao_cotacoes(registrado)
    st.markdown("---")
    alepha.resultados_da_conservadora(servicos.carregar_curvas_conservadora(), servicos.carregar_juros())
    st.markdown("---")
    comparacao.secao(servicos.carregar_comparacao(), servicos.carregar_juros())
    st.markdown("---")
    ui_formulas.secao(servicos.carregar_formulas())
    st.markdown("---")
    ui_simulador.secao(servicos.carregar_simulador(date.today(), precos_do_app.versao_dos_dados()))
    ui_simulador.parecer_do_deepseek(servicos.carregar_parecer_deepseek())
    st.markdown("---")
    ui_opcoes.secao(servicos.carregar_opcoes(date.today(), precos_do_app.versao_dos_dados()))
    st.markdown("---")
    ui_sentinela.secao(servicos.carregar_sentinela(date.today(), precos_do_app.versao_dos_dados()))
    st.markdown("---")
    if MOSTRAR_ACOES_EUA:
        ui_eua.secao(servicos.carregar_eua())
        st.markdown("---")
    secao_mercado()
    st.markdown("---")
    secao_assessor()


def secao_cotacoes(registrado: tuple[date, pd.Series] | None) -> None:
    """The Composite's quotes now (the user, 2026-11-05), from a year of daily closes for the previous close."""
    tickers = () if registrado is None else tuple(registrado[1].index)
    fechamentos = (servicos.carregar_precos(tickers, 1, date.today(), precos_do_app.versao_dos_dados()).precos
                   if tickers else pd.DataFrame())
    cotacoes.secao(tickers, fechamentos, servicos.carregar_cotacoes, servicos.descartar_cotacoes, get_ticker_info())
    if registrado is not None:
        _peso_por_setor(registrado[1])


def _peso_por_setor(alvo: pd.Series) -> None:
    """The Composite's equal weights summed by sector, right under its quotes table (the user, 2026-11-06: the one part
    of the efficient frontier section that stays). The sector comes from the registry, then the lake, then the
    ticker."""
    nomes = list(alvo.index)
    info = _info_com_eua()
    setores = {ticker: info[ticker]["setor"] for ticker in nomes if info.get(ticker, {}).get("setor")}
    setores |= servicos.setores_do_lago(tuple(t for t in nomes if t not in setores), servicos.PASTA_MARCOS_V3)
    setores = {ticker: setores.get(ticker) or ticker.removesuffix(".SA") for ticker in nomes}
    por_setor = pd.Series(1.0 / len(nomes), index=nomes).groupby(lambda ticker: setores[ticker]).sum()
    st.bar_chart(por_setor.sort_values().rename("Peso na Composite"), horizontal=True, x_label="Peso na Composite")


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
                             precos, info, BENCHMARK_ACOES, desempenho, servicos.carregar_ata_copom(hoje),
                             selic=_selic_de_hoje(hoje), dolar=servicos.carregar_dolar(hoje, _quarto_de_hora()))


def _quarto_de_hora() -> int:
    agora = datetime.now()
    return agora.hour * 4 + agora.minute // 15


def _selic_de_hoje(hoje: date) -> float | None:
    """The Selic target in force (BCB SGS 432, ``market_data.obter_selic_meta``); None on its fixed fallback, which
    is not today's rate."""
    selic = servicos.carregar_selic(hoje)
    return None if selic.fonte.startswith("fallback") else selic.valor


def secao_assessor() -> None:
    """The DeepSeek adviser on the registered Aleph Quant plan (information only, DECISOES 136)."""
    registrado = servicos.carregar_plano_registrado()
    chave = servicos.chave_da_api("deepseek", "DEEPSEEK_API_KEY")
    indisponivel = ("sem a Composite registrada" if registrado is None else
                    "falta a chave do DeepSeek nos segredos do app ([deepseek] api_key)" if chave is None else None)
    universo = [] if registrado is None else list(registrado[1].index)
    ui_assessor.secao(indisponivel, universo, lambda: _parecer_do_plano(registrado, chave),
                      codigo=servicos.codigo_do_parecer())


def _parecer_do_plano(registrado: tuple[date, pd.Series], chave: str) -> assessor.Resultado:
    """One opinion at the default depth (max effort, one part): the only one the screen asks since 2026-11-06."""
    profundidade = assessor.PROFUNDIDADES[assessor.PROFUNDIDADE_PADRAO]
    hoje = date.today()
    data, alvo = registrado
    pacote, fontes = parecer.pacote_e_fontes(alvo, data, chave, hoje)
    texto = assessor.mensagem_de_sistema(pacote, assessor.CARTEIRA_COMPOSTA)
    return servicos.gerar_parecer(texto, tuple(alvo.index), hoje, tuple(pacote), fontes, chave, profundidade.esforco,
                                  profundidade.partes)


if __name__ == "__main__":
    main()
