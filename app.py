"""
app.py - Dashboard Streamlit (camada fina sobre o pacote ``core``).
Otimizador de carteiras de ações da B3 para o investidor arrojado.
Autor: Gabriel Estrela Lopes

Fluxo: sidebar -> preços das ações -> momentos (Ledoit-Wolf +
Jorion) -> [growth: filtro + tilt, só na carteira de hoje] -> otimização ->
walk-forward de todas as estratégias no universo pré-filtro.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Otimizador Arrojado B3", page_icon="🚀", layout="wide")

from assets import (BENCHMARK_ACOES, SETORES, get_all_tickers, get_ticker_info,  # noqa: E402
                    get_tickers_by_setor, get_tickers_liquidos)
from core.backtest import ResultadoBacktest, estrategia_equiponderada, walk_forward  # noqa: E402
from core.config import DIAS_UTEIS_ANO, ConfigBacktest, Objetivo, taxa_anual_para_aritmetica  # noqa: E402
from core.estimation import aplicar_tilt_growth, estimar_momentos, retornos_simples  # noqa: E402
from core.growth import filtrar_por_percentil  # noqa: E402
from core.market_data import serie_rf_diaria  # noqa: E402
from core.metrics import contribuicao_risco, drawdown, metricas_rolantes  # noqa: E402
from core.optimization import (ErroInviavel, ErroOtimizacao, ResultadoOtimizacao,  # noqa: E402
                               fronteira_eficiente, otimizar)
from core.ordens import RegrasCompra, acoes_que_cabem, limites_do_plano, plano_de_compra  # noqa: E402
from ui import secoes, servicos  # noqa: E402
from visualizations import (grafico_barras_alocacao, grafico_composicao_pizza,  # noqa: E402
                            grafico_contribuicao_risco, grafico_curvas_capital, grafico_drawdown,
                            grafico_fronteira_eficiente, grafico_matriz_correlacao,
                            grafico_metricas_rolling, grafico_pesos_historicos)

ROTULOS_OBJETIVO = {
    Objetivo.KELLY.value: "Crescimento máximo (Kelly) — recomendado p/ arrojado",
    Objetivo.MAX_SHARPE.value: "Máximo Sharpe (exato)",
    Objetivo.MIN_VOL.value: "Mínima variância",
    Objetivo.HRP.value: "HRP (hierárquico, ignora retorno esperado)",
    Objetivo.MAX_RETORNO.value: "Máximo retorno c/ teto de vol (legado, concentra no erro)",
}

# Desligada: na avaliação pré-registrada de 2026-10-13 (scripts/avaliar_reamostragem.py),
# o Sharpe ficou abaixo do Kelly simples em 5 e 10 anos, e o drawdown não melhorou.
REAMOSTRAGEM_PADRAO = False
AJUDA_REAMOSTRAGEM = (
    "Média de 100 carteiras Kelly calculadas em reamostras do histórico (bootstrap em blocos de 21 "
    "pregões), projetada nos limites. Espalha o peso entre mais ativos. Na avaliação pré-registrada "
    "(walk-forward de 5 e 10 anos), o Sharpe ficou ABAIXO do Kelly simples e o drawdown não melhorou: "
    "opção de estudo, não padrão. Lenta: ~1 min a mais no walk-forward de 5 anos com a lista líquida, "
    "~15 min com todos os ativos."
)
AJUDA_TETO_SETOR = (
    "Soma máxima dos pesos das ações de um mesmo setor (100% = sem limite). Vale para a carteira de hoje, "
    "o plano de compra e o walk-forward; o HRP ignora."
)
# Plano de compra: cada ação comprada tem pelo menos 2 unidades (escolha do usuário, mercado fracionário).
QUANTIDADE_MINIMA_POR_ACAO = 2
PREGOES_PARA_FECHAMENTO = 10      # janela de onde sai o último fechamento quando o Yahoo não tem o negócio do dia


@dataclass(frozen=True)
class Entradas:
    params: servicos.ParametrosOtimizacao
    tickers_acoes: tuple[str, ...]
    anos: int                  # janela de estimação: carteira de hoje e treino rolante do walk-forward
    anos_teste: int
    orcamento: float
    usar_growth: bool
    percentil_growth: int
    peso_tilt: float
    usar_cvm: bool
    shrink_mu: bool
    custo_bps: float
    reamostrar: bool


@dataclass(frozen=True)
class CarteiraCalculada:
    """Pesos e os momentos que os geraram: fronteira, ponto 1/N e contribuição de risco usam os mesmos."""

    resultado: ResultadoOtimizacao
    mu: pd.Series
    cov: pd.DataFrame
    intensidade_shrinkage: float
    frequencia: pd.Series | None = None       # só na reamostrada


def ler_sidebar() -> Entradas:
    with st.sidebar:
        st.markdown("## Parâmetros")
        objetivo = st.selectbox("Objetivo", list(ROTULOS_OBJETIVO), format_func=ROTULOS_OBJETIVO.get)
        aversao = st.slider("Aversão a risco λ (1 = Kelly pleno)", 0.5, 5.0, 2.0, 0.5,
                            disabled=objetivo != Objetivo.KELLY.value,
                            help="λ=1 aproxima a carteira de maior crescimento geométrico. Como μ tem "
                                 "erro de estimação, λ≈2 é mais robusto. Sem caixa no conjunto "
                                 "viável, λ=2 NÃO equivale a 'meio Kelly'.")
        vol_max = st.slider("Teto de volatilidade (% a.a.)", 15, 80, 40, 5,
                            disabled=objetivo in (Objetivo.MIN_VOL.value, Objetivo.HRP.value)) / 100
        orcamento = st.number_input("Valor a investir (R$)", 100.0, 10_000_000.0, 10_000.0, 100.0,
                                    key="orcamento")

        st.markdown("### Universo")
        setores = st.multiselect("Setores (vazio = lista líquida)", SETORES, key="setores")
        todos = st.checkbox(f"Todos os ativos ({len(get_all_tickers())})", value=False)

        st.markdown("### Growth Score (só carteira de hoje)")
        usar_growth = st.checkbox("Filtro + tilt growth", value=True)
        percentil = st.slider("Percentil mínimo", 0, 80, 40, 5, disabled=not usar_growth)
        tilt = st.slider("Tilt (× vol)", 0.0, 1.0, 0.30, 0.05, disabled=not usar_growth)
        usar_cvm = st.checkbox("Receita 10 anos via CVM (lento)", value=False, disabled=not usar_growth)

        with st.expander("Avançado"):
            n_max = st.slider("Máximo de ativos", 3, 30, 10)
            peso_max = st.slider("Peso máximo por ativo (%)", 5, 50, 20) / 100
            teto_setor = st.slider("Teto por setor (%)", 20, 100, 30, 5, key="teto_setor",
                                   help=AJUDA_TETO_SETOR) / 100
            anos = st.slider("Janela de estimação (anos)", 3, 10, 5, key="janela",
                             help="Histórico que estima μ e Σ: na carteira de hoje e em cada rebalanceamento "
                                  "do walk-forward (treino rolante).")
            anos_teste = st.slider("Teste fora da amostra (anos)", 1, 10, 3, key="anos_teste",
                                   help="O walk-forward baixa janela + teste anos de preços: ações com "
                                        "histórico mais curto ficam fora dele (não da carteira de hoje).")
            shrink = st.checkbox("Shrinkage de μ (Jorion)", value=True)
            reamostrar = st.checkbox("Reamostragem de μ (bagging, B=100)", value=REAMOSTRAGEM_PADRAO,
                                     key="reamostrar", disabled=objetivo != Objetivo.KELLY.value,
                                     help=AJUDA_REAMOSTRAGEM)
            custo = st.slider("Custo de transação (bps por lado)", 0, 50, 10)

    if setores:
        acoes = get_tickers_by_setor(setores)
    elif todos:
        acoes = get_all_tickers()
    else:
        acoes = get_tickers_liquidos()
    info = get_ticker_info()
    params = servicos.ParametrosOtimizacao(
        objetivo=objetivo, peso_maximo=peso_max, n_ativos_max=n_max,
        vol_maxima=None if objetivo in (Objetivo.MIN_VOL.value, Objetivo.HRP.value) else vol_max,
        aversao_risco=aversao, teto_setor=teto_setor,
        setores=tuple(sorted((t, info[t]["setor"]) for t in acoes)),
    )
    # O checkbox desabilitado guarda o último valor: a reamostragem só vale para o Kelly.
    return Entradas(params, tuple(acoes), anos, anos_teste, orcamento, usar_growth, percentil, tilt,
                    usar_cvm, shrink, float(custo), reamostrar and objetivo == Objetivo.KELLY.value)


def validar(entradas: Entradas) -> str | None:
    p = entradas.params
    if p.n_ativos_max * p.peso_maximo < 1.0:
        return (f"Inviável: {p.n_ativos_max} ativos × {p.peso_maximo:.0%} = "
                f"{p.n_ativos_max * p.peso_maximo:.0%} < 100%. Aumente um dos dois.")
    if len(entradas.tickers_acoes) < 3:
        return "Selecione setores com pelo menos 3 ativos."
    return None


def main() -> None:
    st.markdown(secoes.CSS, unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Otimizador Arrojado B3</h1>'
                '<p class="subtitle">Kelly/Markowitz robusto · Ledoit-Wolf + Jorion · '
                'Walk-forward com custos · Gabriel Estrela Lopes</p>', unsafe_allow_html=True)
    entradas = ler_sidebar()
    erro = validar(entradas)
    if erro:
        st.error(erro)
        return

    hoje = date.today()
    universo = entradas.tickers_acoes
    anos_walk_forward = entradas.anos + entradas.anos_teste
    with st.spinner("Baixando preços, CDI e Selic..."):
        dados = servicos.carregar_precos(universo + (BENCHMARK_ACOES,), entradas.anos, hoje)
        precos_walk_forward = servicos.carregar_precos(universo + (BENCHMARK_ACOES,), anos_walk_forward,
                                                       hoje).precos
        cdi = servicos.carregar_cdi(anos_walk_forward, hoje)
        selic = servicos.carregar_selic(hoje)
    descartados = dados.descartados_sem_dados + dados.descartados_cobertura
    if descartados:
        st.warning(f"Descartados por falta de dados/cobertura: {', '.join(descartados)}")
    if cdi.empty:
        st.warning("CDI histórico indisponível: Sharpe/Sortino usam a Selic atual constante.")

    precos = dados.precos
    colunas = [t for t in universo if t in precos.columns]
    retornos = retornos_simples(precos[colunas])
    rf_aritmetica = taxa_anual_para_aritmetica(selic.valor)

    c1, c2, c3 = st.columns(3)
    c1.metric("Amostra comum", f"{retornos.index[0]:%m/%Y} a {retornos.index[-1]:%m/%Y}",
              f"{len(retornos)} pregões")
    c2.metric("Ativos válidos", len(colunas))
    c3.metric("Selic meta", f"{selic.valor:.2%}", selic.fonte, delta_color="off")

    carteira_hoje(entradas, precos, retornos, rf_aritmetica)
    secao_backtest(entradas, precos_walk_forward, colunas, cdi, selic.valor)
    st.markdown("---")
    st.markdown(secoes.LIMITACOES)


def carteira_hoje(entradas: Entradas, precos: pd.DataFrame, retornos: pd.DataFrame,
                  rf_aritmetica: float) -> None:
    st.markdown("## Carteira de hoje")
    acoes = list(retornos.columns)
    compraveis = acoes_compraveis(entradas, precos[acoes])
    momentos = estimar_momentos(retornos, shrink_mu=entradas.shrink_mu)
    mu, cov, growth = momentos.mu[compraveis], momentos.cov.loc[compraveis, compraveis], None
    info = get_ticker_info()

    if entradas.usar_growth:
        # O score sai do universo inteiro (a chave do cache não muda com o valor); o corte, só entre as compráveis.
        setores = tuple((t, info.get(t, {}).get("setor", "?")) for t in acoes)
        with st.spinner("Growth Score (fundamentos Yahoo/CVM)..."):
            growth = servicos.carregar_growth(tuple(acoes), precos[acoes], setores,
                                              entradas.usar_cvm, date.today())
        aprovados = filtrar_por_percentil(growth["score"].reindex(compraveis), entradas.percentil_growth,
                                          minimo_ativos=entradas.params.n_ativos_max)
        mu, cov = mu[aprovados], cov.loc[aprovados, aprovados]
        mu = aplicar_tilt_growth(mu, cov, growth["score"], entradas.peso_tilt)
        st.info(f"Filtro growth: {len(acoes)} → {len(aprovados)} ações. Tilt {entradas.peso_tilt:.2f}σ.")

    cfg = entradas.params.para_config(rf_aritmetica)
    try:
        carteira = (_carteira_reamostrada(entradas, retornos[list(mu.index)], rf_aritmetica, growth)
                    if entradas.reamostrar
                    else CarteiraCalculada(otimizar(mu, cov, cfg), mu, cov, momentos.intensidade_shrinkage))
        fronteira = fronteira_eficiente(carteira.mu, carteira.cov, cfg)
    except ErroInviavel as erro:
        st.error(f"Carteira inviável: {erro} Aumente o peso máximo ou o teto por setor, ou inclua mais setores.")
        st.stop()
    except ErroOtimizacao as erro:
        st.error(f"Otimização falhou: {erro}")
        st.stop()
    resultado, mu, cov = carteira.resultado, carteira.mu, carteira.cov
    if resultado.tem_aviso:
        st.warning(resultado.mensagem)
    else:
        st.caption(resultado.mensagem)

    secoes.cards_otimizacao(resultado, entradas.orcamento, carteira.intensidade_shrinkage)
    pesos = resultado.pesos[resultado.pesos > 0]
    n = len(mu)
    ponto_1n = {"retorno": float(mu.mean()), "volatilidade": float((cov.values.sum() / n**2) ** 0.5)}
    col1, col2 = st.columns([1.2, 1])
    col1.plotly_chart(grafico_fronteira_eficiente(
        fronteira, {"retorno": resultado.retorno_esperado, "volatilidade": resultado.volatilidade},
        resultado.objetivo.value, rf_aritmetica, benchmark_1n=ponto_1n), width="stretch")
    if carteira.frequencia is not None:
        col1.caption("A carteira reamostrada fica dentro da fronteira estimada, por construção.")
    col2.plotly_chart(grafico_composicao_pizza(pesos.to_dict(), entradas.orcamento), width="stretch")
    st.plotly_chart(grafico_barras_alocacao(pesos.to_dict(), entradas.orcamento, info), width="stretch")
    secao_plano_de_compra(entradas, pesos, precos, info, growth)

    col_rc, col_corr = st.columns(2)
    col_rc.plotly_chart(grafico_contribuicao_risco(contribuicao_risco(pesos, cov), pesos.to_dict()),
                        width="stretch")
    if len(pesos) > 1:
        col_corr.plotly_chart(grafico_matriz_correlacao(cov.loc[pesos.index, pesos.index]), width="stretch")
    if carteira.frequencia is not None:
        with st.expander("Estabilidade nas reamostras"):
            secoes.tabela_estabilidade(resultado.pesos, carteira.frequencia)
    if growth is not None:
        with st.expander("Componentes do Growth Score"):
            st.dataframe(growth, width="stretch")


def acoes_compraveis(entradas: Entradas, precos: pd.DataFrame) -> list[str]:
    """
    Tira do universo, antes de otimizar, as ações cuja quantidade mínima passa do teto por ação
    (último fechamento). Sem isso, o otimizador daria peso a ações que o plano de compra não
    consegue comprar, e esse peso viraria caixa.
    """
    peso_maximo = entradas.params.peso_maximo
    regras = RegrasCompra(entradas.orcamento, entradas.custo_bps, peso_maximo, QUANTIDADE_MINIMA_POR_ACAO)
    compraveis = acoes_que_cabem(precos.ffill().iloc[-1], regras)
    teto = peso_maximo * entradas.orcamento
    necessarias = math.ceil(1.0 / peso_maximo - 1e-9)        # menos que isso não soma 100% com o teto
    if len(compraveis) < necessarias:
        st.error(f"Com R$ {entradas.orcamento:,.2f} e peso máximo de {peso_maximo:.0%}, cada ação vai até "
                 f"R$ {teto:,.2f}: só {len(compraveis)} cabem com {QUANTIDADE_MINIMA_POR_ACAO} unidades, e a "
                 f"carteira precisa de {necessarias}. Aumente o valor a investir ou o peso máximo.")
        st.stop()
    caras = [t.replace(".SA", "") for t in precos.columns if t not in compraveis]
    if caras:
        st.info(f"Fora da otimização ({len(caras)}): {QUANTIDADE_MINIMA_POR_ACAO} ações passam do teto de "
                f"R$ {teto:,.2f} por ativo ({peso_maximo:.0%} do valor a investir): {', '.join(caras)}.")
    return compraveis


def secao_plano_de_compra(entradas: Entradas, pesos: pd.Series, precos: pd.DataFrame, info: dict,
                          growth: pd.DataFrame | None) -> None:
    st.markdown("### Plano de compra")
    tickers = tuple(pesos.index)
    janela = precos[list(tickers)].tail(PREGOES_PARA_FECHAMENTO)    # mesmos argumentos ao limpar e ao buscar
    if st.button("Atualizar cotações", icon=":material/refresh:", key="atualizar_cotacoes"):
        servicos.descartar_cotacoes(tickers, janela)
    with st.spinner("Buscando as cotações do dia…"):
        cotacoes = servicos.carregar_cotacoes(tickers, janela)
    com_cotacao = cotacoes["preco"].reindex(pesos.index) > 0          # NaN e ticker ausente: False
    if not com_cotacao.all():
        nomes = ", ".join(t.replace(".SA", "") for t in pesos.index[~com_cotacao])
        st.warning(f"Sem cotação: {nomes}. Fora do plano de compra; o valor dela vai para as outras ações.")
    # Reescala para 100%: o valor das ações sem cotação vai para as outras, na proporção do otimizador.
    alvo = pesos[com_cotacao] / pesos[com_cotacao].sum()
    # O HRP ignora os tetos: o plano nunca corta um alvo abaixo do próprio peso (por ação ou por setor).
    teto = min(1.0, max(entradas.params.peso_maximo, float(pesos.max())))
    regras = RegrasCompra(entradas.orcamento, entradas.custo_bps, teto, QUANTIDADE_MINIMA_POR_ACAO,
                          limites_do_plano(entradas.params.grupos, alvo))
    try:
        plano = plano_de_compra(alvo, cotacoes["preco"], regras)
    except ValueError as erro:
        st.error(f"Plano de compra indisponível: {erro}")
        return
    secoes.tabela_carteira(plano, cotacoes, info, growth, date.today())
    secoes.legenda_por_setor(plano, info)
    secoes.resumo_plano(plano, regras)


def _carteira_reamostrada(entradas: Entradas, retornos: pd.DataFrame, rf_aritmetica: float,
                          growth: pd.DataFrame | None) -> CarteiraCalculada:
    """Reestima μ e Σ no universo JÁ filtrado pelo growth: são esses momentos que a tela mostra."""
    with st.spinner("Reamostrando: 100 carteiras Kelly..."):
        reamostrada = servicos.otimizar_reamostrado(
            retornos, entradas.params, rf_aritmetica, entradas.shrink_mu,
            None if growth is None else growth["score"], entradas.peso_tilt)
    return CarteiraCalculada(reamostrada.resultado, reamostrada.mu, reamostrada.cov,
                             reamostrada.intensidade_shrinkage, reamostrada.frequencia)


def secao_backtest(entradas: Entradas, precos: pd.DataFrame, acoes_hoje: list[str], cdi: pd.Series,
                   selic: float) -> None:
    """``precos``: download de janela + teste anos, só do walk-forward; ``acoes_hoje``: universo da carteira de hoje."""
    st.markdown("---")
    st.markdown("## Walk-forward fora da amostra (universo pré-filtro, líquido de custos)")
    colunas = [t for t in entradas.tickers_acoes if t in precos.columns]
    anos_historico = entradas.anos + entradas.anos_teste
    if len(colunas) < 2:          # sem ações, o 1/N divide por zero; com uma só, não há carteira a comparar
        st.info(f"Walk-forward indisponível: só {len(colunas)} ação(ões) com {anos_historico} anos de histórico. "
                "Diminua a janela de estimação ou o teste fora da amostra.")
        return
    retornos = retornos_simples(precos[colunas])
    rf_diaria = serie_rf_diaria(retornos.index, cdi, selic)
    # Um cfg só, para as estratégias e o BOVA11: mesma janela de treino, mesmas datas, mesmos custos.
    cfg_bt = ConfigBacktest(janela_treino=entradas.anos * DIAS_UTEIS_ANO, capital_inicial=entradas.orcamento,
                            custo_bps=entradas.custo_bps, shrink_mu=entradas.shrink_mu)
    espera = ("de 5 estratégias (a reamostrada leva minutos)" if entradas.reamostrar
              else "de 4 estratégias")
    with st.spinner(f"Rodando walk-forward {espera}..."):
        try:
            resultados = servicos.rodar_walk_forward(retornos, entradas.params, cfg_bt, cdi,
                                                     taxa_anual_para_aritmetica(selic), entradas.reamostrar)
        except ValueError as erro:
            st.info(f"Walk-forward indisponível: {erro}")
            return

    # A ordem do dicionário define a principal: a reamostrada vem primeiro quando ligada.
    nome_otimizada = next(iter(resultados))
    otimizada = resultados[nome_otimizada]
    st.caption(_legenda_walk_forward(entradas, otimizada, colunas, acoes_hoje))
    benchmark = None
    if BENCHMARK_ACOES in precos.columns:
        ret_bova = retornos_simples(precos[[BENCHMARK_ACOES]]).reindex(retornos.index).dropna()
        if len(ret_bova) > cfg_bt.janela_treino + 1:
            benchmark = walk_forward(ret_bova, estrategia_equiponderada(), cfg_bt, "BOVA11")

    if otimizada.falhas:
        st.warning(f"{len(otimizada.falhas)} rebalanceamento(s) mantiveram a carteira por falha: "
                   + "; ".join(otimizada.falhas[:3]))
    secoes.tabela_comparativa(resultados, rf_diaria, benchmark)
    if benchmark is not None:
        secoes.bloco_alfa(otimizada, benchmark, rf_diaria)

    curvas = {nome: bt.valores for nome, bt in resultados.items()}
    if benchmark is not None:
        curvas["BOVA11"] = benchmark.valores
    datas = otimizada.valores.index
    curvas["CDI"] = entradas.orcamento * (1 + rf_diaria.reindex(datas).fillna(0.0).iloc[1:]).cumprod()
    curvas["CDI"] = pd.concat([pd.Series([entradas.orcamento], index=datas[:1]), curvas["CDI"]])
    st.plotly_chart(grafico_curvas_capital(curvas, destaque=nome_otimizada), width="stretch")

    serie_dd, _, _ = drawdown(otimizada.valores)
    st.plotly_chart(grafico_drawdown(serie_dd), width="stretch")
    col1, col2 = st.columns(2)
    pesos_hist = otimizada.pesos_alvo.loc[:, otimizada.pesos_alvo.max() > 0.01]
    col1.plotly_chart(grafico_pesos_historicos(pesos_hist), width="stretch")
    rolantes = metricas_rolantes(otimizada.valores.pct_change().dropna(), rf_diaria)
    if not rolantes.empty:
        col2.plotly_chart(grafico_metricas_rolling(rolantes), width="stretch")


def _legenda_walk_forward(entradas: Entradas, otimizada: ResultadoBacktest, colunas_walk_forward: list[str],
                          acoes_hoje: list[str]) -> str:
    fora = [t.replace(".SA", "") for t in acoes_hoje if t not in colunas_walk_forward]
    texto = (f"Treino: {entradas.anos} anos rolando, a mesma janela da carteira de hoje · teste: "
             f"{entradas.anos_teste} ano(s), fora da amostra desde {otimizada.pesos_alvo.index[0]:%d/%m/%Y} · "
             f"{len(colunas_walk_forward)} ações com {entradas.anos + entradas.anos_teste} anos de histórico "
             f"(a carteira de hoje usa {len(acoes_hoje)})")
    return texto + (f". Fora do walk-forward por histórico curto: {', '.join(fora)}." if fora else ".")


if __name__ == "__main__":
    main()
