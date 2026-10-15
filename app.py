"""
app.py - Dashboard Streamlit (camada fina sobre o pacote ``core``).
Otimizador de carteiras de ações da B3 para o investidor arrojado.
Autor: Gabriel Estrela Lopes

Fluxo: sidebar -> preços das ações -> momentos (Ledoit-Wolf +
Jorion) -> [growth: filtro + tilt, só na carteira de hoje] -> otimização ->
walk-forward de todas as estratégias no universo pré-filtro.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Otimizador Arrojado B3", page_icon="🚀", layout="wide")

from assets import (BENCHMARK_ACOES, SETORES, get_all_tickers, get_ticker_info,  # noqa: E402
                    get_tickers_by_setor, get_tickers_liquidos)
from core.backtest import estrategia_equiponderada, walk_forward  # noqa: E402
from core.config import ConfigBacktest, Objetivo, taxa_anual_para_aritmetica  # noqa: E402
from core.estimation import aplicar_tilt_growth, estimar_momentos, retornos_simples  # noqa: E402
from core.growth import filtrar_por_percentil  # noqa: E402
from core.market_data import serie_rf_diaria  # noqa: E402
from core.metrics import contribuicao_risco, drawdown, metricas_rolantes  # noqa: E402
from core.optimization import (ErroOtimizacao, ResultadoOtimizacao, fronteira_eficiente,  # noqa: E402
                               otimizar)
from core.ordens import RegrasCompra, plano_de_compra  # noqa: E402
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
# Plano de compra: cada ação comprada tem pelo menos 2 unidades (escolha do usuário, mercado fracionário).
QUANTIDADE_MINIMA_POR_ACAO = 2
PREGOES_PARA_FECHAMENTO = 10      # janela de onde sai o último fechamento quando o Yahoo não tem o negócio do dia


@dataclass(frozen=True)
class Entradas:
    params: servicos.ParametrosOtimizacao
    tickers_acoes: tuple[str, ...]
    anos: int
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
        setores = st.multiselect("Setores (vazio = lista líquida)", SETORES)
        todos = st.checkbox(f"Todos os ativos ({len(get_all_tickers())})", value=False)

        st.markdown("### Growth Score (só carteira de hoje)")
        usar_growth = st.checkbox("Filtro + tilt growth", value=True)
        percentil = st.slider("Percentil mínimo", 0, 80, 40, 5, disabled=not usar_growth)
        tilt = st.slider("Tilt (× vol)", 0.0, 1.0, 0.30, 0.05, disabled=not usar_growth)
        usar_cvm = st.checkbox("Receita 10 anos via CVM (lento)", value=False, disabled=not usar_growth)

        with st.expander("Avançado"):
            n_max = st.slider("Máximo de ativos", 3, 30, 10)
            peso_max = st.slider("Peso máximo por ativo (%)", 5, 50, 20) / 100
            anos = st.slider("Histórico (anos)", 3, 10, 5,
                             help="Walk-forward precisa de ≥ 2 anos de treino + teste.")
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
    params = servicos.ParametrosOtimizacao(
        objetivo=objetivo, peso_maximo=peso_max, n_ativos_max=n_max,
        vol_maxima=None if objetivo in (Objetivo.MIN_VOL.value, Objetivo.HRP.value) else vol_max,
        aversao_risco=aversao,
    )
    # O checkbox desabilitado guarda o último valor: a reamostragem só vale para o Kelly.
    return Entradas(params, tuple(acoes), anos, orcamento, usar_growth, percentil, tilt,
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
    with st.spinner("Baixando preços, CDI e Selic..."):
        dados = servicos.carregar_precos(universo + (BENCHMARK_ACOES,), entradas.anos, hoje)
        cdi = servicos.carregar_cdi(entradas.anos, hoje)
        selic = servicos.carregar_selic(hoje)
    descartados = dados.descartados_sem_dados + dados.descartados_cobertura
    if descartados:
        st.warning(f"Descartados por falta de dados/cobertura: {', '.join(descartados)}")
    if cdi.empty:
        st.warning("CDI histórico indisponível: Sharpe/Sortino usam a Selic atual constante.")

    precos = dados.precos
    benchmark_disponivel = BENCHMARK_ACOES in precos.columns
    colunas = [t for t in universo if t in precos.columns]
    retornos = retornos_simples(precos[colunas])
    rf_aritmetica = taxa_anual_para_aritmetica(selic.valor)
    rf_diaria = serie_rf_diaria(retornos.index, cdi, selic.valor)

    c1, c2, c3 = st.columns(3)
    c1.metric("Amostra comum", f"{retornos.index[0]:%m/%Y} a {retornos.index[-1]:%m/%Y}",
              f"{len(retornos)} pregões")
    c2.metric("Ativos válidos", len(colunas))
    c3.metric("Selic meta", f"{selic.valor:.2%}", selic.fonte, delta_color="off")

    carteira_hoje(entradas, precos, retornos, rf_aritmetica)
    secao_backtest(entradas, precos, retornos, rf_diaria, rf_aritmetica, cdi, benchmark_disponivel)
    st.markdown("---")
    st.markdown(secoes.LIMITACOES)


def carteira_hoje(entradas: Entradas, precos: pd.DataFrame, retornos: pd.DataFrame,
                  rf_aritmetica: float) -> None:
    st.markdown("## Carteira de hoje")
    momentos = estimar_momentos(retornos, shrink_mu=entradas.shrink_mu)
    mu, cov, growth = momentos.mu, momentos.cov, None
    info = get_ticker_info()

    if entradas.usar_growth:
        acoes = list(retornos.columns)
        setores = tuple((t, info.get(t, {}).get("setor", "?")) for t in acoes)
        with st.spinner("Growth Score (fundamentos Yahoo/CVM)..."):
            growth = servicos.carregar_growth(tuple(acoes), precos[acoes], setores,
                                              entradas.usar_cvm, date.today())
        aprovados = filtrar_por_percentil(growth["score"], entradas.percentil_growth,
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
        st.warning(f"Sem cotação: {nomes}. Fora do plano de compra; o peso alvo fica em caixa.")
    # O HRP ignora o teto de peso: o plano nunca corta um alvo abaixo do próprio peso.
    teto = min(1.0, max(entradas.params.peso_maximo, float(pesos.max())))
    regras = RegrasCompra(entradas.orcamento, entradas.custo_bps, teto, QUANTIDADE_MINIMA_POR_ACAO)
    try:
        plano = plano_de_compra(pesos[com_cotacao], cotacoes["preco"], regras)
    except ValueError as erro:
        st.error(f"Plano de compra indisponível: {erro}")
        return
    secoes.tabela_carteira(plano, cotacoes, info, growth, date.today())
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


def secao_backtest(entradas: Entradas, precos: pd.DataFrame, retornos: pd.DataFrame,
                   rf_diaria: pd.Series, rf_aritmetica: float, cdi: pd.Series,
                   benchmark_disponivel: bool) -> None:
    st.markdown("---")
    st.markdown("## Walk-forward fora da amostra (universo pré-filtro, líquido de custos)")
    espera = ("de 5 estratégias (a reamostrada leva minutos)" if entradas.reamostrar
              else "de 4 estratégias")
    with st.spinner(f"Rodando walk-forward {espera}..."):
        try:
            resultados = servicos.rodar_walk_forward(
                retornos, entradas.params, entradas.shrink_mu, entradas.custo_bps,
                entradas.orcamento, cdi, rf_aritmetica, entradas.reamostrar)
        except ValueError as erro:
            st.info(f"Walk-forward indisponível: {erro}")
            return

    # A ordem do dicionário define a principal: a reamostrada vem primeiro quando ligada.
    nome_otimizada = next(iter(resultados))
    benchmark = None
    if benchmark_disponivel:
        ret_bova = retornos_simples(precos[[BENCHMARK_ACOES]]).reindex(retornos.index).dropna()
        cfg_bt = ConfigBacktest(capital_inicial=entradas.orcamento, custo_bps=entradas.custo_bps)
        if len(ret_bova) > cfg_bt.janela_treino + 1:
            benchmark = walk_forward(ret_bova, estrategia_equiponderada(), cfg_bt, "BOVA11")

    otimizada = resultados[nome_otimizada]
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


if __name__ == "__main__":
    main()
