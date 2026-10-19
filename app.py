"""
app.py - Dashboard Streamlit (camada fina sobre o pacote ``core``).
Otimizador de carteiras de ações da B3 para o investidor arrojado.
Autor: Gabriel Estrela Lopes

Fluxo: sidebar -> preços das ações -> momentos (Ledoit-Wolf +
Jorion) -> otimização -> walk-forward de todas as estratégias. O Growth Score
(core/growth.py) fica fora da tela até ser validado fora da amostra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Otimizador Arrojado B3", page_icon="🚀", layout="wide")

from assets import (BENCHMARK_ACOES, BENCHMARKS, SETORES, get_all_tickers, get_ticker_info,  # noqa: E402
                    get_tickers_by_setor, get_tickers_liquidos)
from core.backtest import ResultadoBacktest, estrategia_equiponderada, walk_forward  # noqa: E402
from core.config import (DIAS_UTEIS_ANO, ConfigBacktest, ConfigOtimizacao, Objetivo,  # noqa: E402
                         taxa_anual_para_aritmetica)
from core.estimation import estimar_momentos, retornos_simples  # noqa: E402
from core import lago  # noqa: E402
from core.market_data import serie_rf_diaria  # noqa: E402
from core.metrics import contribuicao_risco, drawdown, metricas_rolantes  # noqa: E402
from core.optimization import (ErroInviavel, ErroOtimizacao, ResultadoOtimizacao,  # noqa: E402
                               combinar_nucleo_satelite, fronteira_eficiente, metricas_carteira, otimizar,
                               preparar_covariancia, projetar_pesos)
from core.ordens import (RegrasCompra, RegrasDinheiroReal, acoes_que_cabem, limites_do_plano,  # noqa: E402
                         plano_de_compra, plano_dinheiro_real, redistribuir_sem_cotacao)
from core.risco import LIMITES, Violacao, checar_carteira, validar_config  # noqa: E402
from ui import secoes, servicos  # noqa: E402
from visualizations import (grafico_barras_alocacao, grafico_composicao_pizza,  # noqa: E402
                            grafico_contribuicao_risco, grafico_curvas_capital, grafico_drawdown,
                            grafico_fronteira_eficiente, grafico_matriz_correlacao,
                            grafico_metricas_rolling, grafico_pesos_historicos)

# Padrão da carteira do estudo pela regra B do S1-P.b (docs/universo-dinamico-2026-10-18.txt): núcleo Mín. Vol +
# satélite Kelly λ=2 com 30%. Nenhum objetivo é "recomendado": com dinheiro real vale o plano BOVA11 + satélite do
# paper (DECISOES 69); pela regra A (opção 3), o Kelly sozinho fica como opção.
NUCLEO_SATELITE = "nucleo_satelite"
ROTULOS_OBJETIVO = {
    NUCLEO_SATELITE: "Núcleo Mín. Vol + satélite Kelly (padrão do estudo, só ações)",
    Objetivo.KELLY.value: "Crescimento máximo (Kelly)",
    Objetivo.MAX_SHARPE.value: "Máximo Sharpe (exato)",
    Objetivo.MIN_VOL.value: "Mínima variância",
    Objetivo.HRP.value: "HRP (hierárquico, projetado nos tetos)",
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
    "Soma máxima dos pesos das ações de um mesmo setor. Regra fixa de 30%: a tela só aperta. Vale para a carteira "
    "de hoje (o HRP é projetado nos tetos), o plano de compra e o walk-forward."
)
AJUDA_SATELITE = (
    "Parte das ações no satélite Kelly; o resto fica no núcleo de Mínima Variância. Regra fixa: no máximo 50%. "
    "O padrão de 30% é a mistura aprovada no teste pré-registrado sem viés de sobrevivência (S1-P.b)."
)
AVISO_REGRAS_DE_RISCO = (
    "Regras de risco fixas: sem alavancagem, até 20% por ação e por emissor, 30% por setor e 40% a.a. de "
    "volatilidade ex-ante; carteira fora delas não vira plano. Com dinheiro real, o núcleo fica no BOVA11 até o "
    "critério de pronto (alfa líquido contra o BOVA11 com PSR ≥ 0,95 e 6 a 12 meses de paper trading). O stop do "
    "satélite (razão satélite/BOVA11 cair 25% desde o pico) e a pausa de um pregão depois de perda diária de 5% "
    "só valem com as posições registradas no paper trading; até lá, são aviso."
)
# Plano de compra (DECISOES 69): o padrão é a regra do dinheiro real até o critério de pronto, BOVA11 + o satélite
# registrado no paper; a carteira otimizada da barra lateral é o estudo, com o plano dela por escolha.
MODO_DINHEIRO_REAL, MODO_ESTUDO = "dinheiro_real", "estudo"
ROTULOS_MODO = {MODO_DINHEIRO_REAL: "Dinheiro real", MODO_ESTUDO: "Estudo"}
AJUDA_MODO = (
    "Dinheiro real: BOVA11 com ~70% e o satélite registrado no paper trading com ~30% (a regra até o critério de "
    "pronto). Estudo: o plano da carteira otimizada pela barra lateral, só ações, para comparar."
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
    shrink_mu: bool
    custo_bps: float
    reamostrar: bool
    modo_plano: str = MODO_DINHEIRO_REAL


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
        objetivo = st.selectbox("Objetivo", list(ROTULOS_OBJETIVO), format_func=ROTULOS_OBJETIVO.get,
                                key="objetivo")
        com_kelly = objetivo in (Objetivo.KELLY.value, NUCLEO_SATELITE)
        aversao = st.slider("Aversão a risco λ (1 = Kelly pleno)", 0.5, 5.0, 2.0, 0.5,
                            disabled=not com_kelly,
                            help="λ=1 aproxima a carteira de maior crescimento geométrico. Como μ tem "
                                 "erro de estimação, λ≈2 é mais robusto. Sem caixa no conjunto "
                                 "viável, λ=2 NÃO equivale a 'meio Kelly'.")
        fracao_satelite = st.slider("Satélite Kelly (% das ações)", 0, round(LIMITES.fracao_maxima_satelite * 100),
                                    round(LIMITES.fracao_padrao_satelite * 100), 5, key="fracao_satelite",
                                    disabled=objetivo != NUCLEO_SATELITE, help=AJUDA_SATELITE) / 100
        teto_vol = round(LIMITES.vol_maxima * 100)
        vol_max = st.slider("Teto de volatilidade (% a.a.)", 15, teto_vol, teto_vol, 5, key="vol_maxima") / 100
        orcamento = st.number_input("Valor a investir (R$)", 100.0, 10_000_000.0, 10_000.0, 100.0,
                                    key="orcamento")
        modo_plano = st.segmented_control("Plano de compra", list(ROTULOS_MODO), format_func=ROTULOS_MODO.get,
                                          default=MODO_DINHEIRO_REAL, key="modo_plano", help=AJUDA_MODO)

        st.markdown("### Universo")
        setores = st.multiselect("Setores (vazio = lista líquida)", SETORES, key="setores")
        todos = st.checkbox(f"Todos os ativos ({len(get_all_tickers())})", value=False)

        with st.expander("Avançado"):
            n_max = st.slider("Máximo de ativos", 3, 30, 10, key="n_ativos_max",
                              help="Na mistura núcleo + satélite, vale para cada parte: a carteira pode ter até o "
                                   "dobro de ativos.")
            teto_acao = round(LIMITES.peso_maximo_acao * 100)
            peso_max = st.slider("Peso máximo por ativo (%)", 5, teto_acao, teto_acao, key="peso_maximo") / 100
            teto_setor = st.slider("Teto por setor (%)", 20, round(LIMITES.teto_setor * 100),
                                   round(LIMITES.teto_setor * 100), 5, key="teto_setor", help=AJUDA_TETO_SETOR) / 100
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
    # o teto de vol vale para todo objetivo (regra fixa); na mistura, o objetivo do satélite é o Kelly
    params = servicos.ParametrosOtimizacao(
        objetivo=Objetivo.KELLY.value if objetivo == NUCLEO_SATELITE else objetivo, peso_maximo=peso_max,
        n_ativos_max=n_max, vol_maxima=vol_max, aversao_risco=aversao, teto_setor=teto_setor,
        setores=tuple(sorted((t, info[t]["setor"]) for t in acoes)),
        fracao_satelite=fracao_satelite if objetivo == NUCLEO_SATELITE else None,
    )
    # O checkbox desabilitado guarda o último valor: a reamostragem só vale para o Kelly.
    return Entradas(params, tuple(acoes), anos, anos_teste, orcamento, shrink, float(custo),
                    reamostrar and objetivo == Objetivo.KELLY.value, modo_plano or MODO_DINHEIRO_REAL)


def validar(entradas: Entradas) -> str | None:
    p = entradas.params
    if p.n_ativos_max * p.peso_maximo < 1.0:
        return (f"Inviável: {p.n_ativos_max} ativos × {p.peso_maximo:.0%} = "
                f"{p.n_ativos_max * p.peso_maximo:.0%} < 100%. Aumente um dos dois.")
    if len(entradas.tickers_acoes) < 3:
        return "Selecione setores com pelo menos 3 ativos."
    violacoes = validar_config(p.para_config(0.0))
    if violacoes:
        return "Configuração fora das regras de risco fixas: " + _descrever(violacoes)
    return None


def _descrever(violacoes: list[Violacao]) -> str:
    return "; ".join(f"{v.regra.value} em {v.alvo.replace('.SA', '')}: {v.valor:.2%} (limite {v.limite:.0%})"
                     for v in violacoes) + '.'


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
    versao_dados = lago.versao_dos_dados()
    with st.spinner("Carregando preços, CDI e Selic..."):
        dados = servicos.carregar_precos(universo + (BENCHMARK_ACOES,), entradas.anos, hoje, versao_dados)
        precos_walk_forward = servicos.carregar_precos(universo + BENCHMARKS, anos_walk_forward,
                                                       hoje, versao_dados).precos
        cdi = servicos.carregar_cdi(anos_walk_forward, hoje)
        selic = servicos.carregar_selic(hoje)
    aviso = secoes.aviso_descartados(dados.descartados_sem_dados, dados.descartados_cobertura)
    if aviso:
        st.warning(aviso)
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
    st.caption(f"Preços: lago de dados da B3 (COTAHIST e eventos da B3/CVM, retorno total), até {precos.index[-1]:%d/%m/%Y}.")

    carteira_hoje(entradas, precos, retornos, rf_aritmetica)
    secao_backtest(entradas, precos_walk_forward, colunas, cdi, selic.valor)
    st.markdown("---")
    st.markdown(secoes.LIMITACOES)
    secoes.tabela_sem_vies()


def carteira_hoje(entradas: Entradas, precos: pd.DataFrame, retornos: pd.DataFrame,
                  rf_aritmetica: float) -> None:
    st.markdown("## Carteira de hoje")
    acoes = list(retornos.columns)
    # a trava do orçamento só vale para o plano da carteira otimizada: no dinheiro real, o satélite é o registrado
    compraveis = acoes_compraveis(entradas, precos[acoes]) if entradas.modo_plano == MODO_ESTUDO else acoes
    momentos = estimar_momentos(retornos, shrink_mu=entradas.shrink_mu)
    mu, cov = momentos.mu[compraveis], momentos.cov.loc[compraveis, compraveis]
    info = get_ticker_info()

    cfg = entradas.params.para_config(rf_aritmetica)
    try:
        if entradas.reamostrar:
            carteira = _carteira_reamostrada(entradas, retornos[list(mu.index)], rf_aritmetica)
        else:
            resultado = (_nucleo_satelite(mu, cov, cfg, entradas.params.fracao_satelite)
                         if entradas.params.fracao_satelite is not None else otimizar(mu, cov, cfg))
            if resultado.objetivo is Objetivo.HRP:
                resultado = _hrp_nos_tetos(resultado, mu, cov, cfg)
            carteira = CarteiraCalculada(resultado, mu, cov, momentos.intensidade_shrinkage)
        fronteira = fronteira_eficiente(carteira.mu, carteira.cov, cfg)
    except ErroInviavel as erro:
        st.error(f"Carteira inviável: {erro} Inclua mais setores ou mais ativos: o teto por setor é de no máximo "
                 f"{LIMITES.teto_setor:.0%} e o peso máximo, de {LIMITES.peso_maximo_acao:.0%} (regras fixas).")
        st.stop()
    except ErroOtimizacao as erro:
        st.error(f"Otimização falhou: {erro}")
        st.stop()
    resultado, mu, cov = carteira.resultado, carteira.mu, carteira.cov
    violacoes = checar_carteira(resultado.pesos, dict(entradas.params.setores), cov)
    if violacoes:
        st.error("Carteira recusada pelas regras de risco fixas, sem plano de compra: " + _descrever(violacoes))
        return
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
    if entradas.modo_plano == MODO_ESTUDO:
        secao_plano_de_compra(entradas, pesos, precos, info)
    else:
        secao_dinheiro_real(entradas, info)
    st.info(AVISO_REGRAS_DE_RISCO)

    col_rc, col_corr = st.columns(2)
    col_rc.plotly_chart(grafico_contribuicao_risco(contribuicao_risco(pesos, cov), pesos.to_dict()),
                        width="stretch")
    if len(pesos) > 1:
        col_corr.plotly_chart(grafico_matriz_correlacao(cov.loc[pesos.index, pesos.index]), width="stretch")
    if carteira.frequencia is not None:
        with st.expander("Estabilidade nas reamostras"):
            secoes.tabela_estabilidade(resultado.pesos, carteira.frequencia)


def _nucleo_satelite(mu: pd.Series, cov: pd.DataFrame, cfg: ConfigOtimizacao,
                     fracao_satelite: float) -> ResultadoOtimizacao:
    """Núcleo Mín. Vol + satélite (o objetivo do ``cfg``) nos mesmos μ e Σ; a mistura de viáveis é viável."""
    nucleo = otimizar(mu, cov, replace(cfg, objetivo=Objetivo.MIN_VOL))
    satelite = otimizar(mu, cov, cfg)
    pesos = combinar_nucleo_satelite(nucleo.pesos, satelite.pesos, fracao_satelite).reindex(mu.index)
    retorno, vol, sharpe = metricas_carteira(pesos.to_numpy(), mu.to_numpy(), preparar_covariancia(cov),
                                             cfg.taxa_livre_risco)
    status = next((s for s in ("fallback", "impreciso") if s in (nucleo.status, satelite.status)), "otimo")
    mensagem = (f"Núcleo Mín. Vol {1 - fracao_satelite:.0%} + satélite Kelly λ={cfg.aversao_risco:g} "
                f"{fracao_satelite:.0%}. Núcleo: {nucleo.mensagem} Satélite: {satelite.mensagem}")
    return ResultadoOtimizacao(pesos=pesos, retorno_esperado=retorno, volatilidade=vol, sharpe=sharpe,
                               objetivo=cfg.objetivo, status=status, mensagem=mensagem)


def _hrp_nos_tetos(resultado: ResultadoOtimizacao, mu: pd.Series, cov: pd.DataFrame,
                   cfg: ConfigOtimizacao) -> ResultadoOtimizacao:
    """O HRP da tela vira ordem: projetado nos tetos por ação, setor, vol e nº de ativos (o das comparações, não)."""
    projetados = projetar_pesos(resultado.pesos, cov, cfg)
    retorno, vol, sharpe = metricas_carteira(projetados.pesos.to_numpy(), mu.to_numpy(), preparar_covariancia(cov),
                                             cfg.taxa_livre_risco)
    mensagem = f"HRP projetado nos tetos das regras de risco. {projetados.mensagem}".strip()
    return replace(resultado, pesos=projetados.pesos, retorno_esperado=retorno, volatilidade=vol, sharpe=sharpe,
                   status=projetados.status, mensagem=mensagem)


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


def secao_plano_de_compra(entradas: Entradas, pesos: pd.Series, precos: pd.DataFrame, info: dict) -> None:
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
    # O plano nunca corta um alvo abaixo do próprio peso (a carteira já passou pelas regras de risco).
    teto = min(1.0, max(entradas.params.peso_maximo, float(pesos.max())))
    # O valor das ações sem cotação vai para as outras, na proporção do otimizador, sem furar os tetos.
    alvo = redistribuir_sem_cotacao(pesos, com_cotacao, teto, limites_do_plano(entradas.params.grupos, pesos))
    regras = RegrasCompra(entradas.orcamento, entradas.custo_bps, teto, QUANTIDADE_MINIMA_POR_ACAO,
                          limites_do_plano(entradas.params.grupos, alvo))
    try:
        plano = plano_de_compra(alvo, cotacoes["preco"], regras)
    except ValueError as erro:
        st.error(f"Plano de compra indisponível: {erro}")
        return
    # quantidades inteiras: os pesos efetivos passam de novo pelas regras (a vol ex-ante foi conferida no alvo)
    violacoes = checar_carteira(plano.itens["peso_efetivo"], dict(entradas.params.setores))
    if violacoes:
        st.error("Plano de compra recusado pelas regras de risco fixas: " + _descrever(violacoes))
        return
    secoes.tabela_carteira(plano, cotacoes, info, date.today())
    secoes.legenda_por_setor(plano, info)
    secoes.resumo_plano(plano, regras)


def secao_dinheiro_real(entradas: Entradas, info: dict) -> None:
    st.markdown("### Plano de compra: dinheiro real (a regra até o critério de pronto)")
    data, satelite = servicos.carregar_satelite_registrado()
    st.warning(secoes.aviso_dinheiro_real(data))
    tickers = (*satelite.index, BENCHMARK_ACOES)
    hoje = date.today()
    janela = servicos.carregar_precos(tickers, 1, hoje, lago.versao_dos_dados()).precos.tail(PREGOES_PARA_FECHAMENTO)
    if st.button("Atualizar cotações", icon=":material/refresh:", key="atualizar_cotacoes_real"):
        servicos.descartar_cotacoes(tickers, janela)
    with st.spinner("Buscando as cotações do dia…"):
        cotacoes = servicos.carregar_cotacoes(tickers, janela)
    sem_cotacao = [t.removesuffix(".SA") for t in tickers if not cotacoes["preco"].get(t, float("nan")) > 0]
    if sem_cotacao:
        st.error(f"Plano indisponível, sem cotação: {', '.join(sem_cotacao)}.")
        return
    setores = {ticker: info.get(ticker, {}).get("setor", "—") for ticker in satelite.index}
    plano = plano_dinheiro_real(satelite, cotacoes["preco"], setores,
                                RegrasDinheiroReal(total=entradas.orcamento, custo_bps=entradas.custo_bps,
                                                   quantidade_minima=QUANTIDADE_MINIMA_POR_ACAO,
                                                   nucleo=BENCHMARK_ACOES))
    secoes.tabela_dinheiro_real(plano, cotacoes, info, hoje, entradas.orcamento, BENCHMARK_ACOES)


def _carteira_reamostrada(entradas: Entradas, retornos: pd.DataFrame, rf_aritmetica: float) -> CarteiraCalculada:
    """Reestima μ e Σ nas ações compráveis: são esses momentos que a tela mostra."""
    with st.spinner("Reamostrando: 100 carteiras Kelly..."):
        reamostrada = servicos.otimizar_reamostrado(retornos, entradas.params, rf_aritmetica, entradas.shrink_mu)
    return CarteiraCalculada(reamostrada.resultado, reamostrada.mu, reamostrada.cov,
                             reamostrada.intensidade_shrinkage, reamostrada.frequencia)


def secao_backtest(entradas: Entradas, precos: pd.DataFrame, acoes_hoje: list[str], cdi: pd.Series,
                   selic: float) -> None:
    """``precos``: download de janela + teste anos, só do walk-forward; ``acoes_hoje``: universo da carteira de hoje."""
    st.markdown("---")
    st.markdown("## Walk-forward fora da amostra (líquido de custos)")
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
    referencias = _referencias_de_mercado(precos, retornos.index, cfg_bt)

    if otimizada.falhas:
        st.warning(f"{len(otimizada.falhas)} rebalanceamento(s) mantiveram a carteira por falha: "
                   + "; ".join(otimizada.falhas[:3]))
    secoes.tabela_comparativa(resultados, rf_diaria, referencias)
    sem_historico = [_nome(t) for t in BENCHMARKS if _nome(t) not in referencias]
    if sem_historico:
        st.caption(f"Sem histórico para os {anos_historico} anos do walk-forward: {', '.join(sem_historico)}.")
    if _nome(BENCHMARK_ACOES) in referencias:          # o critério pré-registrado é o alfa contra o BOVA11
        secoes.bloco_alfa(otimizada, referencias[_nome(BENCHMARK_ACOES)], rf_diaria)

    curvas = {nome: bt.valores for nome, bt in {**resultados, **referencias}.items()}
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


def _nome(ticker: str) -> str:
    return ticker.replace(".SA", "")


def _referencias_de_mercado(precos: pd.DataFrame, datas: pd.DatetimeIndex,
                            cfg_bt: ConfigBacktest) -> dict[str, ResultadoBacktest]:
    """Walk-forward de cada referência com histórico na janela: mesmas datas, mesmo cfg (treino, custos)."""
    referencias = {}
    for ticker in BENCHMARKS:
        if ticker not in precos.columns:
            continue
        retornos = retornos_simples(precos[[ticker]]).reindex(datas).dropna()
        if len(retornos) > cfg_bt.janela_treino + 1:
            referencias[_nome(ticker)] = walk_forward(retornos, estrategia_equiponderada(), cfg_bt, _nome(ticker))
    return referencias


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
