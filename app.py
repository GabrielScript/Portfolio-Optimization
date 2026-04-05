"""
app.py - Dashboard Streamlit Principal
TCC: Otimização de Carteiras de Investimentos B3
Autor: Gabriel Estrela Lopes

Dashboard interativo para otimização de portfólios usando
a Teoria Moderna do Portfólio de Markowitz.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página (DEVE ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="Otimizador de Carteiras B3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

from assets import ATIVOS_B3, SETORES, get_tickers_by_setor, get_ticker_info, get_all_tickers, get_top75_tickers
from risk_profiles import PERFIS_RISCO, get_perfil, get_nomes_perfis, TAXA_SELIC
from data_loader import carregar_dados_completos, baixar_dados_historicos, baixar_cdi_historico
from optimizer import otimizar_por_perfil, gerar_fronteira_eficiente
from visualizations import (
    grafico_fronteira_eficiente, grafico_composicao_pizza,
    grafico_barras_alocacao, grafico_matriz_correlacao,
    grafico_evolucao_precos,
    grafico_backtesting, grafico_drawdown, grafico_metricas_risco
)
from backtesting import (
    backtesting_walk_forward, backtesting_pesos_fixos, calcular_metricas_risco_portfolio,
    comparar_com_benchmark
)

# Taxa Selic obtida automaticamente via API do BCB (risk_profiles.py)
taxa_selic = TAXA_SELIC


@st.cache_data(show_spinner=False)
def cached_backtest_oos(_precos, perfil, orcamento, n_ativos_max, peso_maximo, _taxa, _serie_cdi):
    return backtesting_walk_forward(
        precos=_precos,
        perfil=perfil,
        janela_treino=252 * 2,
        janela_teste=63,
        capital_inicial=orcamento,
        taxa_livre_risco=_taxa,
        n_ativos_max=n_ativos_max,
        peso_maximo=peso_maximo,
        serie_cdi_diario=_serie_cdi
    )


@st.cache_data(show_spinner=False)
def cached_backtest_is(_precos, pesos_dict, orcamento, _taxa, _serie_cdi):
    return backtesting_pesos_fixos(
        precos=_precos,
        pesos=pesos_dict,
        janela_rebalanceamento=63,
        capital_inicial=orcamento,
        taxa_livre_risco=_taxa,
        serie_cdi_diario=_serie_cdi
    )


# ============== CSS PERSONALIZADO (TEMA ESCURO) ==============
st.markdown("""
<style>
    /* Fundo principal */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 100%);
    }

    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(145deg, #1E2130, #2D3250);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667EEA, #764BA2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value-green {
        font-size: 2.2rem;
        font-weight: 700;
        background: #00D4AA;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value-red {
        font-size: 2.2rem;
        font-weight: 700;
        background: #FF5252;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }

    /* Titulo principal */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #667EEA, #764BA2, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E2130 0%, #0E1117 100%);
    }

    /* Divisores */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667EEA, transparent);
        margin: 2rem 0;
    }

    /* Tabelas */
    .dataframe {
        background: #1E2130 !important;
        border-radius: 8px;
    }

    /* Botoes */
    .stButton > button {
        background: linear-gradient(90deg, #667EEA, #764BA2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* Selic badge */
    .selic-badge {
        background: linear-gradient(145deg, #1a2332, #1E2130);
        border: 1px solid #667EEA44;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        margin: 8px 0;
    }
    .selic-valor {
        font-size: 1.6rem;
        font-weight: 700;
        color: #667EEA;
    }
    .selic-fonte {
        font-size: 0.7rem;
        color: #888;
        margin-top: 4px;
    }

    /* Perfil info card */
    .perfil-info {
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ============== HEADER ==============
    st.markdown('<h1 class="main-title">Otimizador de Carteiras B3</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Teoria Moderna do Portfolio de Markowitz | '
        'TCC - Ciencia de Dados para Negocios - UFPB<br>'
        '<small>Por Gabriel Estrela Lopes</small></p>',
        unsafe_allow_html=True
    )

    # ============== SIDEBAR ==============
    with st.sidebar:
        st.markdown("## Parametros")
        st.markdown("---")

        # ---------- Perfil de Risco ----------
        st.markdown("### Perfil de Risco")
        perfil_nome = st.radio(
            "Selecione seu perfil:",
            options=get_nomes_perfis(),
            horizontal=True,
            help="Define a estrategia de otimizacao"
        )
        perfil = get_perfil(perfil_nome)

        # Cores e objetivos por perfil
        cores_perfil = {"Conservador": "#00D4AA", "Moderado": "#FFB74D", "Agressivo": "#FF5252"}
        objetivos_perfil = {
            "Conservador": "Minimiza Volatilidade",
            "Moderado": "Maximiza Sharpe",
            "Agressivo": "Maximiza Retorno"
        }
        cor = cores_perfil[perfil_nome]

        st.markdown(f"""
        <div class="perfil-info" style="background: {cor}15; border-left: 4px solid {cor};">
            <b>{perfil.icone} {perfil.nome}</b><br>
            <small>{perfil.descricao}</small><br>
            <small style="color: {cor};">Objetivo: {objetivos_perfil[perfil_nome]}</small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ---------- Orcamento ----------
        st.markdown("### Orcamento")
        orcamento = st.number_input(
            "Valor para investir (R$):",
            min_value=1000.0,
            max_value=10000000.0,
            value=10000.0,
            step=1000.0,
            format="%.2f"
        )

        st.markdown("---")

        # ---------- Botao de otimizacao ----------
        otimizar = st.button("OTIMIZAR CARTEIRA", use_container_width=True)

        st.markdown("---")

        # ---------- Taxa Selic (automatica, read-only) ----------
        st.markdown(f"""
        <div class="selic-badge">
            <div class="selic-valor">{taxa_selic*100:.2f}% a.a.</div>
            <div style="color: #aaa; font-size: 0.85rem;">Taxa Selic (Rf)</div>
            <div class="selic-fonte">Obtida via API do Banco Central (SGS 432)</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------- Configuracoes Avancadas ----------
        with st.expander("Configuracoes Avancadas", expanded=False):
            # Setores
            setores_selecionados = st.multiselect(
                "Filtrar por setores:",
                options=SETORES,
                default=[],
                help="Deixe vazio para usar ativos padrao"
            )

            usar_todos_ativos = st.checkbox(
                "Usar todos os ativos (96)",
                value=False,
                help="Mais lento no Cloud. Use apenas se necessario."
            )

            n_ativos_max = st.slider(
                "Maximo de ativos na carteira:",
                min_value=3,
                max_value=30,
                value=10,
                help="Limita o numero de ativos no portfolio"
            )

            periodo_anos = st.slider(
                "Periodo de analise (anos):",
                min_value=1,
                max_value=5,
                value=5,
                help="Quantidade de anos de dados historicos"
            )

            peso_maximo = st.slider(
                "Peso maximo por ativo (%):",
                min_value=5,
                max_value=50,
                value=20,
                help="Limite maximo de alocacao em um unico ativo"
            ) / 100

        # Info de ativos selecionados
        if setores_selecionados:
            n_disponiveis = len(get_tickers_by_setor(setores_selecionados))
        elif usar_todos_ativos:
            n_disponiveis = len(ATIVOS_B3)
        else:
            n_disponiveis = 75

        st.markdown(f"""
        <div style="background: #1E213033; padding: 10px; border-radius: 8px; margin-top: 10px; text-align: center;">
            <small>Ativos disponiveis: <b>{n_disponiveis}</b></small>
        </div>
        """, unsafe_allow_html=True)

    # ============== MAIN CONTENT ==============

    # Obtem tickers filtrados
    if setores_selecionados:
        tickers = get_tickers_by_setor(setores_selecionados)
        st.info(f"Filtrando {len(tickers)} ativos dos setores: {', '.join(setores_selecionados)}")
    elif usar_todos_ativos:
        tickers = get_all_tickers()
        st.warning(f"Usando todos os {len(tickers)} ativos. Pode demorar mais.")
    else:
        tickers = get_top75_tickers()

    if len(tickers) < 3:
        st.error("Selecione setores com pelo menos 3 ativos disponiveis.")
        return

    # Carrega dados
    dados = carregar_dados_completos(tickers, anos=periodo_anos)

    if dados is None:
        st.error("Erro ao carregar dados. Verifique sua conexao.")
        return

    # Info dos dados
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Periodo Analisado", f"{dados['periodo_inicio']} a {dados['periodo_fim']}")
    with col_info2:
        st.metric("Ativos Validos", f"{dados['n_ativos']} ativos")
    with col_info3:
        st.metric("Taxa Selic (Rf)", f"{taxa_selic*100:.2f}% a.a.")

    st.markdown("---")

    # ============== OTIMIZACAO ==============
    params_key = f"{perfil_nome}_{taxa_selic:.4f}_{peso_maximo:.2f}_{n_ativos_max}_{dados['n_ativos']}_{periodo_anos}"

    needs_optimization = (
        otimizar or
        'resultado' not in st.session_state or
        st.session_state.get('params_key') != params_key
    )

    if needs_optimization:
        with st.spinner(f"Otimizando carteira ({perfil_nome})..."):
            resultado = otimizar_por_perfil(
                perfil=perfil_nome,
                retornos_medios=dados['retornos_medios'],
                matriz_cov=dados['matriz_cov'],
                taxa_livre_risco=taxa_selic,
                peso_maximo=peso_maximo,
                n_ativos_max=n_ativos_max
            )

            fronteira = gerar_fronteira_eficiente(
                dados['retornos_medios'],
                dados['matriz_cov'],
                taxa_selic,
                n_pontos=30,
                peso_maximo=peso_maximo,
                n_ativos_max=n_ativos_max
            )

            st.session_state['resultado'] = resultado
            st.session_state['fronteira'] = fronteira
            st.session_state['params_key'] = params_key
    else:
        resultado = st.session_state['resultado']
        fronteira = st.session_state['fronteira']

    if not resultado.sucesso:
        st.warning(f"Otimizacao com aviso: {resultado.mensagem}")

    # ============== METRICAS PRINCIPAIS ==============
    st.markdown("## Resultados da Otimizacao")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{resultado.retorno_esperado*100:.2f}%</div>
            <div class="metric-label">Retorno Esperado (a.a.)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{resultado.volatilidade*100:.2f}%</div>
            <div class="metric-label">Volatilidade (a.a.)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        sharpe_class = "metric-value-green" if resultado.sharpe > 0 else "metric-value-red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="{sharpe_class}">{resultado.sharpe:.3f}</div>
            <div class="metric-label">Indice de Sharpe</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        retorno_reais = orcamento * resultado.retorno_esperado
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">R$ {retorno_reais:,.0f}</div>
            <div class="metric-label">Retorno Anual Estimado</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============== GRAFICOS ==============

    pesos_dict = {t: p for t, p in zip(resultado.tickers, resultado.pesos) if p > 0.001}
    info_tickers = get_ticker_info()

    # Benchmark 1/N: retorno e volatilidade esperados (espaço média-variância)
    n_all = dados['n_ativos']
    pesos_1n_arr = np.ones(n_all) / n_all
    ret_1n_frontier = float(np.dot(pesos_1n_arr, dados['retornos_medios'].values))
    cov_vals = dados['matriz_cov'].values
    vol_1n_frontier = float(np.sqrt(pesos_1n_arr.T @ cov_vals @ pesos_1n_arr))
    benchmark_1n = {'retorno': ret_1n_frontier, 'volatilidade': vol_1n_frontier}

    # Linha 1: Fronteira + Composicao
    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig_fronteira = grafico_fronteira_eficiente(
            fronteira,
            {'retorno': resultado.retorno_esperado, 'volatilidade': resultado.volatilidade},
            perfil_nome,
            taxa_selic,
            benchmark_1n=benchmark_1n
        )
        st.plotly_chart(fig_fronteira, use_container_width=True)

    with col2:
        fig_pizza = grafico_composicao_pizza(pesos_dict, orcamento)
        st.plotly_chart(fig_pizza, use_container_width=True)

    # Linha 2: Barras
    fig_barras = grafico_barras_alocacao(pesos_dict, orcamento, info_tickers)
    st.plotly_chart(fig_barras, use_container_width=True)

    st.markdown("---")

    # ============== TABELA DE ALOCACAO ==============
    st.markdown("## Detalhamento da Carteira")

    ativos_carteira = []
    for ticker, peso in sorted(pesos_dict.items(), key=lambda x: -x[1]):
        info = info_tickers.get(ticker, {})
        ativos_carteira.append({
            'Ticker': ticker.replace('.SA', ''),
            'Nome': info.get('nome', '-'),
            'Setor': info.get('setor', '-'),
            'Peso (%)': f"{peso*100:.2f}%",
            'Valor (R$)': f"R$ {peso*orcamento:,.2f}"
        })

    df_carteira = pd.DataFrame(ativos_carteira)
    st.dataframe(df_carteira, use_container_width=True, hide_index=True)

    # ============== ANALISES ADICIONAIS ==============
    with st.expander("Ver Matriz de Correlacao", expanded=False):
        tickers_carteira = [t for t, p in pesos_dict.items() if p > 0.01]
        if len(tickers_carteira) > 1:
            matriz_filtrada = dados['matriz_cov'].loc[tickers_carteira, tickers_carteira]
            fig_corr = grafico_matriz_correlacao(matriz_filtrada)
            st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Ver Evolucao Historica dos Precos", expanded=False):
        tickers_carteira = [t for t, p in pesos_dict.items() if p > 0.02][:10]
        if tickers_carteira:
            fig_evolucao = grafico_evolucao_precos(dados['precos'], tickers_carteira)
            st.plotly_chart(fig_evolucao, use_container_width=True)

    with st.expander("Ver Metricas Individuais dos Ativos", expanded=False):
        st.dataframe(dados['metricas'], use_container_width=True)

    # ============== BACKTESTING E ANALISE DE RISCO ==============
    st.markdown("---")
    st.markdown("## Backtesting e Analise de Risco")
    st.markdown("*Simulacao historica da carteira otimizada*")

    tipo_backtest = st.radio(
        "Metodologia de Backtesting:",
        options=["Walk-Forward (Out-of-Sample)", "Pesos Fixos (In-Sample)"],
        horizontal=True,
        help=(
            "Walk-Forward: re-otimiza a cada trimestre usando apenas dados passados (sem vies de antecipacao). "
            "Pesos Fixos: aplica os pesos atuais retroativamente (contem look-ahead bias)."
        )
    )

    with st.spinner('Executando backtesting...'):
        # Busca série histórica do CDI diário para Sharpe/Sortino corretos
        serie_cdi = baixar_cdi_historico(anos=periodo_anos)
        if serie_cdi.empty:
            st.warning("CDI histórico indisponível. Sharpe/Sortino usarão Selic constante como fallback.")

        if "Walk-Forward" in tipo_backtest:
            try:
                backtest = cached_backtest_oos(
                    _precos=dados['precos'],
                    perfil=perfil_nome,
                    orcamento=orcamento,
                    n_ativos_max=n_ativos_max,
                    peso_maximo=peso_maximo,
                    _taxa=taxa_selic,
                    _serie_cdi=serie_cdi
                )
            except Exception as e:
                st.error(f"Erro no Walk-Forward: {e}. Usando Pesos Fixos como fallback.")
                backtest = cached_backtest_is(
                    _precos=dados['precos'],
                    pesos_dict=pesos_dict,
                    orcamento=orcamento,
                    _taxa=taxa_selic,
                    _serie_cdi=serie_cdi
                )
        else:
            backtest = cached_backtest_is(
                _precos=dados['precos'],
                pesos_dict=pesos_dict,
                orcamento=orcamento,
                _taxa=taxa_selic,
                _serie_cdi=serie_cdi
            )

        metricas_risco = calcular_metricas_risco_portfolio(
            retornos=dados['retornos'],
            pesos=pesos_dict,
            taxa_livre_risco=taxa_selic
        )

        # Benchmark 1/N (carteira equiponderada) — exigido pela Q1 do TCC
        pesos_1n = {t: 1.0 / dados['n_ativos'] for t in dados['tickers']}
        backtest_1n = cached_backtest_is(
            _precos=dados['precos'],
            pesos_dict=pesos_1n,
            orcamento=orcamento,
            _taxa=taxa_selic,
            _serie_cdi=serie_cdi
        )

        # Dados do Ibovespa para comparacao
        try:
            ibov = baixar_dados_historicos(['^BVSP'], anos=periodo_anos)
            if not ibov.empty:
                ibov_serie = ibov['^BVSP'] if '^BVSP' in ibov.columns else ibov.iloc[:, 0]
            else:
                ibov_serie = None
        except Exception:
            ibov_serie = None

    # Metricas de Backtesting
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">R$ {backtest['capital_final']:,.0f}</div>
            <div class="metric-label">Capital Final</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ret_class = "metric-value-green" if backtest['retorno_total'] > 0 else "metric-value-red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="{ret_class}">{backtest['retorno_total']*100:.1f}%</div>
            <div class="metric-label">Retorno Total</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value-red">{backtest['max_drawdown']*100:.1f}%</div>
            <div class="metric-label">Drawdown Maximo</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        sortino_class = "metric-value-green" if backtest['sortino'] > 1 else "metric-value"
        st.markdown(f"""
        <div class="metric-card">
            <div class="{sortino_class}">{backtest['sortino']:.2f}</div>
            <div class="metric-label">Indice Sortino</div>
        </div>
        """, unsafe_allow_html=True)

    # Grafico de Backtesting
    col1, col2 = st.columns([1.5, 1])

    with col1:
        fig_backtest = grafico_backtesting(
            serie_carteira=backtest['serie_carteira'],
            serie_benchmark=ibov_serie,
            nome_benchmark="Ibovespa",
            serie_1n=backtest_1n['serie_carteira']
        )
        st.plotly_chart(fig_backtest, use_container_width=True)

    with col2:
        fig_risco = grafico_metricas_risco(metricas_risco)
        st.plotly_chart(fig_risco, use_container_width=True)

    # Grafico de Drawdown
    fig_dd = grafico_drawdown(backtest['drawdown_serie'])
    st.plotly_chart(fig_dd, use_container_width=True)

    # ============== TABELA COMPARATIVA: Otimizada vs 1/N vs Ibovespa ==============
    st.markdown("### Comparacao: Otimizada vs Benchmark 1/N")

    comparacao_data = {
        'Metrica': [
            'Retorno Total',
            'Retorno Anualizado',
            'Volatilidade',
            'Sharpe',
            'Sortino',
            'Max Drawdown',
            'VaR 95% (anual)',
            'CVaR 95% (anual)'
        ],
        f'Carteira Otimizada ({perfil_nome})': [
            f"{backtest['retorno_total']*100:.2f}%",
            f"{backtest['retorno_anualizado']*100:.2f}%",
            f"{backtest['volatilidade']*100:.2f}%",
            f"{backtest['sharpe']:.3f}",
            f"{backtest['sortino']:.3f}",
            f"{backtest['max_drawdown']*100:.2f}%",
            f"{backtest.get('var_95_anual', 0)*100:.2f}%",
            f"{backtest.get('cvar_95_anual', 0)*100:.2f}%"
        ],
        'Benchmark 1/N': [
            f"{backtest_1n['retorno_total']*100:.2f}%",
            f"{backtest_1n['retorno_anualizado']*100:.2f}%",
            f"{backtest_1n['volatilidade']*100:.2f}%",
            f"{backtest_1n['sharpe']:.3f}",
            f"{backtest_1n['sortino']:.3f}",
            f"{backtest_1n['max_drawdown']*100:.2f}%",
            f"{backtest_1n.get('var_95_anual', 0)*100:.2f}%",
            f"{backtest_1n.get('cvar_95_anual', 0)*100:.2f}%"
        ]
    }

    df_comparacao = pd.DataFrame(comparacao_data)
    st.dataframe(df_comparacao, use_container_width=True, hide_index=True)

    # Interpretacao das Metricas
    with st.expander("Entenda as Metricas de Risco", expanded=False):
        st.markdown(f"""
        ### Value at Risk (VaR) - {metricas_risco.get('var_95_anual', 0)*100:.2f}% anual
        {metricas_risco.get('interpretacao_var', '')}

        ### Conditional VaR (CVaR) - {metricas_risco.get('cvar_95_anual', 0)*100:.2f}% anual
        {metricas_risco.get('interpretacao_cvar', '')}

        ### Drawdown Maximo - {backtest['max_drawdown']*100:.1f}%
        A maior queda do pico ao vale durante o periodo analisado.
        Durou aproximadamente **{backtest['duracao_max_drawdown']} dias**.

        ### Indice de Sortino - {backtest['sortino']:.2f}
        Similar ao Sharpe, mas considera apenas volatilidade negativa (risco de perda).
        - **> 1**: Bom | **> 2**: Excelente | **< 0**: Retorno abaixo da taxa livre de risco

        ### Curtose - {metricas_risco.get('curtose', 0):.2f}
        Mede a probabilidade de eventos extremos (caudas pesadas).
        - **> 3**: Mais eventos extremos que uma distribuicao normal
        - **< 3**: Menos eventos extremos
        """)

    # ============== LIMITACOES ==============
    st.markdown("---")
    st.markdown("## Limitacoes e Vieses do Estudo")

    st.info("""
    **Vies de Sobrevivencia (Survivor Bias):** A lista de ativos analisados e composta pelos componentes atuais
    (ou recentes) da B3. Empresas que faliram ou fecharam capital podem ter saido das listas,
    o que pode enviesar as simulacoes positivamente.
    """)

    st.success("""
    **Taxa de Risco Variável (CDI Diário):** Os índices de Sharpe e Sortino do backtest utilizam
    a série histórica do CDI diário (BCB SGS série 12) como taxa livre de risco,
    calculando o excesso de retorno dia a dia. Isso corrige a distorção de usar
    uma Selic constante sobre períodos onde a taxa variou de ~2% (2020) a ~15% (2026).
    """)

    st.info("""
    **Custos Transacionais:** Corretagem, emolumentos e impostos nao sao deduzidos nas simulacoes.
    O rendimento liquido real seria inferior aos retornos brutos reportados.
    """)

    # ============== FOOTER ==============
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><b>TCC - Otimizacao de Carteiras de Investimentos</b></p>
        <p>Teoria Moderna do Portfolio de Markowitz</p>
        <p><b>Gabriel Estrela Lopes</b> | Ciencia de Dados para Negocios - UFPB | 2026</p>
        <p style="font-size: 0.75rem; color: #555;">
            Taxa Selic obtida em tempo real via API do Banco Central do Brasil (SGS serie 432)
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
