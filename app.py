"""
app.py - Dashboard Streamlit Principal
TCC: Otimização de Carteiras de Investimentos B3
Autor: Gabriel

Dashboard interativo para otimização de portfólios usando
a Teoria Moderna do Portfólio de Markowitz.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página (DEVE ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="📊 Otimizador de Carteiras B3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

from assets import ATIVOS_B3, SETORES, get_tickers_by_setor, get_ticker_info, get_all_tickers, get_top75_tickers
from risk_profiles import PERFIS_RISCO, get_perfil, get_nomes_perfis, TAXA_SELIC
from data_loader import carregar_dados_completos, baixar_dados_historicos
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

@st.cache_data(show_spinner=False)
def cached_backtest_oos(_precos, perfil, orcamento, taxa_selic, n_ativos_max):
    return backtesting_walk_forward(
        precos=_precos,
        perfil=perfil,
        janela_treino=252 * 2,
        janela_teste=63,
        capital_inicial=orcamento,
        taxa_livre_risco=taxa_selic,
        n_ativos_max=n_ativos_max
    )

@st.cache_data(show_spinner=False)
def cached_backtest_is(_precos, pesos_dict, orcamento, taxa_selic):
    return backtesting_pesos_fixos(
        precos=_precos,
        pesos=pesos_dict,
        janela_rebalanceamento=63,
        capital_inicial=orcamento,
        taxa_livre_risco=taxa_selic
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
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667EEA, #764BA2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    
    /* Título principal */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #667EEA, #764BA2, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Perfil cards */
    .perfil-card {
        background: linear-gradient(145deg, #1E2130, #2D3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .perfil-conservador { border-color: #00D4AA; }
    .perfil-moderado { border-color: #FFB74D; }
    .perfil-agressivo { border-color: #FF5252; }
    
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
    
    /* Botões */
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
</style>
""", unsafe_allow_html=True)


def main():
    # ============== HEADER ==============
    st.markdown('<h1 class="main-title">📊 Otimizador de Carteiras B3</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Teoria Moderna do Portfólio de Markowitz | TCC - Ciência de Dados para Negócios - UFPB<br><small>Por Gabriel Estrela Lopes</small></p>', unsafe_allow_html=True)
    
    # ============== SIDEBAR ==============
    with st.sidebar:
        st.markdown("## ⚙️ Parâmetros")
        st.markdown("---")
        
        # Perfil de Risco
        st.markdown("### 🎯 Perfil de Risco")
        perfil_nome = st.radio(
            "Selecione seu perfil:",
            options=get_nomes_perfis(),
            horizontal=True,
            help="Define a estratégia de otimização"
        )
        perfil = get_perfil(perfil_nome)
        
        # Info do perfil
        cores_perfil = {"Conservador": "#00D4AA", "Moderado": "#FFB74D", "Agressivo": "#FF5252"}
        st.markdown(f"""
        <div style="background: {cores_perfil[perfil_nome]}22; padding: 12px; border-radius: 8px; 
                    border-left: 4px solid {cores_perfil[perfil_nome]}; margin: 10px 0;">
            <b>{perfil.icone} {perfil.nome}</b><br>
            <small>{perfil.descricao}</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Orçamento
        st.markdown("### 💰 Orçamento")
        orcamento = st.number_input(
            "Valor para investir (R$):",
            min_value=1000.0,
            max_value=10000000.0,
            value=10000.0,
            step=10000.0,
            format="%.2f"
        )
        
        st.markdown("---")
        
        # Setores
        st.markdown("### 🏢 Setores")
        setores_selecionados = st.multiselect(
            "Filtrar por setores:",
            options=SETORES,
            default=[],
            help="Deixe vazio para usar ativos padrão"
        )
        
        # Toggle para usar todos os ativos (mais lento)
        usar_todos_ativos = st.checkbox(
            "🔓 Usar todos os ativos (96)",
            value=False,
            help="⚠️ Mais lento no Cloud. Use apenas se necessário."
        )
        
        st.markdown("---")
        
        # Quantidade de ativos
        st.markdown("### 📈 Quantidade de Ativos")
        n_ativos_max = st.slider(
            "Máximo de ativos na carteira:",
            min_value=3,
            max_value=30,
            value=10,
            help="Limita o número de ativos no portfólio"
        )
        
        st.markdown("---")
        
        # Botão de otimização
        otimizar = st.button("🚀 OTIMIZAR CARTEIRA", use_container_width=True)
        
        # Parâmetros Configuráveis
        st.markdown("### ⚙️ Parâmetros Avançados")
        
        periodo_anos = st.slider(
            "Período de análise (anos):",
            min_value=1,
            max_value=5,
            value=5,
            help="Quantidade de anos de dados históricos"
        )
        
        taxa_selic = st.number_input(
            "Taxa Selic (% a.a.):",
            min_value=0.0,
            max_value=30.0,
            value=15.0,
            step=0.25,
            format="%.2f",
            help="Taxa livre de risco para cálculo do Sharpe"
        ) / 100  # Converte para decimal
        
        peso_maximo = st.slider(
            "Peso máximo por ativo (%):",
            min_value=5,
            max_value=50,
            value=20,
            help="Limite máximo de alocação em um único ativo"
        ) / 100  # Converte para decimal
        
        # Info de ativos selecionados
        if setores_selecionados:
            n_disponiveis = len(get_tickers_by_setor(setores_selecionados))
        elif usar_todos_ativos:
            n_disponiveis = len(ATIVOS_B3)
        else:
            n_disponiveis = 75
        
        st.markdown(f"""
        <div style="background: #1E213022; padding: 10px; border-radius: 8px; margin-top: 10px;">
            <small>📊 <b>Ativos para análise:</b> {n_disponiveis}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # ============== MAIN CONTENT ==============
    
    # Obtém tickers filtrados
    if setores_selecionados:
        tickers = get_tickers_by_setor(setores_selecionados)
        st.info(f"🔍 Filtrando {len(tickers)} ativos dos setores: {', '.join(setores_selecionados)}")
    elif usar_todos_ativos:
        tickers = get_all_tickers()
        st.warning(f"⚠️ Usando todos os {len(tickers)} ativos. Pode demorar mais.")
    else:
        tickers = get_top75_tickers()
        st.success(f"🚀 Usando TOP 75 ativos (otimizado para Cloud)")
    
    if len(tickers) < 3:
        st.error("❌ Selecione setores com pelo menos 3 ativos disponíveis.")
        return
    
    # Carrega dados
    dados = carregar_dados_completos(tickers, anos=periodo_anos)
    
    if dados is None:
        st.error("❌ Erro ao carregar dados. Verifique sua conexão.")
        return
    
    # Info dos dados
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📅 Período Analisado", f"{dados['periodo_inicio']} a {dados['periodo_fim']}")
    with col2:
        st.metric("🎯 Ativos Válidos", f"{dados['n_ativos']} ativos")
    
    st.markdown("---")
    
    # ============== OTIMIZAÇÃO ==============
    # Chave única para detectar mudança de parâmetros
    params_key = f"{perfil_nome}_{taxa_selic:.4f}_{peso_maximo:.2f}_{n_ativos_max}_{dados['n_ativos']}"
    
    needs_optimization = (
        otimizar or 
        'resultado' not in st.session_state or
        st.session_state.get('params_key') != params_key
    )
    
    if needs_optimization:
        with st.spinner(f"🔄 Otimizando carteira ({perfil_nome})..."):
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
                n_pontos=20,  # Reduzido para performance no Cloud
                peso_maximo=peso_maximo
            )
            
            st.session_state['resultado'] = resultado
            st.session_state['fronteira'] = fronteira
            st.session_state['params_key'] = params_key
    else:
        resultado = st.session_state['resultado']
        fronteira = st.session_state['fronteira']
    
    if not resultado.sucesso:
        st.warning(f"⚠️ Otimização com aviso: {resultado.mensagem}")
    
    # ============== MÉTRICAS PRINCIPAIS ==============
    st.markdown("## 📊 Resultados da Otimização")
    
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
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{resultado.sharpe:.3f}</div>
            <div class="metric-label">Índice de Sharpe</div>
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
    
    # ============== GRÁFICOS ==============
    
    # Prepara dicionário de pesos
    pesos_dict = {t: p for t, p in zip(resultado.tickers, resultado.pesos) if p > 0.001}
    info_tickers = get_ticker_info()
    
    # Linha 1: Fronteira + Composição
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        fig_fronteira = grafico_fronteira_eficiente(
            fronteira,
            {'retorno': resultado.retorno_esperado, 'volatilidade': resultado.volatilidade},
            perfil_nome,
            taxa_selic
        )
        st.plotly_chart(fig_fronteira, use_container_width=True)
    
    with col2:
        fig_pizza = grafico_composicao_pizza(pesos_dict, orcamento)
        st.plotly_chart(fig_pizza, use_container_width=True)
    
    # Linha 2: Barras
    fig_barras = grafico_barras_alocacao(pesos_dict, orcamento, info_tickers)
    st.plotly_chart(fig_barras, use_container_width=True)
    
    st.markdown("---")
    
    # ============== TABELA DE ALOCAÇÃO ==============
    st.markdown("## 📋 Detalhamento da Carteira")
    
    # Monta DataFrame
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
    
    # ============== ANÁLISES ADICIONAIS ==============
    with st.expander("🔗 Ver Matriz de Correlação", expanded=False):
        # Filtra apenas ativos da carteira
        tickers_carteira = [t for t, p in pesos_dict.items() if p > 0.01]
        if len(tickers_carteira) > 1:
            matriz_filtrada = dados['matriz_cov'].loc[tickers_carteira, tickers_carteira]
            fig_corr = grafico_matriz_correlacao(matriz_filtrada)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with st.expander("📉 Ver Evolução Histórica dos Preços", expanded=False):
        tickers_carteira = [t for t, p in pesos_dict.items() if p > 0.02][:10]
        if tickers_carteira:
            fig_evolucao = grafico_evolucao_precos(dados['precos'], tickers_carteira)
            st.plotly_chart(fig_evolucao, use_container_width=True)
    
    with st.expander("📊 Ver Métricas Individuais dos Ativos", expanded=False):
        st.dataframe(dados['metricas'], use_container_width=True)
    
    # ============== BACKTESTING E ANÁLISE DE RISCO ==============
    st.markdown("---")
    st.markdown("## 📈 Backtesting e Análise de Risco")
    st.markdown("*Simulação histórica da carteira*")
    
    tipo_backtest = st.radio(
        "Metodologia de Backtesting:",
        options=["Walk-Forward (Out-of-Sample / Realista)", "Pesos Fixos (In-Sample / Teórico)"],
        horizontal=True,
        help="O Walk-Forward re-otimiza a carteira periodicamente s/ ver o futuro. 'Pesos Fixos' aplica os pesos de hoje no passado (look-ahead bias)."
    )
    
    with st.spinner('🔄 Executando backtesting... (Walk-Forward pode demorar um pouco mais)'):
        if "Walk-Forward" in tipo_backtest:
            # Walk-Forward OOS
            try:
                backtest = cached_backtest_oos(
                    _precos=dados['precos'],
                    perfil=perfil_nome,
                    orcamento=orcamento,
                    taxa_selic=taxa_selic,
                    n_ativos_max=n_ativos_max
                )
            except Exception as e:
                st.error(f"Erro no Walk-Forward: {e}. Usando fallback para Pesos Fixos.")
                backtest = cached_backtest_is(
                    _precos=dados['precos'],
                    pesos_dict=pesos_dict,
                    orcamento=orcamento,
                    taxa_selic=taxa_selic
                )
        else:
            # In Sample
            backtest = cached_backtest_is(
                _precos=dados['precos'],
                pesos_dict=pesos_dict,
                orcamento=orcamento,
                taxa_selic=taxa_selic
            )
        
        # Calcula métricas de risco da carteira otimizada
        metricas_risco = calcular_metricas_risco_portfolio(
            retornos=dados['retornos'],
            pesos=pesos_dict,
            taxa_livre_risco=taxa_selic
        )
        
        # Baixa dados do Ibovespa para comparação
        try:
            ibov = baixar_dados_historicos(['^BVSP'], anos=periodo_anos)
            if not ibov.empty:
                ibov_serie = ibov['^BVSP'] if '^BVSP' in ibov.columns else ibov.iloc[:, 0]
            else:
                ibov_serie = None
        except:
            ibov_serie = None
    
    # Métricas de Backtesting
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">R$ {backtest['capital_final']:,.0f}</div>
            <div class="metric-label">Capital Final</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cor_retorno = "#00D4AA" if backtest['retorno_total'] > 0 else "#FF5252"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="background: {cor_retorno}; -webkit-background-clip: text;">{backtest['retorno_total']*100:.1f}%</div>
            <div class="metric-label">Retorno Total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="background: #FF5252; -webkit-background-clip: text;">{backtest['max_drawdown']*100:.1f}%</div>
            <div class="metric-label">Drawdown Máximo</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{backtest['sortino']:.2f}</div>
            <div class="metric-label">Índice Sortino</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de Backtesting
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        fig_backtest = grafico_backtesting(
            serie_carteira=backtest['serie_carteira'],
            serie_benchmark=ibov_serie,
            nome_benchmark="Ibovespa"
        )
        st.plotly_chart(fig_backtest, use_container_width=True)
    
    with col2:
        fig_risco = grafico_metricas_risco(metricas_risco)
        st.plotly_chart(fig_risco, use_container_width=True)
    
    # Gráfico de Drawdown
    fig_dd = grafico_drawdown(backtest['drawdown_serie'])
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Interpretação das Métricas
    with st.expander("📖 Entenda as Métricas de Risco", expanded=False):
        st.markdown(f"""
        ### 📊 Value at Risk (VaR) - {metricas_risco.get('var_95_anual', 0)*100:.2f}% anual
        {metricas_risco.get('interpretacao_var', '')}
        
        ### ⚠️ Conditional VaR (CVaR) - {metricas_risco.get('cvar_95_anual', 0)*100:.2f}% anual
        {metricas_risco.get('interpretacao_cvar', '')}
        
        ### 📉 Drawdown Máximo - {backtest['max_drawdown']*100:.1f}%
        A maior queda do pico ao vale durante o período analisado.
        Durou aproximadamente **{backtest['duracao_max_drawdown']} dias**.
        
        ### 🎯 Índice de Sortino - {backtest['sortino']:.2f}
        Similar ao Sharpe, mas considera apenas volatilidade negativa (risco de perda).
        - **> 1**: Bom
        - **> 2**: Excelente
        - **< 0**: Retorno abaixo da taxa livre de risco
        
        ### 📈 Curtose - {metricas_risco.get('curtose', 0):.2f}
        Mede a probabilidade de eventos extremos (caudas pesadas).
        - **> 3**: Mais eventos extremos que uma distribuição normal
        - **< 3**: Menos eventos extremos
        """)
        
    # ============== LIMITAÇÕES ==============
    st.markdown("---")
    st.markdown("## 🔬 Limitações e Vieses do Estudo")
    st.info("""
    **Viés de Sobrevivência (Survivor Bias):** A lista de ativos analisados neste projeto é composta pelos componentes atuais 
    (ou recentes) da B3. Empresas que faliram, fecharam capital, ou entregaram péssimos resultados nas últimas décadas 
    podem ter saído das listas de ativos acompanhados. Como resultado, o backtesting foca numa seleção de empresas 
    que sabidamente sobreviveram até a data final, o que pode enviesar as simulações positivamente.
    """)
    
    # ============== FOOTER ==============
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><b>TCC - Otimização de Carteiras de Investimentos</b></p>
        <p>Teoria Moderna do Portfólio de Markowitz</p>
        <p><b>Gabriel Estrela Lopes</b> | Ciência de Dados para Negócios - UFPB</p>
</div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
