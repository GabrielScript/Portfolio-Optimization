"""
growth_score.py - Score de Growth/VC-lens para ações da B3
App: Otimizador Growth Agressivo B3
Autor: Gabriel Estrela Lopes

Score composto (0-100) por ativo, combinando:
1. Crescimento de receita (peso 30)      - fundamento central do growth investing
2. Qualidade do FCF (peso 20)            - margem FCF >= 15% (threshold do livro
                                           "The Intelligent Quality Investor") e
                                           flag de queima de caixa (lente VC)
3. Intensidade de P&D (peso 15)          - proxy de inovação/disrupção (lente VC);
                                           None é comum na B3 -> score neutro
4. Momentum + Linearidade (peso 20)      - R² da regressão do log-preço
                                           (linearidade >= 0.85 = compounders,
                                           cf. quality investor book) + retorno 12m
5. Proxy de TAM (peso 15)                - percentil de market cap dentro do setor:
                                           cap baixa em setor grande = mais espaço
                                           para crescer (lente VC aplicada a listadas)

O filtro por score é análogo ao "quintile portfolio" (Palomar, cap. 6.4.4),
que é formalmente um portfólio robusto a erros de estimação de mu.

Fundamentals via yfinance (já é dependência do projeto). Falha em qualquer
campo degrada para score neutro do componente — nunca derruba o pipeline.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bridge CVM: demonstrações auditadas 2011+ do repositório irmão ai-hedge-fund.
# api_cvm.py é autocontido (pandas+requests) e o cache local (~160MB) já está
# baixado em ai-hedge-fund/data/cvm_cache — habilita CAGR e linearidade de
# receita/FCF de 10 anos (método do "Intelligent Quality Investor").
# ---------------------------------------------------------------------------
_RAIZ_HEDGE_FUND = Path(__file__).resolve().parent.parent
_CVM_TOOLS = _RAIZ_HEDGE_FUND / "src" / "tools"
CVM_DISPONIVEL = False
try:
    if _CVM_TOOLS.exists():
        sys.path.insert(0, str(_CVM_TOOLS))
        os.environ.setdefault("CVM_CACHE_DIR", str(_RAIZ_HEDGE_FUND / "data" / "cvm_cache"))
        import api_cvm  # noqa: E402
        CVM_DISPONIVEL = True
except Exception as _e:  # pragma: no cover
    logger.warning(f"Bridge CVM indisponível ({_e}); growth score usa só Yahoo TTM.")

# Pesos dos componentes (somam 100)
PESO_RECEITA = 30.0
PESO_FCF = 20.0
PESO_PD = 15.0
PESO_MOMENTUM = 20.0
PESO_TAM = 15.0

# Thresholds (livro quality investor: 15% para CAGR/margem, R² 0.85)
THRESHOLD_CRESCIMENTO_FORTE = 0.20   # >20% a.a. = forte (growth_agent.py)
THRESHOLD_CRESCIMENTO_OK = 0.10      # 10-20% = moderado
THRESHOLD_FCF_MARGIN = 0.15          # margem FCF >= 15%
THRESHOLD_LINEARIDADE = 0.85         # R² >= 0.85 = compounder


def _cagr_e_linearidade(valores: List[float]) -> tuple:
    """
    CAGR e linearidade (R²) de uma série anual (mais antigo -> mais recente).

    Método do "Intelligent Quality Investor": CAGR mede o crescimento,
    R² da regressão do log-valor mede a CONSISTÊNCIA (compounders têm
    R² >= 0.85 — crescimento incremental, não pulo cíclico único).
    Retorna (None, None) se a série for curta ou cruzar zero (log indefinido).
    """
    v = [x for x in valores if x is not None and np.isfinite(x)]
    if len(v) < 4:
        return None, None
    n = len(v) - 1
    if v[0] <= 0 or v[-1] <= 0:
        return None, None
    cagr = (v[-1] / v[0]) ** (1.0 / n) - 1.0
    if any(x <= 0 for x in v):
        return cagr, None  # CAGR ponta-a-ponta ok, linearidade log indefinida
    y = np.log(v)
    x = np.arange(len(y), dtype=float)
    corr = np.corrcoef(x, y)[0, 1]
    r2 = corr ** 2 if np.isfinite(corr) else None
    return cagr, r2


def _historico_cvm(ticker: str) -> Dict:
    """
    Série anual de receita e FCF via demonstrações auditadas da CVM (2011+).
    Retorna listas do mais antigo para o mais recente. Nunca lança exceção.
    """
    out = {'receitas': None, 'fcfs': None}
    if not CVM_DISPONIVEL:
        return out
    try:
        frames = api_cvm.get_annual_frames(
            ticker.replace('.SA', ''), date.today().isoformat(), limit=10
        )
        if frames:
            frames = sorted(frames, key=lambda f: f.report_period)  # antigo -> recente
            out['receitas'] = [f.income.get('revenue') for f in frames]
            out['fcfs'] = [f.cashflow.get('free_cash_flow') for f in frames]
    except Exception as e:
        logger.warning(f"CVM indisponível para {ticker}: {e}")
    return out


def _score_crescimento_receita(revenue_growth: Optional[float],
                               receitas_cvm: Optional[List[float]] = None) -> float:
    """
    Score 0-1 do crescimento de receita.

    Com histórico CVM (10 anos): 60% CAGR longo + 40% linearidade R² —
    separa compounders de pulos cíclicos. Sem CVM: fallback TTM do Yahoo.
    """
    if receitas_cvm:
        cagr, r2 = _cagr_e_linearidade(receitas_cvm)
        if cagr is not None:
            # CAGR: >=15% (threshold do livro) = 1.0; escala linear até lá
            s_cagr = float(np.clip(cagr / 0.15, 0.0, 1.0)) if cagr > 0 else 0.1
            # Linearidade: R² >= 0.85 = 1.0
            s_lin = (1.0 if (r2 or 0) >= THRESHOLD_LINEARIDADE
                     else (r2 or 0) / THRESHOLD_LINEARIDADE * 0.8)
            base = 0.6 * s_cagr + 0.4 * s_lin
            # TTM recente modula levemente: aceleração/desaceleração atual
            if revenue_growth is not None and not np.isnan(revenue_growth):
                ajuste = 0.1 if revenue_growth >= THRESHOLD_CRESCIMENTO_OK else \
                         (-0.1 if revenue_growth < 0 else 0.0)
                base = float(np.clip(base + ajuste, 0.0, 1.0))
            return base

    # Fallback: só TTM Yahoo (1 período — ruidoso, por isso scores contidos)
    if revenue_growth is None or np.isnan(revenue_growth):
        return 0.4  # neutro-baixo: sem dado de crescimento é sinal fraco p/ growth
    if revenue_growth >= THRESHOLD_CRESCIMENTO_FORTE:
        return 1.0
    if revenue_growth >= THRESHOLD_CRESCIMENTO_OK:
        return 0.7
    if revenue_growth > 0:
        return 0.4
    return 0.1  # receita encolhendo


def _score_fcf(fcf: Optional[float], receita: Optional[float],
               fcf_historico: Optional[List[float]]) -> tuple:
    """
    Score 0-1 da qualidade do FCF + flag de queima de caixa.

    Margem FCF >= 15% = 1.0 (threshold do livro). Queima de caixa (FCF < 0
    em 2+ períodos anuais consecutivos) NÃO zera o score — na lente VC,
    burn com receita crescendo é aceitável — mas retorna flag para exibição.
    """
    burn_flag = False
    if fcf_historico and len(fcf_historico) >= 2:
        # dois períodos mais recentes negativos = burning cash
        recentes = [v for v in fcf_historico[:2] if v is not None]
        if len(recentes) == 2 and all(v < 0 for v in recentes):
            burn_flag = True

    if fcf is None or receita is None or receita <= 0:
        return 0.5, burn_flag  # neutro: sem dado

    margem_fcf = fcf / receita
    if margem_fcf >= THRESHOLD_FCF_MARGIN:
        score = 1.0
    elif margem_fcf >= 0.05:
        score = 0.7
    elif margem_fcf >= 0:
        score = 0.5
    else:
        score = 0.3  # FCF negativo: penaliza levemente (não zera — lente VC)
    return score, burn_flag


def _score_pd(rd: Optional[float], receita: Optional[float]) -> float:
    """
    Score 0-1 da intensidade de P&D (R&D / receita).
    Poucas empresas B3 reportam P&D destacado — None = neutro (0.5),
    nunca penaliza (recomendação da exploração do growth_agent.py).
    """
    if rd is None or receita is None or receita <= 0 or rd <= 0:
        return 0.5
    intensidade = rd / receita
    if intensidade >= 0.10:
        return 1.0
    if intensidade >= 0.05:
        return 0.8
    if intensidade >= 0.01:
        return 0.65
    return 0.5


def _score_momentum_linearidade(precos_ativo: pd.Series) -> float:
    """
    Score 0-1 combinando retorno 12m e linearidade (R²) do log-preço.

    Linearidade: R² da regressão linear do ln(preço) vs tempo. Compounders
    têm R² >= 0.85 (quality investor book) — crescimento incremental
    consistente, não um pulo cíclico único.
    """
    serie = precos_ativo.dropna()
    if len(serie) < 60:
        return 0.5

    # Retorno 12m (ou o disponível)
    janela = min(252, len(serie) - 1)
    ret_12m = serie.iloc[-1] / serie.iloc[-1 - janela] - 1

    # Linearidade do log-preço sobre todo o histórico disponível
    y = np.log(serie.values)
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    corr = np.corrcoef(x, y)[0, 1]
    r2 = corr ** 2 if not np.isnan(corr) else 0.0

    # Momentum: mapeia retorno 12m em [0,1] (cap em +60% / -40%)
    mom = np.clip((ret_12m + 0.40) / 1.00, 0.0, 1.0)

    # Linearidade só conta se a tendência for de ALTA (slope > 0);
    # queda perfeitamente linear tem R² alto mas é o oposto do desejado.
    lin = r2 if slope > 0 else 0.0
    lin_score = 1.0 if lin >= THRESHOLD_LINEARIDADE else lin / THRESHOLD_LINEARIDADE * 0.8

    return 0.5 * mom + 0.5 * lin_score


def _score_tam(market_cap: Optional[float], caps_do_setor: List[float]) -> float:
    """
    Proxy de TAM (lente VC): percentil do market cap dentro do próprio setor.
    Cap relativamente pequena = mais espaço para crescer no mercado endereçável
    (assumindo que o setor listado aproxima o TAM). Percentil baixo -> score alto.
    """
    caps_validos = [c for c in caps_do_setor if c is not None and c > 0]
    if market_cap is None or market_cap <= 0 or len(caps_validos) < 3:
        return 0.5
    percentil = sum(1 for c in caps_validos if c <= market_cap) / len(caps_validos)
    # inverte: menor cap no setor = maior score (1 - percentil), com piso 0.2
    # (mega caps ainda podem crescer, só têm menos espaço relativo)
    return max(0.2, 1.0 - percentil)


def _buscar_fundamentals(ticker: str) -> Dict:
    """
    Busca fundamentals de um ticker via yfinance. Nunca lança exceção —
    campos ausentes viram None e degradam para scores neutros.
    """
    out = {
        'ticker': ticker, 'revenue_growth': None, 'fcf': None,
        'receita': None, 'rd': None, 'market_cap': None,
        'fcf_historico': None,
    }
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        out['revenue_growth'] = info.get('revenueGrowth')
        out['fcf'] = info.get('freeCashflow')
        out['receita'] = info.get('totalRevenue')
        out['market_cap'] = info.get('marketCap')

        # Histórico anual de FCF para detectar queima de caixa
        try:
            cf = tk.cashflow
            if cf is not None and not cf.empty and 'Free Cash Flow' in cf.index:
                out['fcf_historico'] = cf.loc['Free Cash Flow'].dropna().tolist()
        except Exception:
            pass

        # P&D destacado (raro na B3)
        try:
            fin = tk.financials
            if fin is not None and not fin.empty and 'Research And Development' in fin.index:
                rd_serie = fin.loc['Research And Development'].dropna()
                if len(rd_serie) > 0:
                    out['rd'] = float(rd_serie.iloc[0])
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Fundamentals indisponíveis para {ticker}: {e}")
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def calcular_growth_scores(tickers: tuple, _precos: pd.DataFrame,
                           setores: Dict[str, str]) -> pd.DataFrame:
    """
    Calcula o Growth Score (0-100) para cada ticker do universo.

    Args:
        tickers: tupla de tickers (tupla p/ ser hashable no cache)
        _precos: DataFrame de preços (prefixo _ = ignorado pelo hash do cache;
                 o conteúdo muda junto com tickers/período, então o cache por
                 tickers+setores é suficiente na prática)
        setores: dict ticker -> setor (para o proxy de TAM)

    Returns:
        DataFrame indexado por ticker com colunas:
        score (0-100), s_receita, s_fcf, s_pd, s_momentum, s_tam,
        revenue_growth, burn_flag
    """
    tickers = list(tickers)

    # Fundamentals em paralelo (I/O-bound; 8 threads corta ~75 tickers p/ ~20s)
    fundamentals = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futuros = {pool.submit(_buscar_fundamentals, t): t for t in tickers}
        for fut in as_completed(futuros):
            f = fut.result()
            fundamentals[f['ticker']] = f

    # Agrupa market caps por setor para o proxy de TAM
    caps_por_setor: Dict[str, List[float]] = {}
    for t in tickers:
        setor = setores.get(t, 'Desconhecido')
        caps_por_setor.setdefault(setor, []).append(
            fundamentals.get(t, {}).get('market_cap')
        )

    linhas = []
    for t in tickers:
        f = fundamentals.get(t, {})
        setor = setores.get(t, 'Desconhecido')

        s_receita = _score_crescimento_receita(f.get('revenue_growth'))
        s_fcf, burn = _score_fcf(f.get('fcf'), f.get('receita'), f.get('fcf_historico'))
        s_pd = _score_pd(f.get('rd'), f.get('receita'))
        s_mom = (_score_momentum_linearidade(_precos[t])
                 if t in _precos.columns else 0.5)
        s_tam = _score_tam(f.get('market_cap'), caps_por_setor.get(setor, []))

        score = (PESO_RECEITA * s_receita + PESO_FCF * s_fcf + PESO_PD * s_pd +
                 PESO_MOMENTUM * s_mom + PESO_TAM * s_tam)

        linhas.append({
            'ticker': t,
            'score': round(score, 1),
            's_receita': round(s_receita, 2),
            's_fcf': round(s_fcf, 2),
            's_pd': round(s_pd, 2),
            's_momentum': round(s_mom, 2),
            's_tam': round(s_tam, 2),
            'revenue_growth': f.get('revenue_growth'),
            'burn_flag': burn,
        })

    df = pd.DataFrame(linhas).set_index('ticker')
    return df.sort_values('score', ascending=False)
