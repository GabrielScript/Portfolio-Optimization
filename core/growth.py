"""
growth.py - Growth Score (0-100) para ações da B3 ("lente VC" em empresas listadas).

Componentes (pesos somam 100):
1. Crescimento de receita (30): com CVM, 60% CAGR de 10 anos + 40% linearidade
   (R^2 do log-receita, critério do "Intelligent Quality Investor"); sem CVM,
   faixas do crescimento TTM do Yahoo.
2. Qualidade do FCF (20): margem FCF >= 15% = máximo; queima de caixa só gera flag.
3. Intensidade de P&D (15): rara na B3; ausência = neutro.
4. Momentum + linearidade do preço (20): retorno 12m + R^2 do log-preço.
5. Proxy de TAM (15): percentil inverso do market cap no setor.

LIMITAÇÃO METODOLÓGICA, a mais importante do app: o score usa fundamentos e
preços de HOJE. Filtrar o universo por ele e depois backtestar no mesmo
período é look-ahead. O componente de momentum seleciona literalmente os
vencedores do período testado. Por isso o walk-forward do app roda no
universo ANTES do filtro. O caminho point-in-time existe para a receita
(``api_cvm.get_annual_frames(ticker, data)`` respeita a data de entrega à
CVM), mas o Yahoo ``info`` não tem histórico.

Proxy de TAM: "cap pequena no setor = mais espaço para crescer" é hipótese do
autor, não resultado empírico estabelecido. Na literatura o efeito tamanho na
B3 é instável.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PESOS = {"receita": 30.0, "fcf": 20.0, "pd": 15.0, "momentum": 20.0, "tam": 15.0}
CRESCIMENTO_FORTE = 0.20
CRESCIMENTO_OK = 0.10
MARGEM_FCF_ALVO = 0.15
LINEARIDADE_ALVO = 0.85
CAGR_ALVO = 0.15
_MAX_THREADS_YAHOO = 4   # acima disso o Yahoo passa a devolver 429


def _carregar_api_cvm():
    """Importa o bridge CVM do repositório irmão (ai-hedge-fund/src/tools), se existir."""
    raiz = Path(__file__).resolve().parents[2]
    pasta = Path(os.environ.get("CVM_TOOLS_DIR", raiz / "src" / "tools"))
    if not (pasta / "api_cvm.py").exists():
        return None
    os.environ.setdefault("CVM_CACHE_DIR", str(raiz / "data" / "cvm_cache"))
    if str(pasta) not in sys.path:
        sys.path.insert(0, str(pasta))
    try:
        return importlib.import_module("api_cvm")
    except ImportError as erro:
        logger.warning("Bridge CVM indisponível: %s", erro)
        return None


def cagr_e_linearidade(valores: list[float]) -> tuple[float | None, float | None]:
    """CAGR ponta a ponta e R^2 do log-valor (série antiga -> recente, >= 4 pontos)."""
    serie = [v for v in valores if v is not None and np.isfinite(v)]
    if len(serie) < 4 or serie[0] <= 0 or serie[-1] <= 0:
        return None, None
    cagr = (serie[-1] / serie[0]) ** (1.0 / (len(serie) - 1)) - 1.0
    if any(v <= 0 for v in serie):
        return cagr, None
    correlacao = np.corrcoef(np.arange(len(serie)), np.log(serie))[0, 1]
    return cagr, (float(correlacao**2) if np.isfinite(correlacao) else None)


def _score_linearidade(r2: float | None) -> float:
    r2 = r2 or 0.0
    return 1.0 if r2 >= LINEARIDADE_ALVO else r2 / LINEARIDADE_ALVO * 0.8


def score_receita(crescimento_ttm: float | None, receitas_cvm: list[float] | None) -> float:
    if receitas_cvm:
        cagr, r2 = cagr_e_linearidade(receitas_cvm)
        if cagr is not None:
            base = 0.6 * (float(np.clip(cagr / CAGR_ALVO, 0, 1)) if cagr > 0 else 0.1)
            base += 0.4 * _score_linearidade(r2)
            if crescimento_ttm is not None and np.isfinite(crescimento_ttm):
                base += 0.1 if crescimento_ttm >= CRESCIMENTO_OK else (-0.1 if crescimento_ttm < 0 else 0.0)
            return float(np.clip(base, 0, 1))
    if crescimento_ttm is None or not np.isfinite(crescimento_ttm):
        return 0.4
    if crescimento_ttm >= CRESCIMENTO_FORTE:
        return 1.0
    if crescimento_ttm >= CRESCIMENTO_OK:
        return 0.7
    return 0.4 if crescimento_ttm > 0 else 0.1


def score_fcf(fcf: float | None, receita: float | None,
              fcf_recentes_primeiro: list[float] | None) -> tuple[float, bool]:
    """Score da margem FCF e flag de queima de caixa (2 anos mais recentes negativos)."""
    recentes = [v for v in (fcf_recentes_primeiro or [])[:2] if v is not None]
    queima = len(recentes) == 2 and all(v < 0 for v in recentes)
    if fcf is None or receita is None or receita <= 0:
        return 0.5, queima
    margem = fcf / receita
    if margem >= MARGEM_FCF_ALVO:
        return 1.0, queima
    if margem >= 0.05:
        return 0.7, queima
    return (0.5 if margem >= 0 else 0.3), queima


def score_pd(pesquisa: float | None, receita: float | None) -> float:
    if pesquisa is None or receita is None or receita <= 0 or pesquisa <= 0:
        return 0.5
    intensidade = pesquisa / receita
    if intensidade >= 0.10:
        return 1.0
    if intensidade >= 0.05:
        return 0.8
    return 0.65 if intensidade >= 0.01 else 0.5


def score_momentum(precos: pd.Series) -> float:
    serie = precos.dropna()
    if len(serie) < 60:
        return 0.5
    janela = min(252, len(serie) - 1)
    retorno_12m = serie.iloc[-1] / serie.iloc[-1 - janela] - 1.0
    log_precos = np.log(serie.values)
    tempo = np.arange(len(log_precos), dtype=float)
    inclinacao = np.polyfit(tempo, log_precos, 1)[0]
    correlacao = np.corrcoef(tempo, log_precos)[0, 1]
    r2 = correlacao**2 if np.isfinite(correlacao) and inclinacao > 0 else 0.0
    momentum = float(np.clip((retorno_12m + 0.40) / 1.00, 0.0, 1.0))
    return 0.5 * momentum + 0.5 * _score_linearidade(r2)


def score_tam(market_cap: float | None, caps_setor: list[float | None]) -> float:
    validos = [c for c in caps_setor if c is not None and c > 0]
    if market_cap is None or market_cap <= 0 or len(validos) < 3:
        return 0.5
    percentil = sum(1 for c in validos if c <= market_cap) / len(validos)
    return max(0.2, 1.0 - percentil)


def buscar_fundamentos_yahoo(ticker: str) -> dict:
    """Fundamentos TTM do Yahoo. Campos ausentes viram None (score neutro)."""
    saida = {"ticker": ticker, "revenue_growth": None, "fcf": None, "receita": None,
             "pd": None, "market_cap": None, "fcf_historico": None}
    try:
        ativo = yf.Ticker(ticker)
        info = ativo.info or {}
        saida.update(revenue_growth=info.get("revenueGrowth"), fcf=info.get("freeCashflow"),
                     receita=info.get("totalRevenue"), market_cap=info.get("marketCap"))
        fluxo = ativo.cashflow
        if fluxo is not None and not fluxo.empty and "Free Cash Flow" in fluxo.index:
            saida["fcf_historico"] = fluxo.loc["Free Cash Flow"].dropna().tolist()
        dre = ativo.financials
        if dre is not None and not dre.empty and "Research And Development" in dre.index:
            serie_pd = dre.loc["Research And Development"].dropna()
            saida["pd"] = float(serie_pd.iloc[0]) if len(serie_pd) else None
    except Exception as erro:  # noqa: BLE001 - fronteira com API externa instável
        # yfinance levanta tipos variados (HTTP 429/YFRateLimitError, JSON, KeyError).
        # Um ticker sem fundamentos degrada para score neutro, sem derrubar o universo.
        logger.warning("Fundamentos indisponíveis para %s: %s: %s", ticker, type(erro).__name__, erro)
    return saida


def receitas_cvm(api_cvm, ticker: str, data_referencia: date) -> list[float] | None:
    """Receitas anuais auditadas públicas até ``data_referencia`` (antiga -> recente)."""
    try:
        quadros = api_cvm.get_annual_frames(ticker.replace(".SA", ""), data_referencia.isoformat(), limit=10)
    except (OSError, ValueError, KeyError) as erro:
        logger.warning("CVM indisponível para %s: %s", ticker, erro)
        return None
    if not quadros:
        return None
    return [q.income.get("revenue") for q in sorted(quadros, key=lambda q: q.report_period)]


def calcular_growth_scores(tickers: list[str], precos: pd.DataFrame, setores: dict[str, str],
                           usar_cvm: bool = False, data_referencia: date | None = None) -> pd.DataFrame:
    """
    Score por ticker (índice) com colunas score, s_receita, s_fcf, s_pd,
    s_momentum, s_tam, revenue_growth e queima_caixa.
    Passe apenas ações: ETFs e índices não têm fundamentos.
    """
    referencia = data_referencia or date.today()
    api_cvm = _carregar_api_cvm() if usar_cvm else None
    with ThreadPoolExecutor(max_workers=_MAX_THREADS_YAHOO) as executor:
        fundamentos = {f["ticker"]: f for f in executor.map(buscar_fundamentos_yahoo, tickers)}

    caps_por_setor: dict[str, list] = {}
    for ticker in tickers:
        caps_por_setor.setdefault(setores.get(ticker, "?"), []).append(fundamentos[ticker]["market_cap"])

    linhas = []
    for ticker in tickers:
        f = fundamentos[ticker]
        historico = receitas_cvm(api_cvm, ticker, referencia) if api_cvm else None
        s_fcf, queima = score_fcf(f["fcf"], f["receita"], f["fcf_historico"])
        componentes = {
            "receita": score_receita(f["revenue_growth"], historico),
            "fcf": s_fcf,
            "pd": score_pd(f["pd"], f["receita"]),
            "momentum": score_momentum(precos[ticker]) if ticker in precos else 0.5,
            "tam": score_tam(f["market_cap"], caps_por_setor[setores.get(ticker, "?")]),
        }
        linhas.append({
            "ticker": ticker,
            "score": round(sum(PESOS[k] * v for k, v in componentes.items()), 1),
            **{f"s_{k}": round(v, 2) for k, v in componentes.items()},
            "revenue_growth": f["revenue_growth"],
            "queima_caixa": queima,
            "fonte_receita": "CVM 10a" if historico else "Yahoo TTM",
        })
    return pd.DataFrame(linhas).set_index("ticker").sort_values("score", ascending=False)


def filtrar_por_percentil(scores: pd.Series, percentil_minimo: float, minimo_ativos: int) -> list[str]:
    """Mantém tickers com score >= percentil; garante ``minimo_ativos`` pelos melhores scores."""
    validos = scores.dropna()
    if validos.empty:
        return []
    corte = np.percentile(validos, percentil_minimo)
    aprovados = validos[validos >= corte].index.tolist()
    if len(aprovados) < minimo_ativos:
        aprovados = validos.sort_values(ascending=False).head(minimo_ativos).index.tolist()
    return aprovados
