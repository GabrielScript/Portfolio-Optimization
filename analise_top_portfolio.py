"""
analise_top_portfolio.py - Seleção em 2 estágios: top ações B3 + top startups (BR/US em BRL).

Uso: python analise_top_portfolio.py

AVISO METODOLÓGICO: selecionar os "top N" e otimizar só eles NA MESMA amostra
é data snooping. O retorno esperado impresso é in-sample e otimista por
construção. Para avaliar a regra de seleção, rode-a dentro do walk-forward
(core.backtest.walk_forward) com uma estratégia que seleciona e otimiza a
cada janela de treino.
"""

from __future__ import annotations

import logging

import pandas as pd

import assets
from core.config import ConfigOtimizacao, Objetivo, taxa_anual_para_aritmetica
from core.estimation import estimar_momentos, retornos_simples
from core.market_data import carregar_precos, obter_selic_meta
from core.optimization import ResultadoOtimizacao, otimizar

ANOS = 3
MIN_PREGOES_STARTUP = 250
CAPITAL = 1000.0


def otimizar_universo(precos: pd.DataFrame, cfg: ConfigOtimizacao, rotulo: str) -> ResultadoOtimizacao:
    retornos = retornos_simples(precos)
    momentos = estimar_momentos(retornos, shrink_mu=True)
    resultado = otimizar(momentos.mu, momentos.cov, cfg)
    print(f"\n{'=' * 70}\n{rotulo}")
    print(f"Amostra: {retornos.index[0]:%m/%Y} a {retornos.index[-1]:%m/%Y} ({len(retornos)} pregões, "
          f"{precos.shape[1]} ativos) | shrinkage Jorion {momentos.intensidade_shrinkage:.0%}")
    print(f"E[R] {resultado.retorno_esperado:.1%} | vol {resultado.volatilidade:.1%} | "
          f"Sharpe {resultado.sharpe:.2f} | {resultado.mensagem}")
    return resultado


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    rf = taxa_anual_para_aritmetica(obter_selic_meta().valor)
    info = {**assets.get_ticker_info(), **assets.get_startup_info()}
    startups = set(assets.get_startup_tickers())

    acoes = [t for t in assets.get_all_tickers() if t not in startups]
    precos_acoes = carregar_precos(acoes, ANOS).precos
    cfg_acoes = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=10,
                                 vol_maxima=0.40, taxa_livre_risco=rf)
    top_acoes = otimizar_universo(precos_acoes, cfg_acoes, "TOP AÇÕES B3 (Kelly, λ=2, vol <= 40%)")

    dados_startups = carregar_precos(assets.get_startup_tickers(), ANOS, cobertura_minima=0.0)
    precos_st = dados_startups.precos
    precos_st = precos_st.loc[:, precos_st.notna().sum() >= MIN_PREGOES_STARTUP]
    cfg_st = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.25, n_ativos_max=5,
                              vol_maxima=0.60, taxa_livre_risco=rf)
    top_st = otimizar_universo(precos_st, cfg_st, "TOP STARTUPS BR+US em BRL (Kelly, vol <= 60%)")

    selecionados = (top_acoes.pesos[top_acoes.pesos > 0].index.tolist()
                    + top_st.pesos[top_st.pesos > 0].index.tolist())
    conjunto = pd.concat([precos_acoes, precos_st], axis=1)[selecionados]
    cfg_final = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=None,
                                 vol_maxima=0.45, taxa_livre_risco=rf)
    final = otimizar_universo(conjunto, cfg_final, f"CARTEIRA CONJUNTA — R$ {CAPITAL:,.0f} (IN-SAMPLE)")

    ultimo_preco = conjunto.ffill().iloc[-1]
    print(f"\n  {'Ticker':<12} {'Nome':<28} {'Peso':>6} {'R$':>9} {'Preço R$':>10} {'Qtd':>8}")
    for ticker, peso in final.pesos[final.pesos > 0].sort_values(ascending=False).items():
        valor = peso * CAPITAL
        print(f"  {ticker:<12} {info.get(ticker, {}).get('nome', ticker):<28} {peso:6.1%} "
              f"{valor:9.2f} {ultimo_preco[ticker]:10.2f} {valor / ultimo_preco[ticker]:8.2f}")
    print("\n  In-sample: não use estes números como expectativa de retorno.")


if __name__ == "__main__":
    main()
