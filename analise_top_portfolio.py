"""
analise_top_portfolio.py - Análise Max Return: Top 10 empresas B3 + Top 5 startups
TCC: Otimização de Carteiras de Investimentos
Autor: Gabriel Estrela Lopes

Pipeline:
1. Baixa 3 anos de histórico de todas as ações da B3 (universo assets.py)
2. Baixa startups BR (B3) e US (Nasdaq/NYSE), converte US para BRL
3. Otimiza Max Return (perfil Agressivo) em cada universo separadamente
4. Otimização conjunta dos 15 selecionados -> alocação de R$ 1.000

Uso: python analise_top_portfolio.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import assets
from data_loader import calcular_retornos, calcular_estatisticas
from optimizer import otimizar_max_retorno

ANOS = 3
MIN_OBS_STARTUP = 250          # >= 1 ano de pregões para startup entrar
CAPITAL = 1000.0


def baixar(tickers, anos=ANOS, thresh_frac=0.8):
    """Download simples (sem cache streamlit), ffill sem bfill."""
    fim = datetime.now()
    inicio = fim - timedelta(days=anos * 365)
    dados = yf.download(
        tickers, start=inicio.strftime('%Y-%m-%d'), end=fim.strftime('%Y-%m-%d'),
        progress=False, auto_adjust=True
    )
    precos = dados['Close'] if len(tickers) > 1 else dados[['Close']]
    if len(tickers) == 1:
        precos.columns = tickers
    precos = precos.dropna(axis=1, thresh=int(len(precos) * thresh_frac))
    return precos.ffill()


def cambio_usdbrl(anos=ANOS):
    fim = datetime.now()
    inicio = fim - timedelta(days=anos * 365 + 30)
    fx = yf.download("USDBRL=X", start=inicio.strftime('%Y-%m-%d'),
                     end=fim.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
    s = fx['Close']
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.ffill()


def otimizar_universo(precos, n_max, vol_max, peso_max, rotulo):
    retornos = calcular_retornos(precos)
    mu, cov = calcular_estatisticas(retornos)
    res = otimizar_max_retorno(mu, cov, peso_maximo=peso_max,
                               vol_maxima=vol_max, n_ativos_max=n_max)
    pesos = pd.Series(res.pesos, index=res.tickers)
    financiados = pesos[pesos > 1e-4]
    # Ranking top-N: financiados primeiro (por peso), completa com maior mu
    vol_ind = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
    resto = mu.drop(financiados.index).sort_values(ascending=False)
    ranking = list(financiados.sort_values(ascending=False).index) + \
              list(resto.head(n_max - len(financiados)).index)
    top = pd.Series([pesos[t] for t in ranking], index=ranking)
    top.attrs['mu'] = mu
    top.attrs['vol'] = vol_ind
    print(f"\n{'='*70}\n{rotulo}")
    print(f"Janela: {retornos.index[0]:%m/%Y} a {retornos.index[-1]:%m/%Y} "
          f"({len(retornos)} pregões, {len(precos.columns)} ativos no universo)")
    print(f"Retorno esperado: {res.retorno_esperado*100:.1f}% a.a. | "
          f"Vol: {res.volatilidade*100:.1f}% a.a. | Sharpe: {res.sharpe:.2f}")
    return top, retornos


def main():
    info_b3 = assets.get_ticker_info()
    info_st = assets.get_startup_info()
    startups_br = {s['ticker'] for s in assets.STARTUPS_BR}

    # ---------- 1. Empresas B3 (universo completo) ----------
    print("Baixando universo B3 completo (3 anos)...")
    precos_b3 = baixar(assets.get_all_tickers())
    # Remove da lista de "empresas" as que também são startups BR
    precos_emp = precos_b3[[c for c in precos_b3.columns if c not in startups_br]]

    top10, ret_emp = otimizar_universo(
        precos_emp, n_max=10, vol_max=0.40, peso_max=0.20,
        rotulo="TOP 10 EMPRESAS B3 — Max Return (vol <= 40%)")
    mu_e, vol_e = top10.attrs['mu'], top10.attrs['vol']
    for t, w in top10.items():
        nome = info_b3.get(t, {}).get('nome', t)
        setor = info_b3.get(t, {}).get('setor', '?')
        print(f"  {t:<12} {nome:<24} {setor:<28} peso {w*100:5.1f}% | "
              f"ret {mu_e[t]*100:6.1f}% | vol {vol_e[t]*100:5.1f}%")

    # ---------- 2. Startups BR + US (US convertido para BRL) ----------
    print("\nBaixando startups BR + US (3 anos, mínimo 1 ano de história)...")
    tickers_st = assets.get_startup_tickers()
    precos_st = baixar(tickers_st, thresh_frac=0.0)  # aceita jovens
    obs_validas = precos_st.notna().sum()
    precos_st = precos_st[obs_validas[obs_validas >= MIN_OBS_STARTUP].index]

    fx = cambio_usdbrl()
    tickers_usd = [t for t in precos_st.columns if not t.endswith('.SA')]
    fx_alinhado = fx.reindex(precos_st.index).ffill().bfill()
    precos_st[tickers_usd] = precos_st[tickers_usd].mul(fx_alinhado, axis=0)

    # Janela comum: começa no IPO mais recente incluído (sem bfill!)
    precos_st = precos_st.dropna()

    top5, ret_st = otimizar_universo(
        precos_st, n_max=5, vol_max=0.60, peso_max=0.25,
        rotulo="TOP 5 STARTUPS (BR + US, em BRL) — Max Return (vol <= 60%)")
    mu_s, vol_s = top5.attrs['mu'], top5.attrs['vol']
    for t, w in top5.items():
        i = info_st.get(t, {})
        print(f"  {t:<12} {i.get('nome', t):<24} {i.get('mercado','?'):<4} "
              f"{i.get('setor','?'):<28} peso {w*100:5.1f}% | "
              f"ret {mu_s[t]*100:6.1f}% | vol {vol_s[t]*100:5.1f}%")

    # ---------- 3. Portfólio conjunto: 15 selecionados, R$ 1.000 ----------
    selecionados = list(top10.index) + list(top5.index)
    precos_joint = pd.concat(
        [precos_emp[list(top10.index)], precos_st[list(top5.index)]],
        axis=1).dropna()

    ret_joint = calcular_retornos(precos_joint)
    mu_j, cov_j = calcular_estatisticas(ret_joint)
    res_j = otimizar_max_retorno(mu_j, cov_j, peso_maximo=0.20, vol_maxima=0.45)
    pesos_j = pd.Series(res_j.pesos, index=res_j.tickers)
    pesos_j = pesos_j[pesos_j > 1e-3].sort_values(ascending=False)

    print(f"\n{'='*70}\nPORTFÓLIO FINAL CONJUNTO — R$ {CAPITAL:,.0f} (Max Return, vol <= 45%)")
    print(f"Janela conjunta: {ret_joint.index[0]:%m/%Y} a {ret_joint.index[-1]:%m/%Y} "
          f"({len(ret_joint)} pregões)")
    print(f"Retorno esperado: {res_j.retorno_esperado*100:.1f}% a.a. | "
          f"Vol: {res_j.volatilidade*100:.1f}% a.a. | Sharpe: {res_j.sharpe:.2f}\n")

    ultimo_preco = precos_joint.iloc[-1]
    print(f"  {'Ticker':<12} {'Nome':<26} {'Peso':>6} {'R$':>9} "
          f"{'Preço (R$)':>11} {'Qtd aprox':>10}")
    for t, w in pesos_j.items():
        nome = (info_b3.get(t) or info_st.get(t) or {}).get('nome', t)
        valor = w * CAPITAL
        preco = float(ultimo_preco[t])
        qtd = valor / preco
        print(f"  {t:<12} {nome:<26} {w*100:5.1f}% {valor:9.2f} "
              f"{preco:11.2f} {qtd:10.2f}")
    print(f"\n  Total alocado: R$ {pesos_j.sum()*CAPITAL:,.2f}")
    print("  Obs: B3 permite mercado fracionário (1 ação); ativos US via "
          "BDR/corretora com fração.")


if __name__ == "__main__":
    main()
