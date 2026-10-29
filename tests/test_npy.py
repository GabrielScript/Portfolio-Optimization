"""NPY (net payout yield) in the Conservadora: docs/superpowers/specs/2026-10-29-npy-conservadora-design.md."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.npy import contagens_por_mes, conservadora_npy, emissao_liquida, net_payout_yield
from core.risco import emissor
from core.sinais import conservadora

RAIZ = Path(__file__).resolve().parent.parent
MESES = pd.date_range("2020-01-01", periods=40, freq="MS")
MES = MESES[-1]


def _contagens(valores: dict[str, list[float]], fim: pd.Timestamp = MES) -> pd.DataFrame:
    """Month x issuer counts ending at ``fim`` (each list is the last len(list) months)."""
    linhas = max(len(v) for v in valores.values())
    indice = pd.date_range(end=fim, periods=linhas, freq="MS")
    return pd.DataFrame({k: [np.nan] * (linhas - len(v)) + v for k, v in valores.items()}, index=indice)


def test_contagens_por_mes_pivota_o_arquivo_de_valor_de_mercado():
    arquivo = pd.DataFrame({"issuer": ["AAAA", "AAAA", "BBBB"], "month": ["2020-01-01", "2020-02-01", "2020-01-01"],
                            "shares_today_units": [10.0, 11.0, 5.0], "market_cap": [1.0, 1.0, 1.0]})
    contagens = contagens_por_mes(arquivo)
    assert contagens.loc[pd.Timestamp("2020-02-01"), "AAAA"] == 11.0
    assert np.isnan(contagens.loc[pd.Timestamp("2020-02-01"), "BBBB"])


def test_emissao_e_a_ultima_contagem_sobre_a_media_dos_24_meses_menos_1():
    contagens = _contagens({"AAAA": [100.0] * 23 + [110.0], "BBBB": [100.0] * 24})
    emissao = emissao_liquida(contagens, MES)
    assert emissao["AAAA"] == pytest.approx(110.0 / ((23 * 100.0 + 110.0) / 24) - 1.0)
    assert emissao["BBBB"] == pytest.approx(0.0)


def test_emissao_nao_ve_meses_depois_do_sinal():
    base = _contagens({"AAAA": [100.0] * 24})
    depois = pd.concat([base, pd.DataFrame({"AAAA": [500.0]}, index=[MES + pd.DateOffset(months=1)])])
    assert emissao_liquida(depois, MES)["AAAA"] == pytest.approx(0.0)


def test_emissao_sem_24_contagens_positivas_vale_zero():
    contagens = _contagens({"CURT": [100.0] * 22 + [150.0], "ZERO": [100.0] * 22 + [0.0, 150.0],
                            "BURA": [100.0] * 10 + [np.nan] + [100.0] * 12 + [150.0], "CHEI": [100.0] * 23 + [150.0]})
    emissao = emissao_liquida(contagens, MES)
    assert emissao.reindex(["CURT", "ZERO", "BURA"]).fillna(0.0).tolist() == [0.0, 0.0, 0.0]
    assert emissao["CHEI"] > 0.4                                       # the same jump with the full history counts


def test_npy_e_o_dy_menos_a_emissao_do_emissor_nas_duas_classes():
    proventos = pd.DataFrame({"security": ["AAAA3", "AAAA4"], "ex_date": [MES - pd.DateOffset(months=2)] * 2,
                              "yield": [0.05, 0.07]})
    contagens = _contagens({"AAAA": [100.0] * 23 + [90.0]})          # a buyback: issuance < 0 raises the NPY
    recompra = emissao_liquida(contagens, MES)["AAAA"]
    npy = net_payout_yield(proventos, contagens, MES, ["AAAA3", "AAAA4", "SEMD3"])
    assert recompra < 0
    assert npy["AAAA3"] == pytest.approx(0.05 - recompra)
    assert npy["AAAA4"] == pytest.approx(0.07 - recompra)
    assert npy["SEMD3"] == pytest.approx(0.0)                          # no payout, no counts: 0, kept


def _painel(gerador: np.random.Generator, papeis: list[str]) -> pd.DataFrame:
    return pd.DataFrame(gerador.normal(0.01, 0.08, (len(MESES), len(papeis))), index=MESES, columns=papeis)


def test_sem_emissao_a_nota_e_identica_a_da_conservadora():
    gerador = np.random.default_rng(7)
    papeis = [f"P{i:03d}3" for i in range(30)]
    retornos = _painel(gerador, papeis)
    proventos = pd.DataFrame({"security": papeis[::2], "ex_date": [MES - pd.DateOffset(months=3)] * 15,
                              "yield": gerador.uniform(0.0, 0.1, 15)})
    contagens = _contagens({emissor(p): [1e6] * 24 for p in papeis})
    pd.testing.assert_series_equal(conservadora_npy(retornos, proventos, contagens, MES, papeis),
                                   conservadora(retornos, proventos, MES, papeis))


def test_a_emissao_rebaixa_o_papel_dentro_da_metade():
    gerador = np.random.default_rng(11)
    papeis = [f"P{i:03d}3" for i in range(20)]
    retornos = _painel(gerador, papeis)
    proventos = pd.DataFrame(columns=["security", "ex_date", "yield"])
    sem = conservadora_npy(retornos, proventos, _contagens({emissor(p): [1e6] * 24 for p in papeis}), MES, papeis)
    alvo = sem.idxmax()
    contagens = _contagens({emissor(p): [1e6] * 23 + [2e6 if p == alvo else 1e6] for p in papeis})
    com = conservadora_npy(retornos, proventos, contagens, MES, papeis)
    assert com[alvo] < sem[alvo]
    assert int(com[alvo]) == int(sem[alvo])                            # it stays in its volatility half


def test_emissor_do_ticker_casa_com_o_emissor_do_arquivo_de_contagens():
    caps = pd.read_parquet(RAIZ / "docs" / "dados" / "2026-10-29-cvm" / "market_caps.parquet",
                           columns=["issuer", "month"])
    emissores = set(caps["issuer"])
    for ticker in ("PETR3", "PETR4", "ITUB4", "VALE3", "BBDC3", "BBDC4", "WEGE3"):
        assert emissor(ticker) in emissores, ticker
