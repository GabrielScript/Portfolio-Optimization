"""Black-Litterman com views dos sinais do S2 (PLAN §5, S3)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from core.backtest import CarteiraRecusada
from core.config import ConfigOtimizacao, Objetivo
from core.views import (Sinal, escore_normal, estrategia_com_views, ic_ponto_no_tempo, mu_com_views,
                        nota_media_dos_escores, pesos_de_mercado, posterior, retorno_equilibrio, view_do_sinal)

PAPEIS = ["AAAA3", "BBBB3", "CCCC3"]
COV_DIAGONAL = pd.DataFrame(np.diag([0.09, 0.16, 0.25]), index=PAPEIS, columns=PAPEIS)     # σ = 30%, 40%, 50%
PI = pd.Series([0.05, 0.06, 0.08], index=PAPEIS)


def test_view_com_omega_do_ic_devolve_o_alfa_de_grinold_sobre_o_prior():
    # Q = Π + σ z e Ω = τ σ² (1 - IC)/IC: o peso da view é IC, e posterior - Π = IC σ z (Grinold), uma vez só
    z = pd.Series([1.0, -0.5, 2.0], index=PAPEIS)
    mu = posterior(PI, COV_DIAGONAL, [view_do_sinal(PI, COV_DIAGONAL, z, ic=0.1)])
    esperado = 0.1 * np.sqrt(np.diag(COV_DIAGONAL)) * z
    np.testing.assert_allclose((mu - PI).to_numpy(), esperado.to_numpy(), rtol=1e-12)


def test_escore_normal_e_o_quantil_do_posto_e_deixa_vazio_quem_nao_tem_nota():
    nota = pd.Series([3.0, np.nan, 1.0, 2.0, 10.0], index=["A", "B", "C", "D", "E"])
    escore = escore_normal(nota)
    assert escore.drop("B").tolist() == pytest.approx(list(norm.ppf([0.625, 0.125, 0.375, 0.875])))
    assert np.isnan(escore["B"])


def test_retorno_de_equilibrio_e_delta_vezes_sigma_vezes_os_pesos_de_mercado():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    pi = retorno_equilibrio(cov, pd.Series({"B": 0.25, "A": 0.75}), delta=2.0)
    assert pi.to_dict() == pytest.approx({"A": 2 * (0.04 * 0.75 + 0.01 * 0.25), "B": 2 * (0.01 * 0.75 + 0.09 * 0.25)})


def test_ic_da_decisao_e_a_media_dos_meses_antes_dela_e_exige_36_meses_positivos():
    # o IC de m-1 (retorno de m-1, fechado na decisão de m) entra; o do próprio m, não
    meses = pd.date_range("2020-01-01", "2023-02-01", freq="MS")         # 37 meses antes de 02/2023, e ele
    ics = pd.Series(0.04, index=meses)
    ics[pd.Timestamp("2023-01-01")], ics[pd.Timestamp("2023-02-01")] = 0.4, 0.9
    assert ic_ponto_no_tempo(ics, pd.Timestamp("2023-02-01")) == pytest.approx((36 * 0.04 + 0.4) / 37)
    assert ic_ponto_no_tempo(ics.iloc[2:], pd.Timestamp("2023-02-01")) is None          # 35 meses
    assert ic_ponto_no_tempo(-ics, pd.Timestamp("2023-02-01")) is None                  # IC médio negativo


def test_pesos_de_mercado_somam_um_nos_papeis_com_valor_de_mercado_e_dao_zero_ao_resto():
    valor_mercado = pd.Series({"AAAA": 300.0, "BBBB": 100.0, "ZZZZ": 600.0})       # ZZZZ fora do treino
    pesos = pesos_de_mercado(valor_mercado, ["AAAA3", "BBBB4", "SEMV3"])
    assert pesos.to_dict() == pytest.approx({"AAAA3": 0.75, "BBBB4": 0.25, "SEMV3": 0.0})


COV_CHEIA = pd.DataFrame([[0.09, 0.02, 0.01], [0.02, 0.16, 0.03], [0.01, 0.03, 0.25]], index=PAPEIS, columns=PAPEIS)
MES = pd.Timestamp("2023-02-01")
VALOR_MERCADO = pd.Series({"AAAA": 500.0, "BBBB": 300.0, "CCCC": 200.0})


def _sinal(meses_de_ic: int, ic: float, nota: list[float]) -> Sinal:
    ics = pd.Series(ic, index=pd.date_range(end=MES - pd.DateOffset(months=1), periods=meses_de_ic, freq="MS"))
    return Sinal(lambda mes, candidatos: pd.Series(nota, index=PAPEIS).reindex(list(candidatos)), ics)


def test_mu_com_views_e_o_posterior_dos_sinais_com_ic_e_o_equilibrio_sem_eles():
    # o 1º sinal tem 40 meses de IC 0,05; o 2º só 30 meses: fica fora
    com_ic, sem_historico = _sinal(40, 0.05, [3.0, 1.0, 2.0]), _sinal(30, 0.20, [1.0, 2.0, 3.0])
    mu = mu_com_views(COV_CHEIA, PAPEIS, MES, VALOR_MERCADO, [com_ic, sem_historico], delta=2.0)
    pi = retorno_equilibrio(COV_CHEIA, pesos_de_mercado(VALOR_MERCADO, PAPEIS), 2.0)
    view = view_do_sinal(pi, COV_CHEIA, escore_normal(pd.Series([3.0, 1.0, 2.0], index=PAPEIS)), 0.05)
    pd.testing.assert_series_equal(mu, posterior(pi, COV_CHEIA, [view]))
    pd.testing.assert_series_equal(mu_com_views(COV_CHEIA, PAPEIS, MES, VALOR_MERCADO, [sem_historico], 2.0), pi)


def test_view_de_valor_e_a_media_dos_escores_dos_sinais_com_os_que_houver():
    # Fórmula Mágica e Trending Value são correlacionados: uma view só, com a média dos dois escores normais
    def primeira(mes, candidatos):
        return pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}).reindex(list(candidatos)).dropna()

    def segunda(mes, candidatos):
        return pd.Series({"B": 30.0, "C": 10.0, "D": 20.0}).reindex(list(candidatos)).dropna()

    nota = nota_media_dos_escores(primeira, segunda)(MES, ["A", "B", "C", "D", "E"])
    z1, z2 = norm.ppf([1 / 6, 3 / 6, 5 / 6]), norm.ppf([5 / 6, 1 / 6, 3 / 6])
    assert nota.to_dict() == pytest.approx({"A": z1[0], "B": (z1[1] + z2[0]) / 2, "C": (z1[2] + z2[1]) / 2,
                                            "D": z2[2]})


MOTOR = [letra * 4 + "3" for letra in "ABCDEFGH"]
SETORES_DO_MOTOR = {papel: f"Setor {i}" for i, papel in enumerate(MOTOR)}
KELLY = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.20, n_ativos_max=10, vol_maxima=0.40,
                         aversao_risco=2.0)
VALOR_DO_MOTOR = pd.DataFrame({"issuer": [papel[:4] for papel in MOTOR], "month": pd.Timestamp("2025-05-01"),
                               "market_cap": 100.0})


def _treino(desvio: float) -> pd.DataFrame:
    datas = pd.bdate_range(end="2025-05-14", periods=300)
    return pd.DataFrame(np.random.default_rng(5).normal(0.0, desvio, (len(datas), 8)), index=datas, columns=MOTOR)


def test_estrategia_com_views_pede_a_nota_do_mes_da_decisao_e_puxa_o_peso_para_o_sinal():
    # nota crescente de A a H com IC de 0,3: H ganha peso e A perde, contra a mesma estratégia sem views
    chamadas = []

    def nota_do_mes(mes, candidatos):
        chamadas.append((mes, list(candidatos)))
        return pd.Series(np.arange(len(candidatos), dtype=float), index=list(candidatos))

    ics = pd.Series(0.3, index=pd.date_range(end="2025-04-01", periods=40, freq="MS"))
    sinais = [Sinal(nota_do_mes, ics)]
    com_views = estrategia_com_views(KELLY, SETORES_DO_MOTOR, sinais, VALOR_DO_MOTOR)(_treino(0.01))
    sem_views = estrategia_com_views(KELLY, SETORES_DO_MOTOR, [], VALOR_DO_MOTOR)(_treino(0.01))
    assert chamadas == [(pd.Timestamp("2025-05-01"), MOTOR)]
    assert com_views.get("HHHH3", 0.0) > sem_views.get("HHHH3", 0.0)
    assert com_views.get("AAAA3", 0.0) < sem_views.get("AAAA3", 0.0)


def test_carteira_com_views_e_vol_acima_do_teto_e_recusada_pelas_regras():
    # 8% ao dia em cada papel: qualquer carteira de 8 passa dos 40% ao ano; o otimizador cai no fallback e a regra recusa
    with pytest.raises(CarteiraRecusada):
        estrategia_com_views(KELLY, SETORES_DO_MOTOR, [], VALOR_DO_MOTOR)(_treino(0.08))
