"""Robustez das métricas fora da amostra: MinTRL, P[> referência] em log-retorno e janelas móveis."""

import math

import numpy as np
import pandas as pd
import pytest

from core.metrics import MomentosSharpe
from core.robustez import anos_minimos_para_psr, janelas_moveis, psr_da_diferenca


def test_anos_minimos_para_psr_conferido_a_mao():
    """SR diário 0,05, normal (assimetria 0, curtose 3): var = 1 + 2/4 · 0,05² = 1,00125;
    T* = 1 + 1,00125 · (1,644854 / 0,05)² = 1.084,57 pregões = 4,3038 anos de 252."""
    m = MomentosSharpe(sharpe=0.05, n_obs=500, assimetria=0.0, curtose=3.0)
    assert anos_minimos_para_psr(m) == pytest.approx(4.3038, abs=1e-4)


@pytest.mark.parametrize("sharpe", [-0.02, 0.0])
def test_sharpe_que_nao_passa_da_referencia_nunca_chega_ao_psr(sharpe):
    # o quadrado esconde o sinal: SR < SR* daria um número finito, como se bastasse esperar
    m = MomentosSharpe(sharpe=sharpe, n_obs=500, assimetria=0.0, curtose=3.0)
    assert anos_minimos_para_psr(m) == math.inf


def _curva(retornos: list[float]) -> pd.Series:
    return pd.Series(np.cumprod([1.0, *(1.0 + np.array(retornos))]), index=pd.bdate_range("2020-01-01", periods=len(retornos) + 1))


def test_p_em_log_retorno_ve_a_riqueza_que_a_media_aritmetica_esconde():
    """+10% e −9,5% alternados: média aritmética +0,25%/dia, log-retorno médio −0,23%/dia (a riqueza encolhe).
    Contra 0,1%/dia fixo, a diferença aritmética é positiva e a de log é negativa."""
    volatil, fixa = _curva([0.10, -0.095] * 100), _curva([0.001] * 200)
    assert psr_da_diferenca(volatil, fixa) > 0.5
    assert psr_da_diferenca(volatil, fixa, log=True) < 0.5


def _crescendo(taxa_diaria: float, datas: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((1.0 + taxa_diaria) ** np.arange(len(datas)), index=datas)


def test_janelas_moveis_comecam_em_cada_fim_de_mes_e_medem_contra_a_referencia():
    """36 fins de mês (jan/2020 a dez/2022) e janelas de 12 meses: 24 janelas, de 31/01/2020 a 29/01/2021 até
    31/12/2021 a 30/12/2022. Crescimento fixo: CAGR = (1 + g)^252 − 1 em qualquer janela."""
    datas = pd.bdate_range("2020-01-01", "2022-12-30")
    curvas = pd.DataFrame({"A": _crescendo(0.001, datas), "BOVA11": _crescendo(0.0005, datas)})
    janelas = janelas_moveis(curvas, "BOVA11", meses=12)
    assert janelas["curva"].unique().tolist() == ["A"]
    assert len(janelas) == 24
    assert (janelas["inicio"].iloc[0], janelas["fim"].iloc[0]) == (pd.Timestamp("2020-01-31"), pd.Timestamp("2021-01-29"))
    assert (janelas["inicio"].iloc[-1], janelas["fim"].iloc[-1]) == (pd.Timestamp("2021-12-31"), pd.Timestamp("2022-12-30"))
    assert janelas["excesso_cagr"].to_numpy() == pytest.approx(1.001**252 - 1.0005**252, abs=1e-9)


def test_curva_que_ainda_nao_existia_fica_fora_das_janelas_em_que_falta():
    """IVVB11 só desde 04/2014: a janela que começa antes não entra (nem com CAGR NaN, nem com a curva cortada)."""
    datas = pd.bdate_range("2020-01-01", "2022-12-30")
    nova = _crescendo(0.001, datas).where(datas >= "2021-06-01")
    curvas = pd.DataFrame({"IVVB11": nova, "BOVA11": _crescendo(0.0005, datas)})
    janelas = janelas_moveis(curvas, "BOVA11", meses=12)
    assert janelas["inicio"].min() == pd.Timestamp("2021-06-30")
    assert janelas["cagr"].notna().all()
