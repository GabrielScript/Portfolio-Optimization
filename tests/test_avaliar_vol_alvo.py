"""Regras do pré-registro do alvo de vol com EGARCH (``scripts/avaliar_vol_alvo.py``), sem dados do lago."""

import pandas as pd
import pytest

from core.backtest import ResultadoBacktest
from core.robustez import psr_da_diferenca
from core.vol_alvo import CurvaComAlvo
from scripts.avaliar_vol_alvo import alvo_passa, comparar


def _comparacao(p: float = 0.97, mdd_alvo: float = -0.30, giro_alvo: float = 2.0) -> dict[str, float]:
    return {"P[alvo > original]": p, "MDD com alvo": mdd_alvo, "MDD original": -0.25, "Giro com alvo": giro_alvo,
            "Giro original": 1.0}


def test_alvo_passa_so_com_p_mdd_e_giro_nas_tres_fases():
    # o molde da R10: na fronteira passa (P = 0,95, MDD 5 p.p. pior, giro no dobro); qualquer fase fora reprova
    assert alvo_passa(pd.DataFrame([_comparacao(p=0.95)] * 3))
    for ruim in (_comparacao(p=0.94), _comparacao(mdd_alvo=-0.31), _comparacao(giro_alvo=2.01)):
        assert not alvo_passa(pd.DataFrame([_comparacao()] * 2 + [ruim]))


DATAS = pd.bdate_range("2024-01-01", periods=10)


def test_comparacao_mede_a_original_so_no_trecho_do_alvo_e_soma_o_giro_da_exposicao():
    # a original cai 50% no pregão 1, antes do alvo começar (pregão 3): o MDD dela no trecho é 0. No trecho, ela
    # rebalanceia no pregão 6 (giro 1,0) com a exposição de 0,5 decidida no 3 (a do fechamento do 6 vale depois):
    # giro com alvo = (0,5 x 1,0 + |Δe| de 0,5 e 0,3) por ano
    original = ResultadoBacktest("original", "walk-forward OOS",
                                 pd.Series([100.0, 50, 60, 66, 70, 75, 80, 86, 90, 95], index=DATAS),
                                 pesos_alvo=pd.DataFrame({"A": 1.0}, index=DATAS[[2, 6]]),
                                 turnover=pd.Series([0.5, 1.0], index=DATAS[[2, 6]]), custos_totais=0.0)
    alvo = CurvaComAlvo(pd.Series([66.0, 67, 69, 71, 73, 74, 76], index=DATAS[3:]),
                        exposicoes=pd.Series([0.5, 0.8], index=DATAS[[3, 6]]),
                        giro=pd.Series([0.5, 0.3], index=DATAS[[3, 6]]))
    linha = comparar(original, alvo)
    anos = 6 / 252
    assert linha["P[alvo > original]"] == pytest.approx(psr_da_diferenca(alvo.valores, original.valores.loc[DATAS[3]:]))
    assert (linha["MDD original"], linha["MDD com alvo"]) == (0.0, 0.0)
    assert (linha["Giro original"], linha["Giro com alvo"]) == (pytest.approx(1.0 / anos),
                                                                pytest.approx((0.5 * 1.0 + 0.8) / anos))
    assert linha["Exposição média"] == pytest.approx(0.65)
