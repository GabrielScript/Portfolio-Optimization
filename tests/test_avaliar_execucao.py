"""Peças do scripts/avaliar_execucao.py (prova final com execução realista, S4)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.backtest import ResultadoBacktest, retornos_da_carteira, walk_forward
from core.config import ConfigBacktest
from core.execucao import Refeita
from scripts.avaliar_conservadora import CONSERVADORA
from scripts.avaliar_execucao import (a_partir_de, comprado_e_mantido, conferir_ancora, decidir_prova,
                                      liquidez_da_tabela, medidas, proventos_diarios)


def test_proventos_do_dia_ex_compostos_e_sem_restituicao_de_capital():
    dividendos = pd.DataFrame({
        "security": ["AAAA3", "AAAA3", "AAAA3", "BBBB3", "CCCC3"],
        "ex_date": ["2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04", "2024-01-03"],
        "corporate_action": ["DIVIDENDO", "JRS CAP PROPRIO", "REST CAP DIN", "RENDIMENTO", "DIVIDENDO"],
        "yield": [0.10, 0.20, 0.50, 0.01, 0.30]})
    datas = pd.bdate_range("2024-01-02", "2024-01-05")
    proventos = proventos_diarios(dividendos, datas, ["AAAA3", "BBBB3"])
    esperado = pd.DataFrame({"AAAA3": [0.0, 1.1 * 1.2 - 1, 0.0, 0.0], "BBBB3": [0.0, 0.0, 0.01, 0.0]}, index=datas)
    pd.testing.assert_frame_equal(proventos, esperado, check_freq=False)


def test_liquidez_da_tabela_vira_mes_por_papel():
    tabela = pd.DataFrame({"month": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 2, 1)],
                           "security": ["AAAA3", "BBBB3", "AAAA3"], "meio_spread": [0.001, 0.002, 0.003],
                           "vol_diaria": [0.02, 0.03, 0.04], "volume_mediano": [1e6, 2e6, 3e6]})
    liquidez = liquidez_da_tabela(tabela)
    meses = pd.DatetimeIndex(["2024-01-01", "2024-02-01"])
    pd.testing.assert_frame_equal(liquidez.meio_spread, pd.DataFrame({"AAAA3": [0.001, 0.003], "BBBB3": [0.002, None]},
                                                                     index=meses), check_index_type=False)
    assert liquidez.vol_diaria.loc["2024-02-01", "AAAA3"] == pytest.approx(0.04)
    assert liquidez.volume_mediano.loc["2024-01-01", "BBBB3"] == pytest.approx(2e6)


def test_a_partir_de_comeca_em_caixa_no_pregao_anterior_a_1a_decisao_do_recorte():
    datas = pd.bdate_range("2013-12-23", periods=10)
    alvos = pd.DataFrame({"A": [1.0, 0.5, 0.2]}, index=datas[[2, 5, 8]])
    curva = ResultadoBacktest("x", "teste", pd.Series(range(10), index=datas, dtype=float), alvos,
                              pd.Series([1.0, 0.5, 0.3], index=alvos.index), 0.0)
    recorte = a_partir_de(curva, pd.Timestamp("2013-12-28"))     # 1ª decisão desde então: 30/12
    assert list(recorte.pesos_alvo.index) == list(datas[[5, 8]])
    assert recorte.valores.index[0] == datas[4]
    assert recorte.valores.index[-1] == datas[-1]
    assert list(recorte.turnover.index) == list(datas[[5, 8]])


def test_a_prova_pede_p_de_pelo_menos_095_nas_tres_fases():
    assert decidir_prova([0.96, 0.95, 0.99])
    assert not decidir_prova([0.96, 0.949, 0.99])


def _curva_do_motor(custo_bps: float) -> tuple[ResultadoBacktest, pd.DataFrame]:
    datas = pd.bdate_range("2020-01-01", periods=150)
    gerador = np.random.default_rng(5)
    precos = pd.DataFrame(100 * np.cumprod(1 + gerador.normal(0.0005, 0.02, (150, 3)), axis=0), index=datas,
                          columns=["A", "B", "C"])
    retornos = retornos_da_carteira(precos)
    cfg = ConfigBacktest(janela_treino=60, janela_teste=21, capital_inicial=100.0, custo_bps=custo_bps)
    return walk_forward(retornos, lambda treino: 1 / (1 + treino.std()), cfg, "x"), retornos


def test_a_ancora_aceita_a_curva_do_motor_refeita_com_o_mesmo_custo_e_para_se_nao_bater():
    curva, retornos = _curva_do_motor(10.0)
    conferir_ancora(curva, retornos, 10.0)
    with pytest.raises(SystemExit):
        conferir_ancora(curva, retornos, 20.0)


def test_comprado_e_mantido_compra_o_ativo_na_1a_decisao_do_recorte():
    datas = pd.bdate_range("2014-01-01", periods=5)
    recorte = ResultadoBacktest("x", "teste", pd.Series(1.0, index=datas),
                                pd.DataFrame({"A": [1.0, 1.0]}, index=datas[[1, 3]]), pd.Series([1.0, 0.2]), 0.0)
    bova = comprado_e_mantido(recorte, "BOVA11")
    assert bova.valores.index.equals(datas)
    pd.testing.assert_frame_equal(bova.pesos_alvo, pd.DataFrame({"BOVA11": [1.0]}, index=datas[[1]]))


def _refeita(saida: pd.Series, valores: pd.Series) -> Refeita:
    return Refeita(valores, saida, pd.Series(dtype=float), pd.Series(0.0, index=valores.index[:1]))


def test_o_p_da_prova_compara_o_valor_de_saida_com_o_do_bova11():
    # patrimônios iguais; só a saída da estratégia cresce mais (menos IR a pagar ao vender)
    datas = pd.bdate_range("2020-01-01", periods=300)
    bova = pd.Series(10_000 * np.cumprod(1 + np.random.default_rng(1).normal(0.0004, 0.01, 300)), index=datas)
    melhor = bova * np.linspace(1.0, 1.2, 300)
    tabela = medidas({CONSERVADORA: _refeita(melhor, bova), "BOVA11": _refeita(bova, bova)})
    assert tabela.loc[CONSERVADORA, "P aritmético"] > 0.99
    assert tabela.loc[CONSERVADORA, "Saída final (R$)"] == pytest.approx(melhor.iloc[-1])
