"""Execução realista de uma curva fora da amostra (S4): custo por liquidez e IR sobre a sequência de carteiras."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import ResultadoBacktest, retornos_da_carteira, walk_forward_dinamico
from core.config import ConfigBacktest
from core.execucao import SEM_CUSTOS, Custos, Liquidez, custo_da_ordem, refazer

PAPEIS = ["AAAA3", "BBBB3", "CCCC3", "DDDD3"]


def _motor() -> tuple[ResultadoBacktest, pd.DataFrame]:
    """Uma rodada real do motor: 4 papéis, pesos que mudam a cada rebalanceio, treino de 60 e teste de 21."""
    datas = pd.bdate_range("2020-01-01", periods=200)
    gerador = np.random.default_rng(9)
    precos = pd.DataFrame(100 * np.cumprod(1 + gerador.normal(0.0005, 0.02, (200, 4)), axis=0), index=datas,
                          columns=PAPEIS)
    universo = pd.DataFrame([(mes.date(), papel) for mes in pd.date_range("2020-01-01", "2020-10-01", freq="MS")
                             for papel in PAPEIS], columns=["month", "security"])

    def estrategia(treino: pd.DataFrame) -> pd.Series:
        media = treino.mean().clip(lower=1e-4)
        return media / media.sum()

    cfg = ConfigBacktest(janela_treino=60, janela_teste=21, capital_inicial=100.0, custo_bps=0.0)
    return walk_forward_dinamico(precos, universo, estrategia, cfg, "motor"), retornos_da_carteira(precos)


def test_sem_custo_e_sem_ir_a_curva_refeita_e_a_do_motor():
    # âncora: a mesma sequência de carteiras, comprada no fechamento anterior ao 1º pregão de cada uma, dá a mesma curva
    resultado, retornos = _motor()
    refeita = refazer(resultado, retornos, capital=100.0, custos=SEM_CUSTOS, com_ir=False)
    assert len(resultado.pesos_alvo) == 7
    pd.testing.assert_series_equal(refeita.valores, resultado.valores, check_names=False, rtol=1e-10)


def test_custo_da_ordem_e_emolumentos_meio_spread_e_impacto_pela_raiz_do_tamanho():
    # R$ 100 mil num papel de R$ 10 milhões por dia, vol diária de 2%: impacto 2% x raiz(1%) = 0,2%
    custo = custo_da_ordem(100_000.0, meio_spread=0.001, vol_diaria=0.02, volume_mediano=10_000_000.0,
                           custos=Custos(emolumentos=0.0003, impacto=1.0))
    assert custo == pytest.approx(100_000.0 * (0.0003 + 0.001 + 0.002))


def _liquidez_constante(meio_spread_do_mes: dict[str, float] | None = None) -> Liquidez:
    meses = pd.date_range("2019-12-01", "2020-12-01", freq="MS")
    spread = pd.DataFrame(0.001, index=meses, columns=PAPEIS)
    for mes, valor in (meio_spread_do_mes or {}).items():
        spread.loc[pd.Timestamp(mes)] = valor
    return Liquidez(meio_spread=spread, vol_diaria=pd.DataFrame(0.02, index=meses, columns=PAPEIS),
                    volume_mediano=pd.DataFrame(1e7, index=meses, columns=PAPEIS))


def test_rebalanceio_paga_o_custo_de_cada_ordem_com_a_liquidez_do_mes_anterior():
    # a 1ª compra é no fechamento de 24/03/2020: vale a liquidez de fevereiro (a de março, 5%, ainda não fechou)
    resultado, retornos = _motor()
    custos = Custos(emolumentos=0.0003, impacto=1.0)
    refeita = refazer(resultado, retornos, capital=1e6, custos=custos,
                      liquidez=_liquidez_constante({"2020-03-01": 0.05}), com_ir=False)
    alvo, primeiro = resultado.pesos_alvo.iloc[0], resultado.pesos_alvo.index[0]
    esperado = sum(custo_da_ordem(peso * 1e6, 0.001, 0.02, 1e7, custos) for peso in alvo)
    assert refeita.custos.iloc[0] == pytest.approx(esperado)
    crescimento = float((alvo * (1.0 + retornos.loc[primeiro, alvo.index])).sum())
    assert refeita.valores.loc[primeiro] == pytest.approx((1e6 - esperado) * crescimento)


DIAS = pd.DatetimeIndex(["2024-01-26", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01"])
DIAS_NO_MEIO_DO_MES = pd.bdate_range("2024-01-08", "2024-01-12")


def _dobra_e_divide(dias: pd.DatetimeIndex = DIAS) -> tuple[ResultadoBacktest, pd.DataFrame]:
    """R$ 30 mil em A no fechamento do 1º dia; A dobra no 3º; no fechamento do 3º a carteira vira metade A, metade B
    (vende R$ 30 mil de A, comprado a R$ 15 mil pelo preço médio). Retornos zero no resto."""
    alvos = pd.DataFrame({"A": [1.0, 0.5], "B": [0.0, 0.5]}, index=dias[[1, 3]])
    resultado = ResultadoBacktest("manual", "teste", pd.Series(1.0, index=dias), alvos,
                                  pd.Series([1.0, 0.5], index=alvos.index), 0.0)
    return resultado, pd.DataFrame({"A": [0.0, 1.0, 0.0, 0.0], "B": 0.0}, index=dias[1:])


def test_venda_pelo_preco_medio_paga_ir_no_fim_do_mes_e_a_saida_cobra_o_ganho_que_falta():
    # vendas de R$ 30 mil (> R$ 20 mil): IR de 15% x R$ 15 mil = R$ 2.250 em 31/01, pago vendendo um pouco de tudo; a
    # saída em 01/02 cobra 15% do ganho que sobrou em A (R$ 14.437,50 depois do pagamento)
    resultado, retornos = _dobra_e_divide()
    refeita = refazer(resultado, retornos, capital=30_000.0)
    assert refeita.impostos.to_dict() == pytest.approx({pd.Timestamp("2024-01-31"): 2_250.0})
    assert refeita.valores.loc["2024-01-31"] == pytest.approx(57_750.0)
    assert refeita.saida.loc["2024-02-01"] == pytest.approx(57_750.0 - 0.15 * 14_437.5)


def test_a_saida_no_meio_do_mes_soma_o_ganho_ja_realizado_ao_que_falta_realizar():
    # em 11/01: R$ 15 mil realizados na venda de 10/01 + R$ 15 mil ainda em A; vendas do mês R$ 30 mil + R$ 60 mil
    resultado, retornos = _dobra_e_divide(DIAS_NO_MEIO_DO_MES)
    refeita = refazer(resultado, retornos, capital=30_000.0)
    assert refeita.saida.loc["2024-01-11"] == pytest.approx(60_000.0 - 0.15 * 30_000.0)


def test_o_ultimo_pregao_da_amostra_nao_apura_o_mes_e_a_saida_cobra_tudo_junto():
    # 12/01 não é o último pregão de janeiro: nada de IR mensal; a saída soma realizado e não realizado num só mês
    resultado, retornos = _dobra_e_divide(DIAS_NO_MEIO_DO_MES)
    refeita = refazer(resultado, retornos, capital=30_000.0)
    assert refeita.impostos.empty
    assert refeita.valores.iloc[-1] == pytest.approx(60_000.0)
    assert refeita.saida.iloc[-1] == pytest.approx(60_000.0 - 0.15 * 30_000.0)


def test_etf_nao_tem_isencao_de_vendas_ate_20_mil():
    # com R$ 10 mil a venda de 30/01 é de R$ 10 mil (ganho R$ 5 mil): isenta em ação, 15% em ETF; idem na saída de 01/02
    resultado, retornos = _dobra_e_divide()
    acao = refazer(resultado, retornos, capital=10_000.0)
    etf = refazer(resultado, retornos, capital=10_000.0, isencao=0.0)
    assert acao.impostos.empty
    assert acao.saida.iloc[-1] == pytest.approx(20_000.0)
    assert etf.impostos.to_dict() == pytest.approx({pd.Timestamp("2024-01-31"): 750.0})
    assert etf.saida.iloc[-1] == pytest.approx(19_250.0 - 0.15 * 4_812.5)


def test_provento_reinvestido_entra_na_base_e_nao_paga_ir_como_ganho_de_capital():
    # 30/01: dos R$ 60 mil, provento de 25% do fechamento ex (R$ 12 mil) reinvestido -> base de A R$ 42 mil; a venda de
    # metade leva R$ 21 mil de base: ganho R$ 9 mil, IR R$ 1.350
    resultado, retornos = _dobra_e_divide()
    proventos = pd.DataFrame({"A": [0.0, 0.25, 0.0, 0.0]}, index=DIAS[1:])
    refeita = refazer(resultado, retornos, capital=30_000.0, proventos=proventos)
    assert refeita.impostos.to_dict() == pytest.approx({pd.Timestamp("2024-01-31"): 1_350.0})


def test_sem_ir_nao_ha_imposto_e_a_saida_e_o_patrimonio():
    resultado, retornos = _dobra_e_divide()
    refeita = refazer(resultado, retornos, capital=30_000.0, com_ir=False)
    assert refeita.impostos.empty
    pd.testing.assert_series_equal(refeita.saida, refeita.valores)
    assert refeita.valores.iloc[-1] == pytest.approx(60_000.0)
