"""Alvo de volatilidade com EGARCH (DECISOES 55): exposição, deriva entre reajustes e falha de convergência."""

import numpy as np
import pandas as pd
import pytest

from core.vol_alvo import curva_com_alvo, exposicao, vol_anual_da_previsao, vol_egarch


def test_exposicao_e_o_alvo_sobre_a_prevista_com_teto_de_100_por_cento():
    # vol prevista no dobro do alvo: metade exposta; abaixo do alvo, sem alavancagem
    assert [exposicao(0.40, 0.20), exposicao(0.10, 0.20)] == pytest.approx([0.5, 1.0])


DATAS = pd.bdate_range("2024-01-01", periods=7)
VALORES = pd.Series(100.0 * np.cumprod([1.0, 1.01, 0.98, 1.03, 1.01, 0.98, 1.04]), index=DATAS)
RF = pd.Series(0.001, index=DATAS)


def _previsao_em_multiplos_do_alvo(*multiplos: float):
    """Vol prevista = cada múltiplo, em ordem, da vol realizada da janela (o alvo): exposição = 1 / múltiplo."""
    fila = iter(multiplos)

    def prever(janela: np.ndarray) -> float | None:
        multiplo = next(fila)
        return None if multiplo is None else multiplo * np.std(janela, ddof=1) * np.sqrt(252)
    return prever


def test_curva_deriva_entre_reajustes_e_paga_custo_sobre_a_mudanca_de_exposicao():
    # janela de 2 retornos, reajuste a cada 2: decide no fechamento do pregão 2 (exposição 0,5, vinda de 1,0) e no do
    # 4 (de volta a 1,0, vinda da 0,5 que derivou); entre os reajustes a parte exposta rende a curva e o resto o CDI
    curva = curva_com_alvo(VALORES, RF, _previsao_em_multiplos_do_alvo(2.0, 1.0), janela=2, passo=2, custo_bps=10.0)
    valor = VALORES.iloc[2] * (1 - 0.5 * 0.001)
    exposto, caixa, esperado = valor * 0.5, valor * 0.5, [VALORES.iloc[2]]
    for retorno in (0.03, 0.01):
        exposto, caixa = exposto * (1 + retorno), caixa * 1.001
        esperado.append(exposto + caixa)
    deriva = exposto / (exposto + caixa)
    valor = (exposto + caixa) * (1 - (1.0 - deriva) * 0.001)
    esperado += [valor * 0.98, valor * 0.98 * 1.04]
    assert curva.valores.tolist() == pytest.approx(esperado)
    assert curva.valores.index.equals(DATAS[2:])
    assert (curva.exposicoes.tolist(), curva.giro.tolist()) == (pytest.approx([0.5, 1.0]),
                                                                pytest.approx([0.5, 1.0 - deriva]))


def test_exposicao_do_reajuste_so_usa_os_retornos_ate_o_fechamento_da_decisao():
    # cada janela que o ajuste recebe = os 10 últimos retornos até o fechamento do pregão da decisão, nem um a mais
    datas = pd.bdate_range("2024-01-01", periods=61)
    retornos = pd.Series(np.random.default_rng(3).normal(0.0005, 0.015, 60), index=datas[1:])
    valores = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + retornos.to_numpy()]), index=datas)
    janelas = []

    def prever(janela: np.ndarray) -> float:
        janelas.append(janela.copy())
        return 1.3 * np.abs(janela).mean() * np.sqrt(252)

    curva = curva_com_alvo(valores, pd.Series(0.0004, index=datas), prever, janela=10, passo=5)
    assert len(janelas) == len(curva.exposicoes) == 10
    for janela, decisao in zip(janelas, curva.exposicoes.index):
        np.testing.assert_allclose(janela, retornos.loc[:decisao].iloc[-10:].to_numpy(), rtol=1e-12)


def test_ajuste_que_nao_converge_mantem_a_exposicao_que_derivou_sem_negocio():
    # o 2º reajuste falha: fica a exposição de 0,5 depois da deriva dos pregões 3 e 4, giro 0, e a data é registrada
    curva = curva_com_alvo(VALORES, RF, _previsao_em_multiplos_do_alvo(2.0, None), janela=2, passo=2, custo_bps=10.0)
    exposto, caixa = 0.5 * 1.03 * 1.01, 0.5 * 1.001 * 1.001
    assert curva.exposicoes.tolist() == pytest.approx([0.5, exposto / (exposto + caixa)])
    assert (curva.giro.tolist(), len(curva.falhas)) == (pytest.approx([0.5, 0.0]), 1)


def test_vol_anual_da_previsao_e_a_media_das_variancias_e_recusa_previsao_degenerada():
    # variâncias diárias em %² (o ajuste usa retornos x 100); o EGARCH às vezes "converge" para 0 ou infinito
    assert vol_anual_da_previsao(np.array([1.0, 3.0])) == pytest.approx(np.sqrt(2.0 * 252) / 100)
    assert [vol_anual_da_previsao(np.array(v)) for v in ([0.0, 0.0], [1.0, np.inf], [1.0, np.nan])] == [None] * 3


def _garch_simulado(n: int = 504) -> np.ndarray:
    """Retornos diários (fração) de um GARCH(1,1) com agrupamento de vol: variância em %² = 0,05 + 0,10 r² + 0,85 σ²."""
    choques = np.random.default_rng(11).standard_normal(n)
    variancia, retornos = 1.0, np.empty(n)
    for dia, choque in enumerate(choques):
        retornos[dia] = np.sqrt(variancia) * choque
        variancia = 0.05 + 0.10 * retornos[dia] ** 2 + 0.85 * variancia
    return retornos / 100.0


def test_vol_egarch_repete_a_previsao_com_a_mesma_semente_e_falha_em_serie_constante():
    retornos = _garch_simulado()
    previsao = vol_egarch(retornos)
    assert previsao == vol_egarch(retornos) and 0.05 < previsao < 1.0
    assert vol_egarch(np.zeros(504)) is None


def test_previsao_a_mais_de_10_vezes_da_vol_realizada_conta_como_falha():
    # nas curvas fora da amostra o EGARCH chegou a "prever" 1e-33 e 6e15 vezes a vol da janela: ponto degenerado
    exposto, caixa = 0.5 * 1.03 * 1.01, 0.5 * 1.001 * 1.001
    for multiplo in (10.5, 1 / 10.5):
        curva = curva_com_alvo(VALORES, RF, _previsao_em_multiplos_do_alvo(2.0, multiplo), janela=2, passo=2)
        assert (curva.exposicoes.iloc[-1], len(curva.falhas)) == (pytest.approx(exposto / (exposto + caixa)), 1)
    curva = curva_com_alvo(VALORES, RF, _previsao_em_multiplos_do_alvo(2.0, 9.5), janela=2, passo=2)
    assert (curva.exposicoes.iloc[-1], len(curva.falhas)) == (pytest.approx(1 / 9.5), 0)
