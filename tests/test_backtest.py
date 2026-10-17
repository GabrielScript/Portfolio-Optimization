import numpy as np
import pandas as pd
import pytest

from core import reamostragem
from core.backtest import (backtest_pesos_fixos, com_janela, estrategia_equiponderada, estrategia_hrp,
                           estrategia_otimizada, estrategia_reamostrada, walk_forward)
from core.config import ConfigBacktest, ConfigOtimizacao, Objetivo
from core.estimation import estimar_momentos
from core.optimization import ErroOtimizacao, otimizar
from core.reamostragem import ConfigReamostragem

CFG_BT = ConfigBacktest(janela_treino=252, janela_teste=63, capital_inicial=1000.0, custo_bps=0.0)
CFG_OPT = ConfigOtimizacao(objetivo=Objetivo.KELLY, peso_maximo=0.5, n_ativos_max=None,
                           vol_maxima=None)


def test_sem_look_ahead_perturbando_o_futuro(retornos_sinteticos):
    """Os pesos do rebalanceamento t não podem mudar se alteramos retornos a partir de t."""
    estrategia = estrategia_otimizada(CFG_OPT, shrink_mu=True)
    original = walk_forward(retornos_sinteticos, estrategia, CFG_BT, "orig")
    corte = CFG_BT.janela_treino + 2 * CFG_BT.janela_teste
    perturbado = retornos_sinteticos.copy()
    perturbado.iloc[corte:] *= -3.0
    alterado = walk_forward(perturbado, estrategia, CFG_BT, "pert")
    data_corte = retornos_sinteticos.index[corte]
    antes = original.pesos_alvo.index <= data_corte
    pd.testing.assert_frame_equal(original.pesos_alvo[antes], alterado.pesos_alvo[antes])
    pd.testing.assert_series_equal(original.valores.loc[:retornos_sinteticos.index[corte - 1]],
                                   alterado.valores.loc[:retornos_sinteticos.index[corte - 1]])


def test_custos_reduzem_valor_final(retornos_sinteticos):
    estrategia = estrategia_hrp()
    sem = walk_forward(retornos_sinteticos, estrategia, CFG_BT, "sem")
    com = walk_forward(retornos_sinteticos, estrategia,
                       ConfigBacktest(janela_treino=252, janela_teste=63, capital_inicial=1000.0,
                                      custo_bps=25.0), "com")
    assert com.valores.iloc[-1] < sem.valores.iloc[-1]
    assert com.custos_totais > 0
    # montagem inicial a partir do caixa: giro = 1 (100% comprado)
    assert com.turnover.iloc[0] == pytest.approx(1.0)


def test_equiponderada_bate_calculo_manual(retornos_sinteticos):
    """Buy-and-hold entre rebalanceamentos: valor = V0 * mean(prod(1+r)) em cada bloco."""
    r = retornos_sinteticos
    res = walk_forward(r, estrategia_equiponderada(), CFG_BT, "1/N")
    valor = 1000.0
    for inicio in range(252, len(r), 63):
        bloco = r.iloc[inicio:inicio + 63]
        valor *= float((1 + bloco).prod().mean())
    assert res.valores.iloc[-1] == pytest.approx(valor, rel=1e-10)
    assert res.valores.index[0] == r.index[251]
    assert len(res.valores) == len(r) - 252 + 1


def test_falha_mantem_carteira_e_registra(retornos_sinteticos):
    chamadas = {"n": 0}

    def estrategia_instavel(treino):
        chamadas["n"] += 1
        if chamadas["n"] == 2:
            raise ErroOtimizacao("solver caiu")
        return pd.Series(1.0 / treino.shape[1], index=treino.columns)

    res = walk_forward(retornos_sinteticos, estrategia_instavel, CFG_BT, "x")
    assert len(res.falhas) == 1 and "solver caiu" in res.falhas[0]
    # carteira mantida: giro zero no rebalanceamento que falhou
    assert res.turnover.iloc[1] == pytest.approx(0.0)


def test_pesos_fixos_rotulado_in_sample(retornos_sinteticos):
    pesos = pd.Series(1.0 / 6, index=retornos_sinteticos.columns)
    res = backtest_pesos_fixos(retornos_sinteticos, pesos, CFG_BT, "fixo")
    assert res.metodologia == "in-sample"
    assert len(res.valores) == len(retornos_sinteticos) + 1


def test_rf_point_in_time_muda_max_sharpe(retornos_sinteticos):
    """Com CDI histórico, a rf usada em cada rebalanceamento é a vigente na data."""
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.5, n_ativos_max=None,
                           vol_maxima=None, taxa_livre_risco=0.0)
    rf_alta = pd.Series(0.0009, index=retornos_sinteticos.index)   # ~25% a.a.
    baixa = walk_forward(retornos_sinteticos, estrategia_otimizada(cfg, True), CFG_BT, "rf0")
    alta = walk_forward(retornos_sinteticos, estrategia_otimizada(cfg, True, rf_alta), CFG_BT, "rf25")
    assert not np.allclose(baixa.pesos_alvo.values, alta.pesos_alvo.values)


def test_reamostrada_com_bootstrap_identidade_iguala_a_otimizada_com_rf_da_data(retornos_sinteticos,
                                                                              monkeypatch):
    """Oráculo da ligação: mesma rf na data, mesmo shrink, mesmo cfg. Max Sharpe depende da rf."""
    monkeypatch.setattr(reamostragem, "indices_bootstrap_circular",
                        lambda n_obs, tamanho_bloco, gerador: np.arange(n_obs))
    cfg = ConfigOtimizacao(objetivo=Objetivo.MAX_SHARPE, peso_maximo=0.5, n_ativos_max=None,
                           vol_maxima=None, taxa_livre_risco=0.0)
    rf_alta = pd.Series(0.0009, index=retornos_sinteticos.index)
    otimizada = walk_forward(retornos_sinteticos, estrategia_otimizada(cfg, True, rf_alta), CFG_BT, "o")
    reamostrada = walk_forward(
        retornos_sinteticos,
        estrategia_reamostrada(cfg, True, rf_alta, ConfigReamostragem(n_reamostras=2)), CFG_BT, "r")
    np.testing.assert_allclose(reamostrada.pesos_alvo.values, otimizada.pesos_alvo.values, atol=1e-4)
    assert not reamostrada.falhas


def test_com_janela_entrega_so_os_ultimos_pregoes_e_devolve_os_pesos_da_estrategia(retornos_sinteticos):
    recebidos = []
    pesos_da_estrategia = pd.Series(1.0 / 6, index=retornos_sinteticos.columns)

    def espia(treino):
        recebidos.append(treino)
        return pesos_da_estrategia

    pesos = com_janela(espia, pregoes=100)(retornos_sinteticos.iloc[:300])
    assert recebidos[0].index.equals(retornos_sinteticos.index[200:300])
    assert pesos is pesos_da_estrategia


def test_estrategia_otimizada_com_meia_vida_otimiza_com_a_sigma_ewma(retornos_sinteticos):
    treino = retornos_sinteticos.iloc[:500]
    pesos = estrategia_otimizada(CFG_OPT, shrink_mu=True, meia_vida_cov=63)(treino)
    momentos = estimar_momentos(treino, shrink_mu=True, meia_vida_cov=63)
    pd.testing.assert_series_equal(pesos, otimizar(momentos.mu, momentos.cov, CFG_OPT).pesos)
    assert not np.allclose(pesos.values, estrategia_otimizada(CFG_OPT, shrink_mu=True)(treino).values,
                           atol=1e-3)


def test_com_janela_rejeita_treino_mais_curto_que_a_janela(retornos_sinteticos):
    with pytest.raises(ValueError, match="100"):
        com_janela(estrategia_equiponderada(), pregoes=100)(retornos_sinteticos.iloc[:99])


def test_historico_curto_rejeitado(retornos_sinteticos):
    with pytest.raises(ValueError):
        walk_forward(retornos_sinteticos.iloc[:200], estrategia_equiponderada(), CFG_BT, "x")
