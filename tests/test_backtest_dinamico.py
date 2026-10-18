"""Walk-forward com universo ponto no tempo (deslistadas incluídas), sem rede."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import (CarteiraRecusada, Congelamento, ErroPrimeiroRebalanceio, ErroUniverso,
                           backtest_pesos_fixos, estrategia_buy_and_hold_ativo, estrategia_equiponderada,
                           retornos_da_carteira, walk_forward, walk_forward_dinamico)
from core.config import ConfigBacktest
from core.risco import Regra, Violacao

DATAS = pd.bdate_range("2024-01-01", periods=150)  # janeiro a julho de 2024
CFG = ConfigBacktest(janela_treino=60, janela_teste=30, capital_inicial=1_000.0, custo_bps=0.0)
MESES = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07"]


def _precos(*papeis: str) -> pd.DataFrame:
    gerador = np.random.default_rng(3)
    retornos = gerador.normal(0.0005, 0.01, (len(DATAS), len(papeis)))
    return pd.DataFrame(10 * np.cumprod(1 + retornos, axis=0), index=DATAS, columns=list(papeis))


def _universo(papeis_por_mes: dict[str, list[str]]) -> pd.DataFrame:
    """Mês (AAAA-MM) -> papéis elegíveis no começo do mês."""
    linhas = [(pd.Timestamp(f"{mes}-01"), papel) for mes, papeis in papeis_por_mes.items() for papel in papeis]
    return pd.DataFrame(linhas, columns=["month", "security"])


def test_so_os_elegiveis_do_mes_recebem_peso():
    universo = _universo({mes: ["A", "B"] for mes in MESES})
    bt = walk_forward_dinamico(_precos("A", "B", "C"), universo, estrategia_equiponderada(), CFG, "1/N")
    assert (bt.pesos_alvo["C"] == 0).all() and np.allclose(bt.pesos_alvo[["A", "B"]], 0.5)


def test_elegivel_sem_a_janela_de_treino_inteira_fica_de_fora():
    # C estreia em março: no 1º rebalanceio (26/03, treino de 60 pregões) o treino dele tem só ~18 pregões
    precos = _precos("A", "B", "C")
    precos.loc[:"2024-02-29", "C"] = np.nan
    universo = _universo({mes: ["A", "B", "C"] for mes in MESES})
    bt = walk_forward_dinamico(precos, universo, estrategia_equiponderada(), CFG, "1/N")
    assert bt.pesos_alvo.iloc[0].to_dict() == pytest.approx({"A": 0.5, "B": 0.5, "C": 0.0})


def test_aceita_o_mes_como_data_do_export_do_lago():
    # o universe_monthly.parquet do lago traz ``month`` como datetime.date, não Timestamp
    universo = _universo({mes: ["A", "B"] for mes in MESES})
    universo["month"] = universo["month"].dt.date
    bt = walk_forward_dinamico(_precos("A", "B", "C"), universo, estrategia_equiponderada(), CFG, "1/N")
    assert (bt.pesos_alvo["C"] == 0).all() and np.allclose(bt.pesos_alvo[["A", "B"]], 0.5)


def test_rebalanceio_sem_nenhum_elegivel_e_erro_e_nao_vira_plano_b():
    # universo só até março: o rebalanceio de 07/05 não tem ninguém; o fallback do motor (manter ou 1/N em todas
    # as colunas) daria um resultado com cara de válido
    universo = _universo({mes: ["A", "B"] for mes in MESES[:3]})
    with pytest.raises(ErroUniverso, match="2024-05-07"):
        walk_forward_dinamico(_precos("A", "B", "C"), universo, estrategia_equiponderada(), CFG, "1/N")


def test_o_movimento_durante_dias_sem_negocio_entra_no_retorno_da_carteira():
    # A: 10, dois pregões sem negócio, volta a 12 (+20%); B parado em 10. 1/N sem custo: 1.000 → 1.100
    precos = pd.DataFrame(10.0, index=DATAS, columns=["A", "B"])
    precos.iloc[100:102, 0] = np.nan
    precos.iloc[102:, 0] = 12.0
    universo = _universo({mes: ["A", "B"] for mes in MESES})
    bt = walk_forward_dinamico(precos, universo, estrategia_equiponderada(), CFG, "1/N")
    assert bt.valores.iloc[-1] == pytest.approx(1_100.0)


def test_referencia_fora_do_universo_roda_nas_mesmas_datas_da_estrategia():
    # BOVA11 não é elegível (ETF): roda no walk-forward comum sobre os mesmos retornos da carteira
    precos = _precos("A", "B", "BOVA11")
    universo = _universo({mes: ["A", "B"] for mes in MESES})
    bt = walk_forward_dinamico(precos, universo, estrategia_equiponderada(), CFG, "1/N")
    ref = walk_forward(retornos_da_carteira(precos)[["BOVA11"]], estrategia_buy_and_hold_ativo("BOVA11"), CFG,
                       "BOVA11")
    assert ref.valores.index.equals(bt.valores.index)
    assert ref.valores.iloc[-1] == pytest.approx(1_000.0 * precos["BOVA11"].iloc[-1] / precos["BOVA11"].iloc[60])


def test_falha_no_primeiro_rebalanceio_dinamico_e_erro():
    # sem carteira para manter, o plano B seria 1/N em todas as colunas (deslistadas e papéis que ainda não existiam)
    def sempre_falha(treino):
        raise ValueError("solver não convergiu")

    precos = _precos("A", "B")
    with pytest.raises(ErroPrimeiroRebalanceio, match="solver não convergiu"):
        walk_forward_dinamico(precos, _universo({mes: ["A", "B"] for mes in MESES}), sempre_falha, CFG, "falha")


def test_carteira_recusada_pelas_regras_mantem_a_atual_e_conta_a_parte_das_falhas():
    # 2º rebalanceio (07/05) recusado: a carteira derivada fica, e a recusa não se mistura com falha de solver
    chamadas = []

    def recusa_a_segunda(treino):
        chamadas.append(treino.index[-1])
        if len(chamadas) == 2:
            raise CarteiraRecusada([Violacao(Regra.SETOR, "Financeiro", 0.36, 0.30)])
        return pd.Series(0.5, index=treino.columns)

    bt = walk_forward_dinamico(_precos("A", "B"), _universo({mes: ["A", "B"] for mes in MESES}), recusa_a_segunda,
                               CFG, "regras")
    segundo = bt.pesos_alvo.index[1]
    assert list(bt.recusas) == [f"{segundo:%Y-%m-%d}"] and bt.recusas[f"{segundo:%Y-%m-%d}"][0].regra is Regra.SETOR
    assert bt.falhas == [] and bt.turnover.iloc[1] == 0.0


def test_rebalanceio_depois_de_perda_de_5_por_cento_vai_para_o_primeiro_pregao_livre():
    # a carteira perde 6% no pregão de decisão do 2º rebalanceio (posição 90): nenhuma ordem no seguinte, o
    # rebalanceio vai para a posição 91; o 3º continua na data dele (120)
    precos = pd.DataFrame(10.0, index=DATAS, columns=["A", "B"])
    precos.iloc[90:] = 9.4
    universo = _universo({mes: ["A", "B"] for mes in MESES})
    bt = walk_forward_dinamico(precos, universo, estrategia_equiponderada(), CFG, "1/N", Congelamento())
    retornos = retornos_da_carteira(precos)
    assert list(bt.pesos_alvo.index) == [retornos.index[60], retornos.index[91], retornos.index[120]]
    assert bt.adiamentos == {f"{retornos.index[90]:%Y-%m-%d}": f"{retornos.index[91]:%Y-%m-%d}"}


def test_mistura_congela_pelos_retornos_do_satelite():
    # núcleo e satélite parados; o satélite (série de fora) perde 6% no pregão 29: a volta aos pesos de 30 vai a 31
    retornos = pd.DataFrame(0.0, index=DATAS[:90], columns=["núcleo", "satélite"])
    satelite = pd.Series(0.0, index=retornos.index)
    satelite.iloc[29] = -0.06
    bt = backtest_pesos_fixos(retornos, pd.Series({"núcleo": 0.7, "satélite": 0.3}), CFG, "mistura",
                              Congelamento(retornos=satelite))
    assert list(bt.pesos_alvo.index) == [retornos.index[0], retornos.index[31], retornos.index[60]]


def test_sem_perda_grande_o_congelamento_nao_muda_nada():
    # vol diária de ~0,7% na carteira: nenhum pregão de -5%, e a curva tem de sair igual, bit a bit
    universo = _universo({mes: ["A", "B"] for mes in MESES})
    sem = walk_forward_dinamico(_precos("A", "B"), universo, estrategia_equiponderada(), CFG, "1/N")
    com = walk_forward_dinamico(_precos("A", "B"), universo, estrategia_equiponderada(), CFG, "1/N", Congelamento())
    pd.testing.assert_series_equal(com.valores, sem.valores, check_exact=True)
    assert com.adiamentos == {} and list(com.pesos_alvo.index) == list(sem.pesos_alvo.index)
