"""Walk-forward com universo ponto no tempo (deslistadas incluídas), sem rede."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import (ErroUniverso, estrategia_buy_and_hold_ativo, estrategia_equiponderada,
                           retornos_da_carteira, walk_forward, walk_forward_dinamico)
from core.config import ConfigBacktest

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
