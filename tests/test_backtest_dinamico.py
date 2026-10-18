"""Walk-forward com universo ponto no tempo (deslistadas incluídas), sem rede."""

import numpy as np
import pandas as pd
import pytest

from core.backtest import estrategia_equiponderada, walk_forward_dinamico
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
    # C estreia em março: no rebalanceio de abril (pregão 60) o treino dele tem só ~30 pregões
    precos = _precos("A", "B", "C")
    precos.loc[:"2024-02-29", "C"] = np.nan
    universo = _universo({mes: ["A", "B", "C"] for mes in MESES})
    bt = walk_forward_dinamico(precos, universo, estrategia_equiponderada(), CFG, "1/N")
    assert bt.pesos_alvo.iloc[0].to_dict() == pytest.approx({"A": 0.5, "B": 0.5, "C": 0.0})
