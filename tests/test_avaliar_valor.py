"""Regras do pré-registro do S2.2 e do S2.3 (``scripts/avaliar_valor.py``), sem dados do lago."""

import numpy as np
import pandas as pd

from core.backtest import retornos_da_carteira
from scripts.avaliar_valor import corte_dos_precos, decidir

MESES = pd.date_range("2012-01-01", "2026-09-01", freq="MS")


def _ics(media: float, ruido: float = 0.01) -> pd.Series:
    return pd.Series(media + ruido * np.random.default_rng(1).standard_normal(len(MESES)), index=MESES)


def test_aprovada_e_o_portao_de_2012_a_2026_sem_condicao_de_extensao():
    # sem fundamentos da CVM antes de 2010, não há década fora da amostra: o portão de 2012-01 a 2026-09 decide sozinho
    assert [decidir(_ics(0.03), [0.5] * 3).aprovada, decidir(_ics(0.01), [0.5] * 3).aprovada] == [True, False]


def test_candidata_exige_p_de_0_95_nas_tres_fases_e_a_aprovacao():
    aprovada, reprovada = _ics(0.03), _ics(0.01)
    assert [decidir(aprovada, [0.95, 0.97, 0.99]).candidata, decidir(aprovada, [0.95, 0.94, 0.99]).candidata,
            decidir(reprovada, [0.99, 0.99, 0.99]).candidata] == [True, False, False]


def test_corte_poe_a_1a_decisao_da_fase_0_no_1o_pregao_do_ano():
    # o motor decide pela 1ª vez no pregão de índice janela_treino dos retornos (o 1º preço não tem retorno)
    datas = pd.bdate_range("2011-06-01", "2012-03-30")
    corte = corte_dos_precos(datas, pd.Timestamp("2012-01-01"), janela_treino=60)
    precos = pd.DataFrame({"AAAA3": np.linspace(10.0, 20.0, len(datas))}, index=datas)
    assert retornos_da_carteira(precos.iloc[corte:]).index[60] == pd.Timestamp("2012-01-02")
