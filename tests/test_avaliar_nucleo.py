"""Pré-registro do núcleo BOVA11 + IVVB11 (``scripts/avaliar_nucleo.py``): começo do trecho e regra de decisão."""

import numpy as np
import pandas as pd

from scripts.avaliar_nucleo import comeco_comum, decidir, excessos_por_subperiodo


def test_trecho_comeca_no_1o_pregao_com_os_dois_cotados_e_guarda_as_lacunas_depois():
    datas = pd.bdate_range("2014-04-24", periods=6)
    precos = pd.DataFrame({"BOVA11": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
                           "IVVB11": [np.nan, np.nan, 60.0, 61.0, np.nan, 63.0]}, index=datas)
    assert comeco_comum(precos).index.equals(datas[2:])


def test_candidato_exige_p_log_de_0_95_nas_tres_fases_e_excesso_em_2_dos_3_subperiodos():
    dois_de_tres = [pd.Series({"2014–2018": 0.02, "2019–2022": -0.01, "2023–2026": 0.03})] * 3
    so_um = [pd.Series({"2014–2018": 0.02, "2019–2022": -0.01, "2023–2026": -0.03})] + dois_de_tres[1:]
    assert [decidir([0.95, 0.97, 0.99], dois_de_tres).candidato, decidir([0.95, 0.94, 0.99], dois_de_tres).candidato,
            decidir([0.99, 0.99, 0.99], so_um).candidato] == [True, False, False]


def test_excesso_do_subperiodo_conta_desde_o_fechamento_anterior_ao_1o_dia():
    datas = pd.bdate_range("2018-12-27", "2019-01-04")       # 27, 28 e 31/12; 1º a 4/1
    valores = pd.Series([1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2], index=datas)
    bova = pd.Series(1.0, index=datas)
    excessos = excessos_por_subperiodo(valores, bova, {"2018": ("2018-01-01", "2018-12-31"),
                                                       "2019": ("2019-01-01", "2019-12-31")})
    assert excessos["2018"] == 0.0 and excessos["2019"] > 0.0
