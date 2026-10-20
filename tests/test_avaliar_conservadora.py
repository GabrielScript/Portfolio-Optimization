"""Regras de decisão do pré-registro do S2.1 (``scripts/avaliar_conservadora.py``), sem dados do lago."""

import numpy as np
import pandas as pd

from scripts.avaliar_conservadora import decidir, retornos_no_universo

MESES = pd.date_range("1998-01-01", "2026-09-01", freq="MS")


def _ics(principal: float, extensao: float, ruido: float = 0.05) -> pd.Series:
    """IC de cada mês: média ``extensao`` em 1998–2007 e ``principal`` de 2008 em diante, com ruído de semente fixa."""
    gerador = np.random.default_rng(1)
    medias = np.where(MESES < pd.Timestamp("2008-01-01"), extensao, principal)
    return pd.Series(medias + ruido * gerador.standard_normal(len(MESES)), index=MESES)


def test_aprovada_exige_o_portao_de_2008_e_ic_positivo_em_1998_2007():
    # IC 0,04 com t bem acima de 2 desde 2008, mas negativo na década anterior: não é aprovada
    so_o_portao = decidir(_ics(0.04, -0.02, ruido=0.01), [0.99, 0.99, 0.99])
    as_duas = decidir(_ics(0.04, 0.01, ruido=0.01), [0.99, 0.99, 0.99])
    assert (so_o_portao.condicao_1, so_o_portao.condicao_2, so_o_portao.aprovada, as_duas.aprovada) == (
        True, False, False, True)


def test_candidata_exige_p_de_0_95_nas_tres_fases_e_a_aprovacao():
    aprovada, reprovada = _ics(0.04, 0.01, ruido=0.01), _ics(0.04, -0.02, ruido=0.01)
    assert [decidir(aprovada, [0.95, 0.97, 0.99]).candidata, decidir(aprovada, [0.95, 0.94, 0.99]).candidata,
            decidir(reprovada, [0.99, 0.99, 0.99]).candidata] == [True, False, False]


def test_retornos_fora_do_universo_do_mes_ficam_vazios():
    meses = pd.date_range("2020-01-01", periods=2, freq="MS")
    retornos = pd.DataFrame({"AAAA3": [0.01, 0.02], "BBBB3": [0.03, 0.04]}, index=meses)
    universo = pd.DataFrame({"month": [meses[0].date(), meses[0].date(), meses[1].date()],
                             "security": ["AAAA3", "BBBB3", "AAAA3"]})
    esperado = pd.DataFrame({"AAAA3": [0.01, 0.02], "BBBB3": [0.03, np.nan]}, index=meses)
    pd.testing.assert_frame_equal(retornos_no_universo(retornos, universo, meses), esperado)
