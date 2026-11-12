"""The "Fermiônica" fictitious account (docs/superpowers/specs/2026-11-11-fermionic-ghost-design.md): its rule, its
record and its names in the simulator, the email and the screen."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from core import fermionica_fantasma as fantasma
from core import top5
from core.diversidade import log_det
from core.fermionica import estrategia_fermionica


def _treino(nomes: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    comum = rng.normal(0, 0.01, size=(300, 1))
    return pd.DataFrame(comum + rng.normal(0, 0.015, size=(300, len(nomes))), columns=nomes,
                        index=pd.bdate_range("2020-01-01", periods=300))


def _nota(mes, candidatos):
    return pd.Series(np.linspace(1, -1, len(candidatos)), index=candidatos)


def test_the_rule_is_the_log_det_selection_at_theta_one_half_on_15_names():
    assert fantasma.THETA == 0.5 and fantasma.CARTEIRA == "Fermiônica"
    nomes = [f"A{i:03d}3" for i in range(40)]
    setores = {nome: f"s{i % 8}" for i, nome in enumerate(nomes)}
    treino = _treino(nomes)
    pesos = fantasma.estrategia(_nota, setores)(treino)
    esperado = estrategia_fermionica(_nota, setores, theta=0.5, diversity=log_det)(treino)
    pd.testing.assert_series_equal(pesos, esperado)
    assert len(pesos) == 15 and pesos.to_numpy() == pytest.approx(1 / 15)
    assert pd.Series([setores[p] for p in pesos.index]).value_counts().max() <= 4


def test_its_record_trades_from_the_first_session_after_the_registration():
    composto = pd.DataFrame({"data": pd.to_datetime(["2026-10-27"] * 2), "ticker": ["X", "Y"], "peso": 0.5})
    pesos = pd.Series(1 / 15, index=[f"P{i}.SA" for i in range(15)])
    registro = top5.registros_devidos(composto, top5.vazio(), lambda _data: pesos, fantasma.INICIO)
    assert fantasma.INICIO == pd.Timestamp("2026-11-12")
    assert set(registro["data"]) == {fantasma.INICIO} and set(registro["base"]) == {pd.Timestamp("2026-10-27")}
    assert len(registro) == 15 and "P0" in set(registro["ticker"])


def test_the_simulator_the_email_and_the_screen_know_the_account():
    from core import alerta_email, simulador
    from ui import simulador as tela
    assert fantasma.CARTEIRA in simulador.CARTEIRAS and fantasma.CARTEIRA not in simulador.CARTEIRAS_COM_IR
    assert set(simulador.CARTEIRAS) <= set(alerta_email.CARTEIRAS)
    nome, descricao = alerta_email.CARTEIRAS[fantasma.CARTEIRA]
    assert "correla" in descricao and "15 ações" in descricao
    assert "Fermiônica" in tela.SOBRE_AS_CARTEIRAS and "12/11/2026" in tela.SOBRE_AS_CARTEIRAS


def test_the_modules_the_app_imports_do_not_need_pyscipopt():
    # the Streamlit app's requirements.txt has no PySCIPOpt: the screen imports the account's module
    codigo = ("import sys; sys.modules['pyscipopt'] = None\n"
              "import core.fermionica_fantasma, core.fermionica, ui.simulador\n")
    feito = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True)
    assert feito.returncode == 0, feito.stderr[-400:]
