"""The 12/3 ghost (core/mista.py): the Composite with 12 names plus the 3 US BDRs of highest 12-1 momentum (the user,
2026-11-07; DECISOES 298-299)."""

import pandas as pd
import pytest

from core import mista, top5
from core.composto import estrategia_por_nota
from core.risco import LIMITES

SETORES = {f"{letra}.SA": setor for letra, setor in zip("ABCDEFGHIJKLMNOP", ["Bancos", "Bancos", "Bancos", "Bancos",
                                                                              "Energia", "Energia", "Energia", "Saúde",
                                                                              "Varejo", "Varejo", "Mineração", "Telecom",
                                                                              "Saneamento", "Seguros", "Papel", "Logística"])}
NOTAS = pd.Series({ticker: float(16 - posicao) for posicao, ticker in enumerate(SETORES)})


def _treino() -> pd.DataFrame:
    datas = pd.bdate_range("2024-01-01", periods=300)
    passo = pd.Series(range(300), index=datas)
    return pd.DataFrame({ticker: 0.001 * ((passo + posicao) % 7 - 3) for posicao, ticker in enumerate(NOTAS.index)},
                        index=datas)


def test_the_br_sleeve_is_the_composite_with_12_names_under_the_whole_portfolios_caps():
    # 4 banks in 12 names are 33% of the sleeve but 4/15 = 26.7% of the whole (DECISOES 298's addendum)
    assert (mista.NOMES_BR, mista.NOMES_EUA) == (12, 3)
    assert mista.limites_br().teto_setor == pytest.approx(LIMITES.teto_setor * 15 / 12)
    nota = lambda mes, candidatos: NOTAS.reindex(candidatos)       # noqa: E731
    pesos = mista.estrategia_br(nota, SETORES)(_treino())
    esperado = estrategia_por_nota(nota, SETORES, nomes=12, limites=mista.limites_br())(_treino())
    pd.testing.assert_series_equal(pesos, esperado)
    assert (pesos > 0).sum() == 12 and sum(pesos[[f"{letra}.SA" for letra in "ABCD"]] > 0) == 4


def test_the_portfolio_is_one_fifteenth_per_name():
    br = pd.Series(1 / 12, index=[f"P{i}.SA" for i in range(12)])
    pesos = mista.combinar(br, ["MUTC34", "W1DC34", "ITLC34"])
    assert len(pesos) == 15 and pesos.to_numpy() == pytest.approx([1 / 15] * 15)
    assert pesos.sum() == pytest.approx(1.0)


def test_fewer_eligible_us_names_give_their_slots_to_the_br_sleeve():
    br = pd.Series(1 / 12, index=[f"P{i}.SA" for i in range(12)])
    pesos = mista.combinar(br, ["MUTC34"])
    assert pesos["MUTC34"] == pytest.approx(1 / 15)
    assert pesos.drop("MUTC34").to_numpy() == pytest.approx([(14 / 15) / 12] * 12)


def test_its_record_trades_from_the_first_session_after_the_registration():
    composto = pd.DataFrame({"data": pd.to_datetime(["2026-10-27"] * 2), "ticker": ["X", "Y"], "peso": 0.5})
    pesos = pd.Series(1 / 15, index=[*(f"P{i}.SA" for i in range(12)), "MUTC34", "W1DC34", "ITLC34"])
    registro = top5.registros_devidos(composto, top5.vazio(), lambda _data: pesos, mista.INICIO)
    assert mista.INICIO == pd.Timestamp("2026-11-09")
    assert set(registro["data"]) == {mista.INICIO} and set(registro["base"]) == {pd.Timestamp("2026-10-27")}
    assert "P0" in set(registro["ticker"]) and "MUTC34" in set(registro["ticker"])


def test_the_email_and_the_screen_say_what_the_12_3_is():
    from core import alerta_email, simulador
    from ui import simulador as tela
    assert set(simulador.CARTEIRAS) <= set(alerta_email.CARTEIRAS)          # every account has a name in the email
    nome, descricao = alerta_email.CARTEIRAS["12/3"]
    assert "BDR" in descricao and "Imposto de Renda" in descricao
    assert "12/3" in tela.SOBRE_AS_CARTEIRAS and "09/11/2026" in tela.SOBRE_AS_CARTEIRAS
