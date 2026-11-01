"""What the two readings of docs/superpowers/specs/2026-11-01-todas-as-estrategias-design.md share: resuming after a crash, the nightly
window, and the curves that have no walk-forward of their own."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest


def test_a_finished_arm_is_read_from_the_disk_and_not_run_again(tmp_path):
    from scripts.todas_comum import resultado_salvo

    chamadas = []

    def rodar() -> dict[str, float]:
        chamadas.append(1)
        return {"valor": 1.5}

    caminho = tmp_path / "braco-fase0.pkl"
    assert resultado_salvo(caminho, rodar) == {"valor": 1.5}
    assert resultado_salvo(caminho, rodar) == {"valor": 1.5}
    assert len(chamadas) == 1


def test_an_arm_that_dies_leaves_nothing_a_resume_would_take_as_finished(tmp_path):
    from scripts.todas_comum import resultado_salvo

    def morre() -> None:
        raise RuntimeError("crash")

    caminho = tmp_path / "braco-fase0.pkl"
    with pytest.raises(RuntimeError):
        resultado_salvo(caminho, morre)
    assert not caminho.exists()


@pytest.mark.parametrize(("hora", "minuto", "espera"), [(21, 29, False), (21, 30, True), (23, 0, True), (0, 29, True), (0, 30, False),
                                                        (8, 24, False), (8, 25, True), (8, 49, True), (8, 50, False), (14, 0, False)])
def test_the_nightly_and_the_paper_windows_are_kept_free(hora, minuto, espera):
    from scripts.todas_comum import na_janela_reservada

    assert na_janela_reservada(datetime(2026, 11, 2, hora, minuto)) is espera


def _serie(valores: list[float]) -> pd.Series:
    return pd.Series(valores, index=pd.date_range("2020-01-31", periods=len(valores), freq="ME"), dtype=float)


def test_the_relative_stop_fires_at_thirty_percent_and_not_at_the_rules_twenty_five():
    from scripts.todas_us import data_do_stop_de_30

    referencia = _serie([1.0] * 7)
    rebalanceios = referencia.index[[3, 6]]
    assert data_do_stop_de_30(_serie([1.0, 1.0, 0.72, 0.72, 0.72, 0.72, 0.72]), rebalanceios, referencia) is None
    assert data_do_stop_de_30(_serie([1.0, 1.0, 0.69, 0.69, 0.69, 0.69, 0.69]), rebalanceios, referencia) == rebalanceios[0]


def test_the_mix_goes_back_to_target_at_each_rebalance_and_pays_only_on_what_changes():
    from scripts.todas_us import mistura_de_curvas

    datas = pd.date_range("2020-01-31", periods=7, freq="ME")
    a = pd.Series(np.cumprod([1.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0]), index=datas)
    b = pd.Series(1.0, index=datas)
    mistura = mistura_de_curvas({"a": a, "b": b}, {"a": 0.5, "b": 0.5}, janela_teste=3, custo_bps=0.0)
    assert mistura.index.equals(datas)
    # no cost: three months up 10% on a half that drifts, then back to 50/50 and flat
    assert mistura.iloc[3] == pytest.approx(0.5 * 1.331 + 0.5, rel=1e-9)
    assert mistura.iloc[-1] == pytest.approx(mistura.iloc[3], rel=1e-9)


def test_the_batchs_report_gives_each_phases_cagr():
    from scripts.todas_us import cagrs_do_relatorio

    texto = ("=== Phase 0: 11/2019 to 09/2026, 82 months; refusals {'Composto': 0} ===\n"
             "                                CAGR   Vol\nComposto                       0.185 0.183\nConservadora   0.172 0.196\n"
             "=== Phase 1: 12/2019 to 09/2026, 81 months; refusals {} ===\n"
             "                                CAGR   Vol\nComposto                       0.201 0.186\nConservadora   -0.188 0.202\n"
             "=== Gate: P[> SPY] and DSR ===\nComposto: no\n")
    assert cagrs_do_relatorio(texto, ("Composto", "Conservadora")) == {0: {"Composto": 0.185, "Conservadora": 0.172},
                                                                       1: {"Composto": 0.201, "Conservadora": -0.188}}


def test_a_phase_that_does_not_reproduce_the_batch_stops_the_reading():
    from scripts.todas_us import conferir_encanamento

    conferir_encanamento({"Composto": 0.185}, {"Composto": 0.1855}, 0)
    with pytest.raises(SystemExit):
        conferir_encanamento({"Composto": 0.185}, {"Composto": 0.187}, 0)


def test_the_t_bill_curve_starts_at_one_on_the_arms_dates():
    from scripts.todas_us import curva_do_t_bill

    datas = pd.date_range("2020-01-31", periods=3, freq="ME")
    rf = pd.Series([0.5, 0.01, 0.02], index=datas)        # the first month's return comes before the curve's start
    assert curva_do_t_bill(rf, datas).to_numpy() == pytest.approx([1.0, 1.01, 1.01 * 1.02])


def test_a_cash_column_for_the_arm_that_holds_cash_follows_the_t_bill_from_the_first_price():
    from scripts.todas_us import com_caixa_do_t_bill

    datas = pd.date_range("2020-01-31", periods=3, freq="ME")
    precos = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}, index=datas)
    rf = pd.Series([0.5, 0.01, 0.02], index=datas)
    from core.vencedoras import CAIXA
    assert com_caixa_do_t_bill(precos, rf)[CAIXA].to_numpy() == pytest.approx([1.0, 1.01, 1.01 * 1.02])
