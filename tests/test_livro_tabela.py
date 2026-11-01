"""The screen's table of the book's strategies: one row per arm and per market, from the saved curves and gates."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def _resultado(crescimento_mensal: float, giro: float = 1.0, recusas: int = 0) -> SimpleNamespace:
    datas = pd.date_range("2020-01-31", periods=25, freq="ME")
    valores = pd.Series(np.cumprod(np.r_[1.0, np.full(24, 1.0 + crescimento_mensal)]), index=datas)
    return SimpleNamespace(valores=valores, recusas=[None] * recusas, turnover_anual_medio=giro)


def _portao(nome: str, passa: bool = False) -> pd.DataFrame:
    return pd.DataFrame([{"braco": nome, "p_composto": [0.2, 0.1, 0.3], "p_conservadora": [0.4, 0.5, 0.6], "p_referencia": [0.7, 0.8, 0.9],
                          "dsr": [0.5, 0.4, 0.6], "passa": passa}])


def test_a_row_takes_each_arms_worst_and_best_phase_and_its_gate():
    from scripts.livro_tabela import linhas_do_lote

    resultados = {"A": {0: _resultado(0.01, giro=2.0), 1: _resultado(0.02, giro=4.0, recusas=3), 2: _resultado(0.015, giro=3.0)}}
    linha = linhas_do_lote(1, resultados, _portao("A"), 12, lambda r: r.turnover_anual_medio).iloc[0]
    assert linha["braco"] == "A"
    assert linha["lote"] == 1
    assert linha["cagr_min"] == pytest.approx(1.01 ** 12 - 1.0)
    assert linha["cagr_max"] == pytest.approx(1.02 ** 12 - 1.0)
    assert linha["giro"] == pytest.approx(3.0)
    assert linha["recusas"] == 3
    assert linha["p_composto"] == 0.1
    assert linha["p_conservadora"] == 0.4
    assert linha["p_referencia"] == 0.7
    assert linha["dsr"] == 0.4
    assert not linha["passa"]


def test_the_drawdown_is_the_worst_phases_and_the_volatility_annual():
    from scripts.livro_tabela import linhas_do_lote

    datas = pd.date_range("2020-01-31", periods=5, freq="ME")
    curva = SimpleNamespace(valores=pd.Series([1.0, 1.1, 0.88, 0.99, 1.1], index=datas), recusas=[], turnover_anual_medio=0.0)
    linha = linhas_do_lote(2, {"A": {0: curva}}, _portao("A"), 12, lambda r: r.turnover_anual_medio).iloc[0]
    assert linha["queda"] == pytest.approx(-0.20)
    assert linha["vol"] == pytest.approx(curva.valores.pct_change().dropna().std() * np.sqrt(12))


def test_an_arm_without_a_gate_row_stops_the_table():
    from scripts.livro_tabela import linhas_do_lote

    with pytest.raises(KeyError):
        linhas_do_lote(1, {"B": {0: _resultado(0.01)}}, _portao("A"), 12, lambda r: r.turnover_anual_medio)


def _tabela(passa: bool = False) -> pd.DataFrame:
    return pd.DataFrame([{"lote": 1, "braco": "HRP-projetado", "cagr_min": 0.17, "cagr_max": 0.2, "vol": 0.2, "queda": -0.43, "giro": 2.0, "recusas": 0,
                          "p_composto": 0.1, "p_conservadora": 0.47, "p_referencia": 0.9, "dsr": 0.5, "passa": passa},
                         {"lote": 2, "braco": "Kelly-exato", "cagr_min": 0.1, "cagr_max": 0.12, "vol": 0.3, "queda": -0.6, "giro": 5.0, "recusas": 1,
                          "p_composto": 0.05, "p_conservadora": 0.2, "p_referencia": 0.6, "dsr": 0.3, "passa": False}])


def _tela(pasta: str):
    from pathlib import Path

    from ui.estrategias_livro import carregar, secao
    secao(carregar(Path(pasta)))


def test_the_screen_shows_each_market_it_has_sorted_by_return_and_says_which_is_missing(tmp_path):
    _tabela().to_parquet(tmp_path / "livro-tabela-b3.parquet", index=False)
    tela = AppTest.from_function(_tela, args=(str(tmp_path),), default_timeout=60).run()
    assert not tela.exception
    assert tela.dataframe[0].value["Estratégia"].tolist() == ["HRP-projetado", "Kelly-exato"]
    assert "Nenhuma estratégia" in tela.warning[0].value
    assert any("Ações americanas" in info.value for info in tela.info)


def test_the_screen_names_an_arm_that_passed_only_as_a_candidate(tmp_path):
    _tabela(passa=True).to_parquet(tmp_path / "livro-tabela-us.parquet", index=False)
    tela = AppTest.from_function(_tela, args=(str(tmp_path),), default_timeout=60).run()
    assert "HRP-projetado (Ações americanas)" in tela.warning[0].value
    assert "candidata" in tela.warning[0].value


def test_the_screen_says_so_before_any_table_exists(tmp_path):
    tela = AppTest.from_function(_tela, args=(str(tmp_path),), default_timeout=60).run()
    assert not tela.exception
    assert not tela.dataframe
    assert "ainda não terminou" in tela.info[0].value
