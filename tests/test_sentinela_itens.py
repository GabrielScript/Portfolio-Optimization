import numpy as np
import pandas as pd

from core.sentinela.itens import MACRO, Item, Serie, itens_aluguel, itens_cvm, itens_macro, itens_mercado


def _fatos() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["PETR4", "VALE3", "ITUB4"],
        "empresa": ["PETROLEO BRASILEIRO S.A.", "VALE S.A.", "ITAU UNIBANCO"],
        "categoria": ["Fato Relevante", "Comunicado ao Mercado", "Fato Relevante"],
        "assunto": ["Pedido de recuperação judicial de 1.234,5 milhões", "Apresentação", "x" * 900],
        "data": ["2026-11-04", "2026-11-03", "2026-11-02"],
        "link": ["https://rad/1", "https://rad/2", "https://rad/3"]})


def test_cvm_item_per_filing_of_the_given_names_with_verbatim_subject_and_a_stable_id():
    itens = itens_cvm(_fatos(), ["PETR4", "ITUB4"])
    assert [item.ticker for item in itens] == ["PETR4", "ITUB4"]
    petr = itens[0]
    assert petr.fonte == "cvm" and petr.data == "2026-11-04"
    assert "Fato Relevante" in petr.texto and "Pedido de recuperação judicial de 1.234,5 milhões" in petr.texto
    assert petr.id.startswith("cvm-") and petr.id == itens_cvm(_fatos(), ["PETR4"])[0].id


def test_cvm_text_is_capped_at_600_characters():
    assert len(itens_cvm(_fatos(), ["ITUB4"])[0].texto) == 600


def test_item_line_is_id_source_date_text():
    item = Item("cvm-1", "cvm", "2026-11-04", "PETR4", "texto")
    assert item.linha() == "cvm-1 | cvm | 2026-11-04 | texto"


def _precos(ultimo_retorno: float, ultimo_volume: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    gerador = np.random.default_rng(0)
    dias = pd.bdate_range("2026-06-01", periods=80)
    retornos = gerador.normal(0, 0.01, len(dias))
    retornos[-1] = ultimo_retorno
    fechamentos = pd.DataFrame({"PETR4": 30 * np.cumprod(1 + retornos)}, index=dias)
    volumes = pd.DataFrame({"PETR4": np.full(len(dias), 1e6)}, index=dias)
    volumes.iloc[-1] = ultimo_volume
    return fechamentos, volumes


def test_a_quiet_session_gives_no_market_item():
    assert itens_mercado(*_precos(0.002, 1.1e6), sessoes=1) == []


def test_a_move_beyond_2_5_sigma_gives_one_item_whose_numbers_are_written_in_its_text():
    itens = itens_mercado(*_precos(-0.08, 1.1e6), sessoes=1)
    assert len(itens) == 1
    item = itens[0]
    assert item.fonte == "mercado" and item.ticker == "PETR4" and item.data == "2026-09-18"
    assert "-8,00%" in item.texto and "18/09/2026" in item.texto


def test_volume_beyond_3x_the_21_session_median_gives_an_item():
    itens = itens_mercado(*_precos(0.002, 4e6), sessoes=1)
    assert len(itens) == 1 and "4,0 vezes" in itens[0].texto


def test_market_items_cover_the_last_sessions_so_a_missed_night_is_not_lost():
    fechamentos, volumes = _precos(0.002, 1e6)
    volumes.iloc[-3] = 5e6
    assert itens_mercado(fechamentos, volumes, sessoes=1) == []
    assert len(itens_mercado(fechamentos, volumes, sessoes=5)) == 1


def test_lending_jump_beyond_3x_the_21_session_median():
    dias = pd.bdate_range("2026-10-01", periods=25)
    saldos = pd.DataFrame({"PETR4": np.r_[np.full(24, 1e6), 4e6]}, index=dias)
    taxas = pd.DataFrame({"PETR4": np.full(25, 0.0006)}, index=dias)
    itens = itens_aluguel(saldos, taxas, sessoes=1)
    assert len(itens) == 1 and itens[0].fonte == "mercado" and "4,0 vezes" in itens[0].texto
    taxas.iloc[-1] = 0.003
    assert len(itens_aluguel(saldos, taxas, sessoes=1)) == 2


def test_macro_item_is_the_latest_release_with_the_previous_one_and_its_id_changes_only_with_a_new_release():
    valores = pd.Series([0.48, 0.82], index=pd.to_datetime(["2026-08-01", "2026-09-01"]))
    serie = Serie("ipca", "bcb", "IPCA mensal (%), SGS 433", valores)
    (item,) = itens_macro([serie])
    assert item.ticker == MACRO and item.fonte == "bcb" and item.data == "2026-09-01"
    assert item.texto == "IPCA mensal (%), SGS 433: 0,82 em 09/2026 (anterior: 0,48 em 08/2026)"
    assert itens_macro([serie])[0].id == item.id
    nova = Serie("ipca", "bcb", serie.rotulo, pd.concat([valores, pd.Series([0.3], [pd.Timestamp("2026-10-01")])]))
    assert itens_macro([nova])[0].id != item.id


def test_daily_macro_series_show_the_day():
    valores = pd.Series([15.0, 14.75], index=pd.to_datetime(["2026-11-03", "2026-11-04"]))
    (item,) = itens_macro([Serie("selic", "bcb", "Selic meta (% a.a.)", valores, mensal=False)])
    assert item.texto == "Selic meta (% a.a.): 14,75 em 04/11/2026 (anterior: 15 em 03/11/2026)"


def test_an_empty_series_gives_no_item():
    assert itens_macro([Serie("x", "ibge", "X", pd.Series(dtype=float))]) == []
