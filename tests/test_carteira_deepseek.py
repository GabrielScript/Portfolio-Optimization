"""The simulator's DeepSeek portfolio (``scripts/carteira_deepseek.py``, DECISOES 240): the Composite's names the adviser
judges favorable, equal weights, recorded for the day after the opinion."""

from datetime import date

import pandas as pd

from scripts import carteira_deepseek as cd

UNIVERSO = ["PETR4", "VALE3", "ITSA4", "BBSE3"]


def _parecer(posturas: dict[str, str]) -> dict:
    return {"acoes": [{"ticker": ticker, "postura": postura} for ticker, postura in posturas.items()]}


def test_the_favorable_names_share_the_portfolio_equally_and_the_rest_get_zero():
    pesos = cd.pesos_dos_verdes(_parecer({"PETR4": "favoravel", "VALE3": "neutra", "ITSA4": "favoravel",
                                          "BBSE3": "desfavoravel"}), UNIVERSO)
    assert pesos.to_dict() == {"PETR4": 0.5, "VALE3": 0.0, "ITSA4": 0.5, "BBSE3": 0.0}


def test_without_a_favorable_name_every_weight_is_zero_and_a_stock_without_opinion_gets_zero():
    pesos = cd.pesos_dos_verdes(_parecer({"PETR4": "neutra"}), UNIVERSO)
    assert pesos.to_dict() == {ticker: 0.0 for ticker in UNIVERSO}


def test_the_record_is_dated_the_day_after_the_opinion():
    registro = cd.registro_do_parecer(pd.Series({"PETR4": 1.0, "VALE3": 0.0}), date(2026, 11, 3))
    assert list(registro.columns) == ["data", "ticker", "peso"]
    assert set(registro["data"]) == {pd.Timestamp("2026-11-04")} and list(registro["peso"]) == [1.0, 0.0]


def test_the_screen_table_shows_each_composite_name_with_its_light_favorable_first():
    from ui.simulador import tabela_do_parecer
    registro = {"universo": ["PETR4", "VALE3", "ITSA4"], "parecer": {"acoes": [
        {"ticker": "VALE3", "postura": "desfavoravel", "confianca": 0.4},
        {"ticker": "PETR4", "postura": "favoravel", "confianca": 0.7}]}}
    tabela = tabela_do_parecer(registro)
    assert tabela["Ticker"].tolist() == ["PETR4", "ITSA4", "VALE3"]
    assert tabela["Parecer do DeepSeek"].tolist() == ["🟢 compra", "🟡 fora (sem parecer)", "🔴 fora"]
    assert tabela["Confiança"].iloc[[0, 2]].tolist() == [0.7, 0.4] and pd.isna(tabela["Confiança"].iloc[1])


def test_an_opinion_is_due_first_after_a_new_composite_record_and_then_every_28_days():
    composto = pd.DataFrame({"data": pd.to_datetime(["2026-10-27"] * 2), "ticker": ["PETR4", "VALE3"]})
    vazio = pd.DataFrame(columns=["data", "ticker", "peso"])
    assert cd.parecer_devido(composto, vazio, date(2026, 11, 3))
    feito = cd.registro_do_parecer(pd.Series({"PETR4": 1.0}), date(2026, 11, 3))
    assert not cd.parecer_devido(composto, feito, date(2026, 11, 20))
    assert cd.parecer_devido(composto, feito, date(2026, 12, 1))                 # 28 days after 03/11
    novo = pd.concat([composto, pd.DataFrame({"data": [pd.Timestamp("2026-11-10")], "ticker": ["ITSA4"]})])
    assert cd.parecer_devido(novo, feito, date(2026, 11, 11))                    # the Composite rebalanced


def test_a_composite_record_written_after_the_opinion_but_dated_before_it_is_still_judged():
    # code review of 2026-11-03: the opinion of 03/11 judged the record of 27/10; paper-v3 then writes one dated 02/11
    composto = pd.DataFrame({"data": pd.to_datetime(["2026-10-27", "2026-11-02"]), "ticker": ["PETR4", "VALE3"]})
    feito = cd.registro_do_parecer(pd.Series({"PETR4": 1.0}), date(2026, 11, 3))
    assert cd.parecer_devido(composto, feito, date(2026, 11, 4), base_julgada=pd.Timestamp("2026-10-27"))
    assert not cd.parecer_devido(composto, feito, date(2026, 11, 4), base_julgada=pd.Timestamp("2026-11-02"))


def test_the_judged_base_is_read_from_the_newest_opinion_file(tmp_path):
    assert cd.ultima_base_julgada(tmp_path) is None
    (tmp_path / "2026-11-03.json").write_text('{"base_do_composto": "2026-10-27"}', encoding="utf-8")
    (tmp_path / "2026-12-01.json").write_text('{"base_do_composto": "2026-11-28"}', encoding="utf-8")
    assert cd.ultima_base_julgada(tmp_path) == pd.Timestamp("2026-11-28")
