"""The live accounts' charts and return tables in Brazilian format (the finish review of 2026-11-06): R$ axes that do
not start at zero, dd/mm dates, drawdowns in percent, and a dash where an account is too young for the window."""

import pandas as pd

from ui import graficos


def _curvas() -> pd.DataFrame:
    dias = pd.bdate_range("2026-11-03", periods=4)
    return pd.DataFrame({"Composto": [10000, 10020, 10050, 10067.6], "Selic": [10000, 10004, 10008, 10012]}, index=dias)


def test_the_money_chart_zooms_on_the_curves_with_brazilian_numbers_and_dates():
    spec = graficos.curvas_em_reais(_curvas()).to_dict()
    y, x = spec["encoding"]["y"], spec["encoding"]["x"]
    assert y["scale"]["zero"] is False and "R$" in y["title"]
    assert x["axis"]["format"] == "%d/%m" and x["axis"]["tickCount"] == {"interval": "day", "step": 1}   # one tick a day
    assert spec["config"]["locale"]["number"]["decimal"] == ","
    assert spec["config"]["locale"]["number"]["thousands"] == "."


def test_the_drawdown_chart_reads_in_percent():
    quedas = _curvas() / _curvas().cummax() - 1
    spec = graficos.quedas_em_percentual(quedas).to_dict()
    assert spec["encoding"]["y"]["axis"]["format"] == ".1%"
    assert "(%)" in spec["encoding"]["y"]["title"]


def test_return_tables_show_a_dash_for_a_window_the_account_has_not_lived():
    retornos = pd.DataFrame({"Dia": [0.0046, -0.0109], "Semana": [0.0089, None]}, index=["Composto", "Controle"])
    texto = graficos.percentuais_em_texto(retornos)
    assert texto.loc["Composto", "Dia"] == "0,46%" and texto.loc["Controle", "Dia"] == "\u22121,09%"
    assert texto.loc["Controle", "Semana"] == "\u2014"
    assert "None" not in texto.to_string()


def test_tables_show_fixed_brazilian_decimals_and_keep_numbers_underneath():
    # the second finish review: "localized" dropped trailing zeros ("5,2" beside "5,19") and the quotes kept US decimals
    tabela = pd.DataFrame({"Valor (R$)": [673.28, 728.0, None], "Peso": [0.0669, 0.0, 0.07], "Qtd.": [128, 16, 3]})
    estilo = graficos.estilo_br(tabela, reais=["Valor (R$)"], percentuais=["Peso"])
    html = estilo.to_html()
    assert "673,28" in html and "728,00" in html and "6,69%" in html and "0,00%" in html and "\u2014" in html
    assert estilo.data["Valor (R$)"].iloc[0] == 673.28                    # sorting still sees the number


def test_sector_weights_read_in_percent_with_whole_sector_names():
    pesos = pd.Series({"Petróleo, Gás e Biocombustíveis": 0.2, "Tecnologia da Informação": 0.067, "Bancos": 0.133})
    spec = graficos.barras_por_setor(pesos).to_dict()
    assert spec["encoding"]["x"]["axis"]["format"] == ".0%" and "Peso na Composite" in spec["encoding"]["x"]["title"]
    assert spec["encoding"]["y"]["axis"]["labelLimit"] >= 280 and spec["encoding"]["y"]["title"] is None
