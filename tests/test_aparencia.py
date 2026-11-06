"""The appearance picker (the user's design request of 2026-11-06): royal blue, graphite and silver with chrome shine,
a slow flow and perfect contrast, plus Streamlit's own light and dark. Contrast is held to WCAG 2.1 AA."""

import pandas as pd
import pytest

from tests.test_tema import contraste, tema_nativo
from ui import aparencia


@pytest.mark.parametrize("tipo", ["dark", "light"])
@pytest.mark.parametrize("nome", [aparencia.ROYAL, aparencia.GRAFITE])
def test_every_palette_keeps_the_native_text_readable_on_its_ground_and_panels(nome, tipo):
    # the CSS repaints grounds and panels only; Streamlit still writes the text, so both must agree
    paleta = aparencia.paleta(nome, tipo)
    texto = tema_nativo(tipo)["textColor"]
    for fundo in (*paleta.fundo, paleta.painel):
        assert contraste(texto, fundo) >= 4.5, (fundo, texto)
        assert contraste(paleta.texto_secundario, fundo) >= 4.5, (fundo, paleta.texto_secundario)
        assert contraste(paleta.alta, fundo) >= 4.5 and contraste(paleta.baixa, fundo) >= 4.5, fundo


@pytest.mark.parametrize("tipo", ["dark", "light"])
@pytest.mark.parametrize("nome", [aparencia.ROYAL, aparencia.GRAFITE])
def test_chart_colors_stay_visible_on_every_ground_of_the_palette(nome, tipo):
    paleta = aparencia.paleta(nome, tipo)
    for cor in tema_nativo(tipo)["chartCategoricalColors"]:
        for fundo in (*paleta.fundo, paleta.painel):
            assert contraste(cor, fundo) >= 3.0, (cor, fundo)


def test_royal_is_royal_blue_and_graphite_is_gray():
    def azul_domina(cor):
        r, g, b = (int(cor.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        return b - max(r, g)
    assert all(azul_domina(c) > 20 for c in aparencia.paleta(aparencia.ROYAL, "dark").fundo)
    assert all(abs(azul_domina(c)) < 16 for c in aparencia.paleta(aparencia.GRAFITE, "dark").fundo)


def test_the_streamlit_default_injects_no_css():
    assert aparencia.css(aparencia.PADRAO, "dark") == "" and aparencia.css(aparencia.PADRAO, "light") == ""


def test_the_css_moves_slowly_and_stops_for_reduced_motion():
    css = aparencia.css(aparencia.ROYAL, "dark")
    assert "@keyframes" in css and "prefers-reduced-motion: reduce" in css
    assert "::selection" in css and ":focus-visible" in css and "scrollbar-color" in css
    assert aparencia.css(aparencia.ROYAL, "light") != css       # the light variant has its own ground


def test_an_unknown_or_missing_theme_type_falls_back_to_dark():
    assert aparencia.tipo(None) == "dark" and aparencia.tipo("light") == "light" and aparencia.tipo("x") == "dark"


def test_the_quote_band_shows_each_ticker_price_and_change_in_brazilian_format():
    tabela = pd.DataFrame({"Ticker": ["JHSF3", "UGPA3", "XPTO3"], "Cotação (R$)": [9.91, 29.95, 1234.5],
                           "Variação no dia": [0.0498, -0.0212, float("nan")]})
    html = aparencia.fita(tabela, "dark")
    assert "JHSF3" in html and "R$ 9,91" in html and "+4,98%" in html
    assert "−2,12%" in html                                 # a real minus sign, not a hyphen
    assert "R$ 1.234,50" in html and html.count("XPTO3") == 2     # no change, no badge; the set repeats for the loop
    assert 'aria-hidden="true"' in html                           # the repeated set is hidden from screen readers
    assert aparencia.paleta(aparencia.ROYAL, "dark").alta in html


def test_the_quote_band_escapes_the_ticker():
    tabela = pd.DataFrame({"Ticker": ["<b>"], "Cotação (R$)": [1.0], "Variação no dia": [0.0]})
    assert "<b>" not in aparencia.fita(tabela, "dark")


def test_an_empty_quote_table_draws_no_band():
    assert aparencia.fita(pd.DataFrame(columns=["Ticker", "Cotação (R$)", "Variação no dia"]), "dark") == ""


def test_the_equity_chart_follows_the_theme_with_the_highlight_first_and_a_clear_ground():
    # no hard-coded black: Streamlit's theme gives the colorway (config.toml), the appearance shows through
    from visualizations import grafico_curvas_capital
    dias = pd.bdate_range("2024-01-01", periods=5)
    curvas = {"BOVA11": pd.Series([1, 2, 3, 4, 5.0], index=dias), "Composite": pd.Series([1, 1.1, 1.3, 1.2, 1.5], index=dias)}
    fig = grafico_curvas_capital(curvas, destaque="Composite")
    assert fig.data[0].name == "Composite" and fig.data[0].line.width > fig.data[1].line.width
    assert all(trace.line.color is None for trace in fig.data)
    assert fig.layout.paper_bgcolor == fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    import plotly.graph_objects as go
    assert fig.layout.template == go.Figure().layout.template             # the default: no plotly_dark under it


def test_the_picked_segment_reads_white_on_royal_in_every_appearance():
    # Streamlit draws the picked segment as royal text on a 10% royal tint (about 3:1 on the night grounds)
    assert 'button[aria-checked="true"]' in aparencia.SELETOR_CSS and "#FFFFFF" in aparencia.SELETOR_CSS
    assert contraste("#FFFFFF", "#3A5FD9") >= 4.5


def test_the_top_bar_keeps_a_ground_so_the_page_does_not_scroll_under_the_logo():
    css = aparencia.css(aparencia.ROYAL, "dark")
    barra = css[css.index('[data-testid="stHeader"] {'):]
    assert f"background: {aparencia.paleta(aparencia.ROYAL, 'dark').fundo[0]};" in barra.split("}")[0]   # opaque


def test_every_section_is_a_lit_panel_with_a_silver_top_edge_that_never_moves():
    # the whole page, not just the first viewport (the user's /goal, 2026-11-06): each section a panel on the floor
    css = aparencia.css(aparencia.ROYAL, "dark")
    assert '[class*="st-key-aq-secao-"]' in css and 'st-key-aq-secao-"]::before' not in css   # no stripe
    assert '[data-testid="stHeader"] { z-index: 999990' in css or "z-index: 999990" in css   # the bar stays on top
    assert "translate" not in css.split("@keyframes aq-aurora")[0]        # hover feedback never shifts layout
    assert "max-width: 78ch" in css                                       # readable measure for the prose


def test_the_band_pauses_on_keyboard_focus_too():
    tabela = pd.DataFrame({"Ticker": ["JHSF3"], "Cotação (R$)": [9.91], "Variação no dia": [0.05]})
    html = aparencia.fita(tabela, "dark")
    assert 'tabindex="0"' in html and ":focus-within" in html


def test_the_panel_helper_keys_each_section():
    import streamlit as st
    assert aparencia.chave_da_secao("mercado") == "aq-secao-mercado"
    assert callable(aparencia.painel) and st is not None


def test_native_charts_drop_their_black_ground_and_keep_their_width():
    # Streamlit paints the Vega svg with the native background (a black block on the royal panel, seen 2026-11-06);
    # padding on the chart box pushed the chart past the panel's right edge
    css = aparencia.css(aparencia.ROYAL, "dark")
    assert '[data-testid="stVegaLiteChart"] svg' in css and "background-color: transparent !important" in css
    regras = [r.split("{", 1) for r in css.split("}") if "{" in r]
    assert not [sel for sel, corpo in regras if "VegaLiteChart" in sel and "padding" in corpo]


def test_the_picker_wraps_on_a_phone_instead_of_cutting_the_last_option():
    import inspect
    assert "wrap=True" in inspect.getsource(aparencia.escolher)


def test_finish_review_the_nameplate_is_solid_type_and_the_grid_has_one_alpha():
    # the finish review (2026-11-06): the gradient "chrome" read as a grey fade, not metal; hover on panels taller
    # than the screen signals nothing and broke the one-alpha hairline grid
    for tipo in ("dark", "light"):
        css = aparencia.css(aparencia.ROYAL, tipo)
        assert "background-clip: text" not in css and ":hover" not in css
    escuro = aparencia.paleta(aparencia.ROYAL, "dark").fio
    assert aparencia.paleta(aparencia.GRAFITE, "dark").fio == escuro
    tabela = pd.DataFrame({"Ticker": ["JHSF3"], "Cotação (R$)": [9.91], "Variação no dia": [0.05]})
    assert escuro in aparencia.fita(tabela, "dark")


def test_numbers_follow_one_four_step_magnitude_ramp():
    assert aparencia.RAMPA == ("2rem", "1.25rem", "1rem", "0.8125rem")
    css = aparencia.css(aparencia.ROYAL, "dark")
    assert '[data-testid="stMetricValue"]' in css and "2rem" in css and '[data-testid="stMetricDelta"]' in css
