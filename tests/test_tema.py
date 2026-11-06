"""The app's native theme (the user's design request of 2026-11-06: royal blue, gray and silver, with Streamlit's own
light and dark kept as the third choice), checked against WCAG 2.1 contrast so the colors stay readable."""

import tomllib
from pathlib import Path

import pytest

from ui.marca import arquivo_da_logo, arquivo_do_icone

CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def _luminancia(cor: str) -> float:
    canais = [int(cor.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lineares = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in canais]
    return 0.2126 * lineares[0] + 0.7152 * lineares[1] + 0.0722 * lineares[2]


def contraste(a: str, b: str) -> float:
    claro, escuro = sorted((_luminancia(a), _luminancia(b)), reverse=True)
    return (claro + 0.05) / (escuro + 0.05)


def tema_nativo(tipo: str) -> dict:
    """``[theme]`` with ``[theme.light]`` or ``[theme.dark]`` over it, the way Streamlit resolves them."""
    tema = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["theme"]
    return {**{k: v for k, v in tema.items() if not isinstance(v, dict)}, **tema[tipo]}


def test_the_wcag_contrast_formula():
    assert contraste("#FFFFFF", "#000000") == pytest.approx(21.0)
    assert contraste("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)


def test_both_streamlit_defaults_stay_available_through_the_settings_toggle():
    # [theme.light] and [theme.dark] together turn on Streamlit's light/dark switch (menu > Settings)
    claro, escuro = tema_nativo("light"), tema_nativo("dark")
    assert claro["backgroundColor"].upper() == "#FFFFFF" and escuro["backgroundColor"].upper() == "#0E1117"
    assert claro["primaryColor"].upper() == escuro["primaryColor"].upper() == "#3A5FD9"     # royal blue actions
    assert "Manrope" in claro["headingFont"] and "Inter" in claro["font"]


@pytest.mark.parametrize("tipo", ["light", "dark"])
def test_text_links_and_buttons_meet_wcag_aa(tipo):
    tema = tema_nativo(tipo)
    fundo, painel = tema["backgroundColor"], tema["secondaryBackgroundColor"]
    for texto in (tema["textColor"], tema["linkColor"]):
        assert min(contraste(texto, fundo), contraste(texto, painel)) >= 4.5, texto
    assert contraste("#FFFFFF", tema["primaryColor"]) >= 4.5                  # Streamlit writes button text white
    assert contraste(tema["primaryColor"], fundo) >= 3.0                      # sliders, toggles, focus rings
    for cor in tema["chartCategoricalColors"]:
        assert contraste(cor, fundo) >= 3.0, cor                              # chart lines against the background


def test_the_logo_shows_only_once_the_user_adds_the_file(tmp_path):
    # the user makes the logo (2026-11-04); until a file is there the app shows none
    assert arquivo_da_logo(tmp_path) is None
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    assert arquivo_da_logo(tmp_path) == tmp_path / "logo.png"


def test_the_icon_is_the_compact_mark_when_there_is_one_else_the_logo(tmp_path):
    # the full logo (symbol + name) is too wide for the tab and the collapsed sidebar: a separate icon file wins there
    (tmp_path / "logo.svg").write_text("<svg/>")
    assert arquivo_do_icone(tmp_path) == tmp_path / "logo.svg"
    (tmp_path / "icone.svg").write_text("<svg/>")
    assert arquivo_do_icone(tmp_path) == tmp_path / "icone.svg"


def test_the_shipped_logo_and_icon_are_vector_files_with_the_gold_gradient():
    for arquivo in (arquivo_da_logo(), arquivo_do_icone()):
        texto = arquivo.read_text(encoding="utf-8")
        assert arquivo.suffix == ".svg" and 'url(#gold)' in texto and "<image" not in texto   # no embedded bitmap


def test_the_icon_is_square_so_the_tab_and_the_collapsed_sidebar_show_it_full_size():
    import re
    caixa = re.search(r'viewBox="([^"]+)"', arquivo_do_icone().read_text(encoding="utf-8")).group(1).split()
    assert float(caixa[2]) == float(caixa[3])


def test_no_file_of_the_app_still_carries_the_old_name():
    # the user renamed the app "Aleph Quant" on 2026-11-04 and asked that no "Neumann" stay; DECISOES.md is the
    # history log and keeps the old entries as written
    import subprocess
    achados = subprocess.run(["git", "grep", "-l", "-i", "neumann", "--", ".", ":!DECISOES.md",
                              ":!tests/test_tema.py", ":!tests/test_app.py"],
                             capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.split()
    assert achados == []


@pytest.mark.parametrize("tipo", ["light", "dark"])
def test_ten_chart_colors_so_plotly_keeps_the_theme(tipo):
    # Streamlit's Plotly template has ten color slots: with fewer, the rest arrive as "undefined" and Plotly drops the
    # whole colorway for its own default (seen on the page, 2026-11-06)
    assert len(tema_nativo(tipo)["chartCategoricalColors"]) >= 10
