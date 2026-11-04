"""The app's native theme (the user's design request of 2026-11-04: premium black, stylized gold and royal blue),
checked against WCAG 2.1 contrast so the colors stay readable."""

import tomllib
from pathlib import Path

import pytest

from ui.marca import arquivo_da_logo

CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def _luminancia(cor: str) -> float:
    canais = [int(cor.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lineares = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in canais]
    return 0.2126 * lineares[0] + 0.7152 * lineares[1] + 0.0722 * lineares[2]


def contraste(a: str, b: str) -> float:
    claro, escuro = sorted((_luminancia(a), _luminancia(b)), reverse=True)
    return (claro + 0.05) / (escuro + 0.05)


@pytest.fixture(scope="module")
def tema() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))["theme"]


def test_the_wcag_contrast_formula():
    assert contraste("#FFFFFF", "#000000") == pytest.approx(21.0)
    assert contraste("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)


def test_the_theme_is_premium_black_with_gold_and_royal_blue(tema):
    assert tema["base"] == "dark"
    assert _luminancia(tema["backgroundColor"]) < 0.01                       # near black
    assert tema["linkColor"] == tema["chartCategoricalColors"][0]             # the gold leads links and charts
    assert tema["primaryColor"].upper() == "#3A5FD9"                          # royal blue for the actions


def test_text_links_and_buttons_meet_wcag_aa(tema):
    fundo, painel = tema["backgroundColor"], tema["secondaryBackgroundColor"]
    for texto in (tema["textColor"], tema["linkColor"], tema["dataframeHeaderTextColor"]):
        assert min(contraste(texto, fundo), contraste(texto, painel)) >= 4.5, texto
    assert contraste("#FFFFFF", tema["primaryColor"]) >= 4.5                  # Streamlit writes button text white
    assert contraste(tema["primaryColor"], fundo) >= 3.0                      # sliders, toggles, focus rings
    for cor in tema["chartCategoricalColors"]:
        assert contraste(cor, fundo) >= 3.0, cor                              # chart lines against the background
    barra = {**tema, **tema.get("sidebar", {})}
    assert contraste(barra["textColor"], barra["backgroundColor"]) >= 4.5


def test_the_logo_shows_only_once_the_user_adds_the_file(tmp_path):
    # the user makes the logo (2026-11-04); until a file is there the app shows none
    assert arquivo_da_logo(tmp_path) is None
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    assert arquivo_da_logo(tmp_path) == tmp_path / "logo.png"
