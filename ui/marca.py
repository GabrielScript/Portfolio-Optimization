"""
marca.py - The app's logo: the user makes it (2026-11-04) and drops it in ``marca/`` as logo.svg, .png, .webp or
.jpg; until then the app shows none. Not ``assets/``: the root's ``assets.py`` module already has that name.
"""

from __future__ import annotations

from pathlib import Path

PASTA_DA_MARCA = Path(__file__).resolve().parents[1] / "marca"
NOMES_DA_LOGO = ("logo.svg", "logo.png", "logo.webp", "logo.jpg")
ICONE_PADRAO = ":material/query_stats:"


def arquivo_da_logo(pasta: Path = PASTA_DA_MARCA) -> Path | None:
    """The first logo file found in ``pasta``, or None."""
    return next((pasta / nome for nome in NOMES_DA_LOGO if (pasta / nome).is_file()), None)
