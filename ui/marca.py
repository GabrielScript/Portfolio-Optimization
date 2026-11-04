"""
marca.py - The app's logo: the user's own eagle (2026-11-04), traced to vector. ``marca/logo.svg`` is the wide
version (symbol + name) for the sidebar; ``marca/icone.svg`` is the symbol alone for the browser tab and the
collapsed sidebar. Either may also be .png, .webp or .jpg. Not ``assets/``: the root's ``assets.py`` already has
that name.
"""

from __future__ import annotations

from pathlib import Path

PASTA_DA_MARCA = Path(__file__).resolve().parents[1] / "marca"
NOMES_DA_LOGO = ("logo.svg", "logo.png", "logo.webp", "logo.jpg")
ICONE_PADRAO = ":material/query_stats:"


def arquivo_da_logo(pasta: Path = PASTA_DA_MARCA) -> Path | None:
    """The first logo file found in ``pasta``, or None."""
    return next((pasta / nome for nome in NOMES_DA_LOGO if (pasta / nome).is_file()), None)


def arquivo_do_icone(pasta: Path = PASTA_DA_MARCA) -> Path | None:
    """The compact mark (``icone.*``) if there is one, else the logo itself."""
    icones = (pasta / nome.replace("logo", "icone") for nome in NOMES_DA_LOGO)
    return next((icone for icone in icones if icone.is_file()), None) or arquivo_da_logo(pasta)
