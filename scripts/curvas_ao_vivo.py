"""
curvas_ao_vivo.py - Descriptive rerun of attempt 51's registered loop (`scripts/composto_crescimento_avaliar.py`: the
composite, the Conservadora and BOVA11 in the 3 phases) on the lake's live exports up to the last close, for the Alepha
screen's results by period and by calendar year. Not a reading and not a trial (DECISOES 172 and 188): both
configurations run as registered, N does not change, the plumbing check is off on purpose (the data go past the
published 16/10/2026) and the gate lines in the output mean nothing here. The frozen 23/10 curves stay untouched.

    .venv/Scripts/python -X utf8 scripts/curvas_ao_vivo.py [LAKE_EXPORTS_FOLDER]     (default: data/b3lake)

Copies the exports it reads to a temporary folder with their SHA256SUMS (the frozen loaders check them; the hashes go
to the output), then writes docs/dados/curvas-ao-vivo.parquet (phase, date, Conservadora, Composto, BOVA11, BITH11,
QBTC11, CDI) and docs/curvas-ao-vivo.txt. The two crypto ETFs (the app's closes: lago/precos_diarios.parquet spliced
with lago/precos_recentes.parquet) and the lake's daily CDI ride along for the screen's year table. Commit both files to refresh the screen.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import sys
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

LAGO = RAIZ / "data" / "b3lake"
ARQUIVOS = ("universe_prices.parquet", "universe_monthly.parquet", "sectors.parquet", "monthly_total_returns.parquet",
            "dividends.parquet", "cdi_daily.parquet", "fundamentals.parquet", "market_caps.parquet", "manifest.json")
CRIPTO = ("BITH11", "QBTC11")
CURVAS = RAIZ / "docs" / "dados" / "curvas-ao-vivo.parquet"
SAIDA = RAIZ / "docs" / "curvas-ao-vivo.txt"
COLUNAS = ["fase", "date", "Conservadora", "Composto", "BOVA11"]


def congelar(origem: Path, destino: Path, nomes: Sequence[str]) -> dict[str, str]:
    """Copies ``nomes`` from ``origem`` to ``destino`` with the SHA256SUMS the frozen-input checks read and returns
    the hashes; a missing file stops it (SystemExit) before anything is written."""
    faltando = [nome for nome in nomes if not (origem / nome).exists()]
    if faltando:
        raise SystemExit(f"Missing in {origem}: {', '.join(faltando)}")
    destino.mkdir(parents=True, exist_ok=True)
    somas = {}
    for nome in nomes:
        shutil.copyfile(origem / nome, destino / nome)
        somas[nome] = hashlib.sha256((destino / nome).read_bytes()).hexdigest()
    (destino / "SHA256SUMS").write_text("".join(f"{valor} *{nome}\n" for nome, valor in somas.items()),
                                        encoding="ascii", newline="\n")
    return somas


def quadro_ao_vivo(curvas: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    """Attempt 51's curves file (``fase``, ``date``, one column per curve) with ``extras`` (dates x columns: the
    crypto ETFs' closes and the CDI) on each row's session."""
    return (curvas[COLUNAS].merge(extras.rename_axis("date").reset_index(), on="date", how="left")
            .reset_index(drop=True))


def extras_do_app(cdi: Path) -> pd.DataFrame:
    """The crypto ETFs' daily closes from the app's spliced price files and the lake's CDI (decimal per day)."""
    from core.precos_do_app import ler_fechamentos
    etfs = ler_fechamentos([f"{papel}.SA" for papel in CRIPTO], date(1994, 7, 1), date.today())
    etfs = etfs.rename(columns=lambda papel: papel.removesuffix(".SA"))
    tabela = pd.read_parquet(cdi)
    taxa = pd.Series(tabela["cdi_pct"].to_numpy() / 100.0, index=pd.to_datetime(tabela["date"]), name="CDI")
    return etfs.join(taxa, how="outer").sort_index()


def main(argv: list[str]) -> int:
    from scripts import composto_crescimento_avaliar as tentativa_51
    from scripts import qubo_universo_avaliar

    lago = Path(argv[0]) if argv else LAGO
    with tempfile.TemporaryDirectory(prefix="curvas-ao-vivo-") as temporaria:
        entradas = Path(temporaria)
        somas = congelar(lago, entradas, ARQUIVOS)
        ultima = pd.Timestamp(pd.to_datetime(pd.read_parquet(entradas / "universe_prices.parquet",
                                                             columns=["date"])["date"]).max())
        tentativa_51.FOLDER = tentativa_51.MACRO_FOLDER = qubo_universo_avaliar.FOLDER = entradas
        qubo_universo_avaliar.PASTA_CVM = entradas
        tentativa_51.LAST_SESSION = ultima
        tentativa_51.CURVES_FILE = entradas / "curvas.parquet"
        tentativa_51.check_plumbing = lambda fase, cagr: None      # live data: the published CAGRs no longer apply
        with SAIDA.open("w", encoding="utf-8", newline="\n") as saida, contextlib.redirect_stdout(saida):
            print(f"DESCRIPTIVE live rerun of attempt 51's loop to {ultima:%d/%m/%Y} (not a reading, not a trial; "
                  f"the gate lines below mean nothing here). Lake exports from {lago.name} (sha256): {somas}")
            codigo = tentativa_51.main()
        quadro = quadro_ao_vivo(pd.read_parquet(entradas / "curvas.parquet"),
                                extras_do_app(entradas / "cdi_daily.parquet"))
    quadro.to_parquet(CURVAS, index=False)
    print(f"{CURVAS.relative_to(RAIZ)} up to {ultima:%d/%m/%Y}; output in {SAIDA.relative_to(RAIZ)}")
    return codigo


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
