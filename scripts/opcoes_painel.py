"""
Builds the frozen option-signal panel of `docs/superpowers/specs/2026-10-29-opcoes-veto-design.md` (pre-registered
before any signal existed) from the lake's `option_signals_monthly.parquet` and the frozen universe. Run once:

    python -X utf8 scripts/opcoes_painel.py

Keeps only chains of issuers in the month's universe (so ETFs, BDRs and index options are out), takes each issuer's
most traded chain for iv_atm and skew and every chain for pc_ratio (`core.opcoes.sinais_por_emissor`), and writes
docs/dados/2026-11-05-opcoes/option_panel.parquet with SHA256SUMS (the lake export's sha256 too). Prints only
coverage, never a return or an IC.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.opcoes import EMISSORES_MINIMOS, SINAIS, sinais_por_emissor  # noqa: E402
from core.risco import emissor  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

EXPORT = Path(os.environ.get("LAKE_DIR", "D:/b3-datalake")) / "exports" / "option_signals_monthly.parquet"
SAIDA = FOLDER.parent / "2026-11-05-opcoes"
PAINEL = SAIDA / "option_panel.parquet"
PERIODO = (pd.Timestamp("2008-01-01"), pd.Timestamp("2026-09-01"))


def painel_no_universo(cadeias: pd.DataFrame, universo: pd.DataFrame) -> pd.DataFrame:
    """The issuer panel built only from chains whose issuer is in that month's universe."""
    dentro = (universo.assign(month=pd.to_datetime(universo["month"]), issuer=universo["security"].map(emissor))
              [["month", "issuer"]].drop_duplicates())
    cadeias = cadeias.assign(month=pd.to_datetime(cadeias["month"]))
    cadeias = cadeias.merge(dentro, left_on=["month", "underlying_issuer"], right_on=["month", "issuer"])
    return sinais_por_emissor(cadeias.drop(columns="issuer"))


def main() -> int:
    inicio = time.perf_counter()
    hash_export = hashlib.sha256(EXPORT.read_bytes()).hexdigest()
    sha256 = {EXPORT.name: hash_export} | conferir_hashes(FOLDER, ["universe_monthly.parquet"])
    print(f"Files (sha256): {sha256}")
    cadeias = pd.read_parquet(EXPORT)
    universo = pd.read_parquet(FOLDER / "universe_monthly.parquet")
    painel = painel_no_universo(cadeias, universo)
    painel = painel[painel["month"].between(*PERIODO)].reset_index(drop=True)

    SAIDA.mkdir(parents=True, exist_ok=True)
    painel.to_parquet(PAINEL, index=False)
    hash_painel = hashlib.sha256(PAINEL.read_bytes()).hexdigest()
    (SAIDA / "SHA256SUMS").write_text(f"{hash_painel} *{PAINEL.name}\n", encoding="ascii", newline="\n")
    (SAIDA / "LAKE_EXPORT_SHA256").write_text(f"{hash_export} *{EXPORT.name}\n", encoding="ascii", newline="\n")
    print(f"Panel {len(painel):,} issuer-months, sha256 {hash_painel}")
    for nome, coluna in SINAIS.items():
        por_mes = painel.dropna(subset=[coluna]).groupby("month").size()
        lidos = por_mes[por_mes >= EMISSORES_MINIMOS]
        print(f"{nome} ({coluna}): issuers per month median {por_mes.median():.0f} (max {por_mes.max()}); months with "
              f">= {EMISSORES_MINIMOS}: {len(lidos)}, first {lidos.index.min():%Y-%m}" if len(lidos) else
              f"{nome} ({coluna}): no month with {EMISSORES_MINIMOS} issuers")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
