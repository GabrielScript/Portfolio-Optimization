"""
Freeze the lake's innovation exports for the Inovemetria reading (``docs/superpowers/specs/2026-11-12-inovemetria-design.md``):

    python -X utf8 scripts/inovemetria_montar.py

Copies `D:/b3-datalake/exports/innovation/` into `docs/dados/2026-11-12-inovemetria/`, keeps the intangibles received
up to 2026-10-24 (the CVM folder of the other accounts) with the issuer and sector of the frozen sectors file, and
writes `SHA256SUMS` and `LEIAME.md`. Run once; the readings check the hashes.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ORIGEM = Path("D:/b3-datalake/exports/innovation")
DESTINO = RAIZ / "docs" / "dados" / "2026-11-12-inovemetria"
SETORES = RAIZ / "docs" / "dados" / "2026-10-22" / "sectors.parquet"
CORTE = pd.Timestamp("2026-10-24")
COPIADOS = ("patents.parquet", "subsidiaries.parquet", "issuer_roots.parquet", "ipca.parquet")


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nome in COPIADOS:
        shutil.copyfile(ORIGEM / nome, DESTINO / nome)
    setores = pd.read_parquet(SETORES)[["issuer", "code_cvm", "sector"]].dropna(subset=["code_cvm"])
    intangiveis = pd.read_parquet(ORIGEM / "intangibles.parquet")
    intangiveis = intangiveis[pd.to_datetime(intangiveis["available_from"]) <= CORTE]
    intangiveis = intangiveis.merge(setores, on="code_cvm", how="inner")
    intangiveis.to_parquet(DESTINO / "intangibles.parquet", index=False)
    nomes = sorted(p.name for p in DESTINO.glob("*.parquet"))
    linhas = [f"{hashlib.sha256((DESTINO / n).read_bytes()).hexdigest()}  {n}" for n in nomes]
    (DESTINO / "SHA256SUMS").write_text("\n".join(linhas) + "\n", encoding="ascii")
    contagens = {n: len(pd.read_parquet(DESTINO / n)) for n in nomes}
    (DESTINO / "LEIAME.md").write_text(_leiame(contagens), encoding="utf-8")
    print(contagens)
    return 0


def _leiame(contagens: dict[str, int]) -> str:
    linhas = "\n".join(f"- `{nome}`: {n:,} rows" for nome, n in contagens.items())
    return f"""# Innovation data of 2026-11-12 (Inovemetria)

Inputs of the Inovemetria reading (`docs/superpowers/specs/2026-11-12-inovemetria-design.md`), built by the lake's
`b3lake innovation` command (branch `feat/innovation-data`) on 12/11/2026 and frozen by `scripts/inovemetria_montar.py`.

{linhas}

- `intangibles.parquet`: CVM account 1.02.04.01 (intangibles without goodwill), consolidated first, each value
  dated by its document's receipt (`available_from`), only rows received up to 2026-10-24, with issuer and sector
  of `docs/dados/2026-10-22/sectors.parquet`.
- `patents.parquet`: INPI BADEPI v12.0 (snapshot 13/10/2026, dispatches to 30/12/2025) applications of listed
  companies and of companies they declared controlled; `known_from` = first publication (RPI 3.1, 1.1 or 1.3).
- `subsidiaries.parquet`: CVM FRE `participacao_sociedade`, "Controlada" or >= 50%, dated by the FRE receipt.
- `issuer_roots.parquet`: CVM code -> CNPJ root (CVM registry). `ipca.parquet`: BCB SGS 433, monthly %.

Known limits: one BADEPI snapshot (today's holder and today's WIPO fields, back-dated); holding companies inherit
their subsidiaries' patents (Itaúsa, Bradespar); goodwill filed inside 1.02.04.01 and concession assets count as
intangibles.
"""


if __name__ == "__main__":
    sys.exit(main())
