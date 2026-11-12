"""
Freezes CVM's art. 11 filings (open data ``cia_aberta-doc-vlmo``, yearly zips) as one row per filing, with what the
reading uses and nothing return-based (``docs/superpowers/specs/2026-11-12-informados-design.md``):

    python -X utf8 scripts/informados_congelar.py D:/inovemetria_raw/vlmo
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.informados import FILE, FOLDER, read_filings  # noqa: E402


def main(origem: Path) -> int:
    filings = read_filings(origem)
    FOLDER.mkdir(parents=True, exist_ok=True)
    filings.to_parquet(FOLDER / FILE, index=False)
    digest = hashlib.sha256((FOLDER / FILE).read_bytes()).hexdigest()
    (FOLDER / "SHA256SUMS").write_text(f"{digest}  {FILE}\n", encoding="utf-8", newline="\n")
    com_negocio = filings[(filings["bought"] + filings["sold"]) > 0]
    print(f"{len(filings)} filings, {filings['cnpj_root'].nunique()} companies, reference "
          f"{filings['reference'].min():%Y-%m} to {filings['reference'].max():%Y-%m}; {len(com_negocio)} with a cash "
          f"trade by a reporting person; sha256 {digest}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: informados_congelar.py <folder with vlmo_<year>.zip>")
    raise SystemExit(main(Path(sys.argv[1])))
