"""
data.py - The frozen inputs of the Inovemetria reading: the CVM accounts of 2026-10-24 plus intangibles, INPI patent
applications dated by first publication, controlled companies from the CVM FRE, CNPJ roots, activity sectors and the
IPCA (``docs/dados/2026-11-12-inovemetria/LEIAME.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[2]
FOLDER = RAIZ / "docs" / "dados" / "2026-11-12-inovemetria"
CVM_FOLDER = RAIZ / "docs" / "dados" / "2026-10-24-cvm"
SECTORS = RAIZ / "docs" / "dados" / "2026-10-22" / "sectors.parquet"
FILES = ("intangibles.parquet", "patents.parquet", "subsidiaries.parquet", "issuer_roots.parquet", "ipca.parquet")


@dataclass(frozen=True)
class InnovationData:
    fundamentals: pd.DataFrame   # issuer, account, dt_refer, value, available_from (+ intangibles)
    patents: pd.DataFrame        # application, cnpj_root, filed_on, known_from, fields
    subsidiaries: pd.DataFrame   # issuer, cnpj_root, dt_refer, known_from
    roots: pd.Series             # issuer -> CNPJ root
    activity: pd.Series          # issuer -> CVM activity sector (setor_ativ)
    ipca: pd.Series              # month start -> index level


def _dates(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return frame.assign(**{c: pd.to_datetime(frame[c]) for c in columns})


def load(folder: Path = FOLDER) -> tuple[InnovationData, dict[str, str]]:
    """The frozen folder (hashes checked against its ``SHA256SUMS``) and the CVM accounts of 2026-10-24."""
    from scripts.avaliar_robustez import conferir_hashes   # heavy module: imported only when the data is loaded
    sha256 = conferir_hashes(folder, FILES) | conferir_hashes(CVM_FOLDER, ["fundamentals.parquet"])
    columns = ["issuer", "account", "dt_refer", "value", "available_from"]
    fundamentals = pd.concat([pd.read_parquet(CVM_FOLDER / "fundamentals.parquet", columns=columns),
                              pd.read_parquet(folder / "intangibles.parquet", columns=columns)], ignore_index=True)
    sectors = pd.read_parquet(SECTORS).dropna(subset=["code_cvm"])
    issuer_of = sectors.drop_duplicates("code_cvm").set_index("code_cvm")["issuer"]
    roots = pd.read_parquet(folder / "issuer_roots.parquet")
    roots = roots.assign(issuer=roots["code_cvm"].map(issuer_of)).dropna(subset=["issuer"])
    subsidiaries = pd.read_parquet(folder / "subsidiaries.parquet")
    subsidiaries = subsidiaries.assign(issuer=subsidiaries["code_cvm"].map(issuer_of)).dropna(subset=["issuer"])
    ipca = pd.read_parquet(folder / "ipca.parquet")
    data = InnovationData(
        fundamentals=_dates(fundamentals, "dt_refer", "available_from"),
        patents=_dates(pd.read_parquet(folder / "patents.parquet"), "filed_on", "known_from"),
        subsidiaries=_dates(subsidiaries[["issuer", "cnpj_root", "dt_refer", "known_from"]], "dt_refer", "known_from"),
        roots=roots.drop_duplicates("issuer", keep="last").set_index("issuer")["cnpj_root"],
        activity=sectors.drop_duplicates("issuer").set_index("issuer")["setor_ativ"],
        ipca=pd.Series((1.0 + ipca["ipca_pct"].to_numpy() / 100.0).cumprod(),
                       index=pd.to_datetime(ipca["month"])))
    return data, sha256


def issuer_patents(data: InnovationData, month: pd.Timestamp) -> pd.DataFrame:
    """issuer, application, known_from, fields: applications published before ``month`` under the issuer's own CNPJ
    root or that of a company its latest FRE received before ``month`` declares controlled; once per issuer."""
    known = data.subsidiaries[data.subsidiaries["known_from"] < month]
    latest = known[known.groupby("issuer")["known_from"].transform("max") == known["known_from"]]
    owners = pd.concat([pd.DataFrame({"issuer": data.roots.index, "cnpj_root": data.roots.to_numpy()}),
                        latest[["issuer", "cnpj_root"]]], ignore_index=True).drop_duplicates()
    public = data.patents[data.patents["known_from"] < month]
    held = owners.merge(public, on="cnpj_root", how="inner")
    return (held.drop_duplicates(["issuer", "application"])[["issuer", "application", "known_from", "fields"]]
            .reset_index(drop=True))
