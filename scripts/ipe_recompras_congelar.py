"""
ipe_recompras_congelar.py - Freezes the buyback rows of the CVM IPE index for the SUE/buyback reading
(docs/superpowers/specs/2026-11-08-pead-recompra-design.md, sections 2 and 4). Reads the yearly IPE files 2010-2026
from ``../../cvm/``, keeps the rows ``core.recompra.e_recompra`` accepts, attaches the issuer of each CVM code from the
frozen fundamentals file, and writes ``docs/dados/2026-11-08-ipe/recompras.parquet`` with its hash and the source files'
hashes. Counts only: no price or return is read.

    python -X utf8 scripts/ipe_recompras_congelar.py
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from core.recompra import e_recompra  # noqa: E402

ORIGEM = RAIZ.parent.parent / "cvm"
DESTINO = RAIZ / "docs" / "dados" / "2026-11-08-ipe"
FUNDAMENTOS = RAIZ / "docs" / "dados" / "2026-10-24-cvm" / "fundamentals.parquet"
ANOS = range(2010, 2027)
COLUNAS = ["Codigo_CVM", "Categoria", "Tipo", "Especie", "Assunto", "Data_Referencia", "Data_Entrega", "Versao",
           "Tipo_Apresentacao", "Protocolo_Entrega"]


def _sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def ler_ipe(arquivo: Path) -> pd.DataFrame:
    with zipfile.ZipFile(arquivo) as pacote:
        with pacote.open(pacote.namelist()[0]) as csv:
            return pd.read_csv(csv, sep=";", encoding="latin-1", dtype=str, usecols=COLUNAS)


def main() -> int:
    arquivos = [ORIGEM / f"ipe_cia_aberta_{ano}.zip" for ano in ANOS]
    todas = pd.concat([ler_ipe(arquivo) for arquivo in arquivos], ignore_index=True)
    aceitas = todas[[e_recompra(*linha) for linha in
                     todas[["Categoria", "Tipo", "Especie", "Assunto"]].itertuples(index=False, name=None)]].copy()
    emissores = (pd.read_parquet(FUNDAMENTOS, columns=["code_cvm", "issuer"]).drop_duplicates()
                 .astype({"code_cvm": str}))
    # a renamed company keeps one CVM code and two issuer roots (spec addendum 1): the row goes to each
    aceitas = aceitas.merge(emissores.rename(columns={"code_cvm": "Codigo_CVM"}), on="Codigo_CVM", how="left")
    for coluna in ("Data_Referencia", "Data_Entrega"):
        aceitas[coluna] = pd.to_datetime(aceitas[coluna], errors="coerce")
    DESTINO.mkdir(parents=True, exist_ok=True)
    saida = DESTINO / "recompras.parquet"
    aceitas.reset_index(drop=True).to_parquet(saida, index=False)
    linhas = [f"{_sha256(saida)} *{saida.name}"]
    linhas += [f"{_sha256(arquivo)} *../../../../../cvm/{arquivo.name}" for arquivo in arquivos]
    (DESTINO / "SHA256SUMS").write_bytes(("\n".join(linhas) + "\n").encode("ascii"))
    sem_emissor = aceitas["issuer"].isna()
    datas_ruins = aceitas["Data_Referencia"].isna() | aceitas["Data_Entrega"].isna()
    resumo = (f"IPE rows read {len(todas):,}; buyback rows x issuers {len(aceitas):,}; without an issuer in the fundamentals "
              f"file {int(sem_emissor.sum()):,} ({aceitas.loc[sem_emissor, 'Codigo_CVM'].nunique()} CVM codes); "
              f"with an unreadable date {int(datas_ruins.sum()):,}")
    print(resumo)
    print(aceitas.loc[~sem_emissor].groupby(aceitas["Data_Referencia"].dt.year).size().to_string())
    (DESTINO / "LEIAME.md").write_bytes((
        "# CVM IPE buyback rows, frozen 2026-11-08\n\n"
        "Built by `scripts/ipe_recompras_congelar.py` from the CVM IPE yearly index files 2010-2026 "
        "(dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE, kept in `../../cvm/`), rows accepted by "
        "`core.recompra.e_recompra` (spec `docs/superpowers/specs/2026-11-08-pead-recompra-design.md`, section 4). "
        "`issuer` = each issuer of the row's CVM code in `docs/dados/2026-10-24-cvm/fundamentals.parquet` (one row per "
        "issuer, spec addendum 1; empty when the code has none). The index keeps one row per document version.\n\n" + resumo + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
