"""
debentures.py - The company's own debentures as a credit signal on the Composite
(docs/superpowers/specs/2026-11-24-debentures-design.md, committed before any return-based number).

The SND files (debentures.com.br, ANBIMA) stay on ``D:/b3-datalake/raw/snd/`` and are never committed (DECISOES 110);
their hashes are in ``docs/dados/2026-11-24-snd/SHA256SUMS``. Research only: the live Composite is not changed.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = Path("D:/b3-datalake/raw/snd")
SUMS = RAIZ / "docs" / "dados" / "2026-11-24-snd" / "SHA256SUMS"
TRADES, CHARACTERISTICS = "snd_trades.parquet", "caracteristicas/caracteristicas.txt"
HEADER_PREFIX = "Codigo do Ativo"
COLUMNS = {"Codigo do Ativo": "cod_ativo", "indice": "indice", "CNPJ": "cnpj"}


def read_characteristics(text: str) -> pd.DataFrame:
    """The SND characteristics export as ``cod_ativo``, ``cnpj_root`` (first 8 CNPJ digits) and ``indice``, columns
    taken by name from the header line (the export opens with a free-text preamble)."""
    linhas = text.splitlines()
    inicio = next(i for i, linha in enumerate(linhas) if linha.startswith(HEADER_PREFIX))
    tabela = pd.read_csv(io.StringIO("\n".join(linhas[inicio:])), sep="\t", dtype=str, keep_default_na=False)
    tabela.columns = [coluna.strip() for coluna in tabela.columns]
    tabela = tabela[list(COLUMNS)].rename(columns=COLUMNS).apply(lambda coluna: coluna.str.strip())
    raiz = tabela["cnpj"].str.replace(r"\D", "", regex=True).str[:8]
    return pd.DataFrame({"cod_ativo": tabela["cod_ativo"], "cnpj_root": raiz, "indice": tabela["indice"]})


def load_snd(raw: Path = RAW, sums: Path = SUMS) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """The SND trades (``data`` as a date) and characteristics on ``raw``, each file's sha256 checked against
    ``sums`` (names relative to ``raw``); stops on a mismatch."""
    esperados = {}
    for linha in Path(sums).read_text(encoding="ascii").splitlines():
        valor, nome = linha.split(maxsplit=1)
        esperados[nome.lstrip("*")] = valor
    obtidos = {nome: hashlib.sha256((Path(raw) / nome).read_bytes()).hexdigest() for nome in esperados}
    errados = [nome for nome in esperados if obtidos[nome] != esperados[nome]]
    if errados:
        raise SystemExit(f"SND files changed since the pre-registration: {errados}")
    trades = pd.read_parquet(Path(raw) / TRADES)
    trades["data"] = pd.to_datetime(trades["data"])
    caracteristicas = read_characteristics((Path(raw) / CHARACTERISTICS).read_text(encoding="utf-8"))
    return trades, caracteristicas, obtidos


def issuer_roots(code_roots: pd.DataFrame, sectors: pd.DataFrame) -> pd.Series:
    """issuer -> CNPJ root: CVM code -> root (CVM registry) joined to the lake's issuer -> CVM code."""
    codigos = sectors.dropna(subset=["code_cvm"]).astype({"code_cvm": "int64"}).drop_duplicates("code_cvm")
    juntos = codigos.merge(code_roots.astype({"code_cvm": "int64"}), on="code_cvm", how="inner")
    return juntos.drop_duplicates("issuer", keep="last").set_index("issuer")["cnpj_root"].rename(None)
