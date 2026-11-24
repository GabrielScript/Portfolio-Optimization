"""
debentures.py - The company's own debentures as a credit signal on the Composite
(docs/superpowers/specs/2026-11-24-debentures-design.md, committed before any return-based number).

The SND files (debentures.com.br, ANBIMA) stay on ``D:/b3-datalake/raw/snd/`` and are never committed (DECISOES 110);
their hashes are in ``docs/dados/2026-11-24-snd/SHA256SUMS``. Research only: the live Composite is not changed.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from core.risco import emissor
from core.valor import NotaDoMes

RAIZ = Path(__file__).resolve().parents[1]
RAW = Path("D:/b3-datalake/raw/snd")
SUMS = RAIZ / "docs" / "dados" / "2026-11-24-snd" / "SHA256SUMS"
TRADES, CHARACTERISTICS = "snd_trades.parquet", "caracteristicas/caracteristicas.txt"
HEADER_PREFIX = "Codigo do Ativo"
DI = "DI"
LAG_CHANGE = 3                  # CRD-V: M - 1 minus M - 4
KINDS = ("level", "change")
BOUNDS = (50.0, 150.0)          # "% PU da Curva" outside: misprints and defaulted papers trading on recovery
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


def monthly_measure(trades: pd.DataFrame, characteristics: pd.DataFrame) -> pd.Series:
    """(month start, CNPJ root) -> the mean "% PU da Curva" of the issuer's own DI-linked debentures traded in the
    month, weighted by quantity x mean unit price; rows with the percentage missing or outside ``BOUNDS`` dropped."""
    di = characteristics.loc[characteristics["indice"] == DI, ["cod_ativo", "cnpj_root"]].drop_duplicates("cod_ativo")
    linhas = trades.assign(cod_ativo=trades["cod_ativo"].str.strip()).merge(di, on="cod_ativo", how="inner")
    linhas = linhas[linhas["pct_pu_curva"].between(*BOUNDS)]
    peso = linhas["quantidade"] * linhas["pu_med"]
    linhas = linhas.assign(peso=peso, produto=peso * linhas["pct_pu_curva"],
                           month=linhas["data"].dt.to_period("M").dt.to_timestamp())
    somas = linhas.groupby(["month", "cnpj_root"])[["produto", "peso"]].sum()
    somas = somas[somas["peso"] > 0]
    return (somas["produto"] / somas["peso"]).rename(None)


def _of_month(measure: pd.Series, month: pd.Timestamp) -> pd.Series:
    """CNPJ root -> the measure of ``month`` (empty when the month has none)."""
    try:
        return measure.xs(month, level=0)
    except KeyError:
        return pd.Series(dtype=float)


def credit_score(measure: pd.Series, roots: pd.Series, kind: str) -> NotaDoMes:
    """The score of decision month M for the candidates, from trades through the last session of M - 1:
    ``"level"`` (CRD-N) the measure of M - 1; ``"change"`` (CRD-V) the measure of M - 1 minus that of M - 4, missing
    if either is. A higher value is healthier credit. A candidate whose issuer has no
    CNPJ root, or whose root has no measure, is absent."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")

    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        anterior = mes - pd.DateOffset(months=1)
        valores = _of_month(measure, anterior)
        if kind == "change":
            valores = valores.sub(_of_month(measure, anterior - pd.DateOffset(months=LAG_CHANGE))).dropna()
        raiz = pd.Series({papel: roots.get(emissor(papel)) for papel in candidatos}, dtype=object).dropna()
        return raiz.map(valores).dropna().astype(float)
    return _nota
