"""
informados.py - Insider dealing as a score, the weight a new score earns beside the Composite, and the mean of
strategies (docs/superpowers/specs/2026-11-12-informados-design.md).

The insider data are CVM's art. 11 filings ("Valores Mobiliários Negociados e Detidos", open data
``cia_aberta-doc-vlmo``): every month each listed company reports what its officers, directors, fiscal council,
controller and those tied to them bought and sold. A filing is known on its delivery date, about ten days into the
next month, and can be refiled.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from core.backtest import Estrategia
from core.risco import emissor
from core.valor import NotaDoMes
from core.views import escore_normal

REFERENCE_MONTHS = 6             # reference months behind a score
MIN_MONTHS = 36                  # months of IC before a score earns any weight
OWN_COMPANY = "Companhia"
ASSETS = ("Ações", "Units")
BOUGHT, SOLD = "Compra à vista", "Venda à vista"
KEYS = ["cnpj_root", "reference", "version"]
FOLDER = Path(__file__).resolve().parents[1] / "docs" / "dados" / "2026-11-12-vlmo"
FILE = "filings.parquet"


def load_filings(folder: Path = FOLDER) -> tuple[pd.DataFrame, dict[str, str]]:
    """The frozen filings (hash checked against the folder's ``SHA256SUMS``)."""
    from scripts.avaliar_robustez import conferir_hashes   # heavy module: imported only when the data is loaded
    sha256 = conferir_hashes(folder, [FILE])
    return pd.read_parquet(folder / FILE), sha256


def _root(cnpj: pd.Series) -> pd.Series:
    return cnpj.str.replace(r"\D", "", regex=True).str[:8]


def _csv(arquivo: zipfile.ZipFile, nome: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(arquivo.read(nome)), sep=";", encoding="latin1", dtype=str)


def read_filings(folder: Path) -> pd.DataFrame:
    """CVM's yearly zips in ``folder`` as one row per filing (company, reference month, version): its delivery date
    and the BRL bought and sold on the cash market, in the company's own shares or units, by the persons and bodies
    that report (the rows of the company's own treasury carry no ``Tipo_Cargo`` and stay out). A filing with no such
    trade is kept with zeros: it can replace an earlier version that had one."""
    movimentos, documentos = [], []
    for caminho in sorted(Path(folder).glob("vlmo_*.zip")):
        with zipfile.ZipFile(caminho) as arquivo:
            for nome in arquivo.namelist():
                (movimentos if "_con_" in nome else documentos).append(_csv(arquivo, nome))
    documentos = pd.concat(documentos, ignore_index=True)
    filings = pd.DataFrame({"cnpj_root": _root(documentos["CNPJ_Companhia"]),
                            "reference": pd.to_datetime(documentos["Data_Referencia"]),
                            "version": documentos["Versao"].astype(int),
                            "delivered": pd.to_datetime(documentos["Data_Entrega"])})
    filings = filings.dropna().sort_values("delivered").drop_duplicates(KEYS, keep="first")
    m = pd.concat(movimentos, ignore_index=True)
    m = m[(m["Tipo_Empresa"] == OWN_COMPANY) & m["Tipo_Ativo"].isin(ASSETS) & m["Tipo_Cargo"].notna()
          & (m["Tipo_Cargo"].str.strip() != "") & m["Tipo_Movimentacao"].isin((BOUGHT, SOLD))]
    numero = {coluna: pd.to_numeric(m[coluna], errors="coerce") for coluna in ("Volume", "Quantidade", "Preco_Unitario")}
    volume = numero["Volume"].fillna(numero["Quantidade"] * numero["Preco_Unitario"]).abs()
    trades = pd.DataFrame({"cnpj_root": _root(m["CNPJ_Companhia"]), "reference": pd.to_datetime(m["Data_Referencia"]),
                           "version": m["Versao"].astype(int),
                           "bought": volume.where(m["Tipo_Movimentacao"] == BOUGHT, 0.0),
                           "sold": volume.where(m["Tipo_Movimentacao"] == SOLD, 0.0)})
    somas = trades.groupby(KEYS, as_index=False)[["bought", "sold"]].sum()
    return filings.merge(somas, on=KEYS, how="left").fillna({"bought": 0.0, "sold": 0.0}).reset_index(drop=True)


def net_purchase(filings: pd.DataFrame, month: pd.Timestamp, months: int = REFERENCE_MONTHS) -> pd.Series:
    """(bought - sold) / (bought + sold) per CNPJ root over the ``months`` reference months before ``month``, from
    the filings delivered strictly before ``month`` (the latest such version of each reference month). A company
    with no trade in the window is absent."""
    inicio = month - pd.DateOffset(months=months)
    conhecidos = filings[(filings["delivered"] < month) & (filings["reference"] >= inicio)
                         & (filings["reference"] < month)]
    ultimos = conhecidos.sort_values("version").drop_duplicates(["cnpj_root", "reference"], keep="last")
    somas = ultimos.groupby("cnpj_root")[["bought", "sold"]].sum()
    total = somas["bought"] + somas["sold"]
    return ((somas["bought"] - somas["sold"]) / total)[total > 0]


def insider_score(filings: pd.DataFrame, roots: pd.Series) -> NotaDoMes:
    """The score of the month for the candidates: the issuer's net purchase ratio (``roots``: issuer -> CNPJ root)."""
    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        razao = net_purchase(filings, mes)
        raiz = pd.Series({papel: roots.get(emissor(papel)) for papel in candidatos}, dtype=object).dropna()
        return raiz.map(razao).dropna().astype(float)
    return _nota


def earned_weight(ic_extra: pd.Series, ic_base: pd.Series, month: pd.Timestamp, min_months: int = MIN_MONTHS) -> float:
    """clip(mean IC of the extra score / mean IC of the base score, 0, 1) over the months before ``month`` that have
    both ICs (a month's IC is known when the month ends); 0 with fewer than ``min_months`` of them or a base that
    has not earned anything."""
    juntos = pd.DataFrame({"extra": ic_extra, "base": ic_base}).dropna()
    passado = juntos[juntos.index < month]
    if len(passado) < min_months or passado["base"].mean() <= 0:
        return 0.0
    return float(min(max(passado["extra"].mean() / passado["base"].mean(), 0.0), 1.0))


def weighted_score(base: NotaDoMes, extra: NotaDoMes, weight_of_month: Callable[[pd.Timestamp], float]) -> NotaDoMes:
    """z(base) + weight x z(extra) over the names with a base score, z the house normal score among them; a name
    without the extra score adds nothing."""
    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        principal = escore_normal(base(mes, candidatos).dropna())
        peso = weight_of_month(mes)
        if peso == 0.0 or principal.empty:
            return principal
        adicional = escore_normal(extra(mes, list(principal.index)).reindex(principal.index).dropna())
        return principal.add(peso * adicional.reindex(principal.index).fillna(0.0))
    return _nota


def mean_of(strategies: Sequence[Estrategia]) -> Estrategia:
    """The strategy that holds the mean of the strategies' books."""
    if not strategies:
        raise ValueError("at least one strategy is required")

    def _estrategia(treino: pd.DataFrame) -> pd.Series:
        livros = pd.concat([estrategia(treino) for estrategia in strategies], axis=1).fillna(0.0)
        return livros.mean(axis=1)
    return _estrategia
