"""
debentures.py - The company's own debentures as a credit signal on the Composite
(docs/superpowers/specs/2026-11-24-debentures-design.md, committed before any return-based number).

The SND files (debentures.com.br, ANBIMA) stay on ``D:/b3-datalake/raw/snd/`` and are never committed (DECISOES 110);
their hashes are in ``docs/dados/2026-11-24-snd/SHA256SUMS``. Research only: the live Composite is not changed.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from core.risco import emissor
from core.valor import NotaDoMes
from core.views import escore_normal

RAIZ = Path(__file__).resolve().parents[1]
RAW = Path("D:/b3-datalake/raw/snd")
SUMS = RAIZ / "docs" / "dados" / "2026-11-24-snd" / "SHA256SUMS"
TRADES, CHARACTERISTICS = "snd_trades.parquet", "caracteristicas/caracteristicas.txt"
HEADER_PREFIX = "Codigo do Ativo"
DI = "DI"
LAG_CHANGE = 3                  # CRD-V: M - 1 minus M - 4
KINDS = ("level", "change")
VETO_QUANTILE, VETO_MIN_NAMES = 0.10, 20
MIN_MONTHS = 36                 # months of IC before a score earns any weight (DECISOES 354)
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


# earned_weight and weighted_score: unchanged from ``core/informados.py`` on ``feat/informados`` (DECISOES 354's rule)
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


def vetoed_names(panel: pd.DataFrame, quantile: float = VETO_QUANTILE,
                 min_names: int = VETO_MIN_NAMES) -> dict[pd.Timestamp, frozenset[str]]:
    """Per month (row of ``panel``: the score over the month's universe names), the names at or below the score's
    ``quantile`` (pandas, linear) among those scored; none in a month with fewer than ``min_names`` scored."""
    vetados = {}
    for mes, linha in panel.iterrows():
        notas = linha.dropna()
        if len(notas) < min_names:
            vetados[mes] = frozenset()
        else:
            vetados[mes] = frozenset(notas.index[notas <= notas.quantile(quantile)])
    return vetados


def veto_score(base: NotaDoMes, vetoed: Mapping[pd.Timestamp, frozenset[str]]) -> NotaDoMes:
    """``base`` without the candidates vetoed in the month (a month absent from ``vetoed`` vetoes nothing)."""
    def _nota(mes: pd.Timestamp, candidatos: Sequence[str]) -> pd.Series:
        fora = vetoed.get(mes, frozenset())
        return base(mes, [papel for papel in candidatos if papel not in fora])
    return _nota
