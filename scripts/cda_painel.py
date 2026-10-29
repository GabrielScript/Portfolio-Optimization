"""
Builds the frozen best-ideas panel of `docs/superpowers/specs/2026-10-29-cda-best-ideas-design.md` (pre-registered in
`132a4c1`) from the lake's CDA export, the frozen market caps and the frozen universe. Run once:

    python -X utf8 scripts/cda_painel.py

Writes docs/dados/2026-10-29-cda/best_ideas_panel.parquet and its SHA256SUMS line. Prints only coverage (funds and
named issuers per month), never a return or an IC.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.melhores_ideias import ideias_por_carteira, painel_de_ideias, publico_a_partir, referencia  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

EXPORTS = Path(os.environ.get("LAKE_DIR", "D:/b3-datalake")) / "exports"
ESPERADOS = {"fund_stock_holdings.parquet": "cd997a5d06f1b41b4988ef0da06518c1a032768380bd2ebee350af5e78b14feb",
             "fund_nav.parquet": "e12d47008b06a5a7eca999c0e00c3554bda862dfaceeacbb6f8893a4896121ee"}
PASTA_CAPS = FOLDER.parent / "2026-10-29-cvm"
SAIDA = FOLDER.parent / "2026-10-29-cda"
PAINEL = SAIDA / "best_ideas_panel.parquet"
MESES = pd.date_range("2011-01-01", "2026-09-01", freq="MS")
CORTE = pd.Timestamp("2026-04-30")
PRIMEIRA_CARTEIRA = pd.Timestamp("2009-12-31")   # 12 months before the first signal month


def conferir_export(pasta: Path, esperados: dict[str, str]) -> dict[str, str]:
    """The sha256 of each file, refusing (SystemExit) any that differs from the pre-registered one."""
    achados = {}
    for nome, esperado in esperados.items():
        achado = hashlib.sha256((pasta / nome).read_bytes()).hexdigest()
        if achado != esperado:
            raise SystemExit(f"{nome}: sha256 {achado} is not the pre-registered {esperado}")
        achados[nome] = achado
    return achados


def main() -> int:
    inicio = time.perf_counter()
    sha256 = conferir_export(EXPORTS, ESPERADOS) | conferir_hashes(PASTA_CAPS, ["market_caps.parquet"]) \
        | conferir_hashes(FOLDER, ["universe_monthly.parquet"])
    print(f"Files (sha256): {sha256}")
    posicoes = pd.read_parquet(EXPORTS / "fund_stock_holdings.parquet")
    posicoes = posicoes.assign(month=pd.to_datetime(posicoes["month"]),
                               confid_until=pd.to_datetime(posicoes["confid_until"]))
    posicoes = posicoes[(posicoes["month"] >= PRIMEIRA_CARTEIRA) & (posicoes["month"] <= CORTE)]
    navs = pd.read_parquet(EXPORTS / "fund_nav.parquet")
    navs = navs.assign(month=pd.to_datetime(navs["month"]))
    caps = pd.read_parquet(PASTA_CAPS / "market_caps.parquet", columns=["issuer", "month", "market_cap"])
    universo = pd.read_parquet(FOLDER / "universe_monthly.parquet")
    print(f"Positions {len(posicoes):,} of {posicoes['cnpj'].nunique():,} funds, portfolios "
          f"{posicoes['month'].min():%Y-%m} to {posicoes['month'].max():%Y-%m}", flush=True)

    publico = publico_a_partir(posicoes)
    ideias = ideias_por_carteira(posicoes, navs, caps, universo)
    print(f"Portfolios {len(ideias):,}, qualifying {int(ideias['qualifica'].sum()):,} "
          f"({time.perf_counter() - inicio:.0f} s)", flush=True)
    painel = painel_de_ideias(ideias, publico, referencia(caps, universo), universo, MESES, CORTE)

    SAIDA.mkdir(parents=True, exist_ok=True)
    painel.to_parquet(PAINEL, index=False)
    hash_painel = hashlib.sha256(PAINEL.read_bytes()).hexdigest()
    (SAIDA / "SHA256SUMS").write_text(f"{hash_painel} *{PAINEL.name}\n", encoding="ascii", newline="\n")
    por_mes = painel.groupby("month").agg(fundos=("qualifying", "first"), nomeados=("score", lambda s: (s > 0).sum()))
    print(f"Panel {len(painel):,} rows, sha256 {hash_painel}")
    print(f"Qualifying funds per month: min {por_mes['fundos'].min()}, median {por_mes['fundos'].median():.0f}, "
          f"max {por_mes['fundos'].max()}; months under 30: {(por_mes['fundos'] < 30).sum()}")
    print(f"Issuers named per month: median {por_mes['nomeados'].median():.0f}")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
