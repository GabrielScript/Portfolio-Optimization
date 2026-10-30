"""
Builds the frozen top-funds panel of `docs/superpowers/specs/2026-10-29-top-fundos-copia-design.md` (pre-registered in
`feb033f`) from the lake's daily-report quotas and attempt 52's CDA files, market caps and universe. Run once:

    python -X utf8 scripts/top_fundos_painel.py

Writes docs/dados/2026-10-29-top-fundos/top_fundos_panel.parquet and its SHA256SUMS line. Prints only coverage
(qualifying, quoted and ranked funds per year, months dropped), never a return, a ranking, a score or an IC.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.melhores_ideias import ideias_por_carteira, publico_a_partir, referencia  # noqa: E402
from core.top_fundos import Carteiras, meses_cheios, painel_top_fundos, retornos_mensais  # noqa: E402
from scripts.avaliar_robustez import conferir_hashes  # noqa: E402
from scripts.cda_painel import CORTE, EXPORTS, MESES, PASTA_CAPS, PRIMEIRA_CARTEIRA, conferir_export  # noqa: E402
from scripts.cda_painel import ESPERADOS as ESPERADOS_CDA  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

COTAS = "fund_monthly_quota.parquet"
ESPERADOS = {COTAS: "296fd03dcb36e250bcf12664ce77085414bbfa91b2a0903aad2e801a711dfbab", **ESPERADOS_CDA}
COLUNAS_DAS_COTAS = ["cnpj", "subclass", "month", "date", "quota", "nav"]
SAIDA = FOLDER.parent / "2026-10-29-top-fundos"
PAINEL = SAIDA / "top_fundos_panel.parquet"


def cobertura_por_ano(painel: pd.DataFrame) -> pd.DataFrame:
    """year -> months, median qualifying funds, median shares of them with a return in the window and ranked, and the
    months dropped (fewer than 30 qualifying or 15 ranked)."""
    contagens = painel.groupby("month")[["qualifying", "with_quotas", "ranked"]].first()
    meses = pd.DataFrame({"qualifying": contagens["qualifying"],
                          "with_quotas": contagens["with_quotas"] / contagens["qualifying"],
                          "ranked": contagens["ranked"] / contagens["qualifying"],
                          "dropped": ~contagens.index.isin(meses_cheios(painel))})
    return meses.groupby(meses.index.year).agg(
        months=("qualifying", "size"), qualifying=("qualifying", "median"), with_quotas=("with_quotas", "median"),
        ranked=("ranked", "median"), dropped=("dropped", "sum"))


def carteiras_do_cda() -> tuple[Carteiras, pd.DataFrame]:
    """Attempt 52's CDA tables (same files, cut and functions as scripts/cda_painel.py) and the frozen universe."""
    posicoes = pd.read_parquet(EXPORTS / "fund_stock_holdings.parquet")
    posicoes = posicoes.assign(month=pd.to_datetime(posicoes["month"]),
                               confid_until=pd.to_datetime(posicoes["confid_until"]))
    posicoes = posicoes[(posicoes["month"] >= PRIMEIRA_CARTEIRA) & (posicoes["month"] <= CORTE)]
    navs = pd.read_parquet(EXPORTS / "fund_nav.parquet")
    navs = navs.assign(month=pd.to_datetime(navs["month"]))
    caps = pd.read_parquet(PASTA_CAPS / "market_caps.parquet", columns=["issuer", "month", "market_cap"])
    universo = pd.read_parquet(FOLDER / "universe_monthly.parquet")
    carteiras = Carteiras(ideias=ideias_por_carteira(posicoes, navs, caps, universo),
                          publico=publico_a_partir(posicoes), bench=referencia(caps, universo), corte=CORTE)
    return carteiras, universo


def retornos_dos_fundos(cnpjs: list[str]) -> pd.DataFrame:
    """Monthly returns (core.top_fundos.retornos_mensais) of the CDA funds only, from the daily-report quotas."""
    cotas = pd.read_parquet(EXPORTS / COTAS, columns=COLUNAS_DAS_COTAS, filters=[("cnpj", "in", cnpjs)])
    cotas = cotas.assign(month=pd.to_datetime(cotas["month"]), date=pd.to_datetime(cotas["date"]))
    return retornos_mensais(cotas)


def main() -> int:
    inicio = time.perf_counter()
    sha256 = conferir_export(EXPORTS, ESPERADOS) | conferir_hashes(PASTA_CAPS, ["market_caps.parquet"]) \
        | conferir_hashes(FOLDER, ["universe_monthly.parquet"])
    print(f"Files (sha256): {sha256}")
    carteiras, universo = carteiras_do_cda()
    fundos_do_cda = sorted(carteiras.ideias.index.get_level_values("cnpj").unique())
    qualificados = sorted(carteiras.ideias.index[carteiras.ideias["qualifica"]].get_level_values("cnpj").unique())
    print(f"CDA portfolios {len(carteiras.ideias):,} of {len(fundos_do_cda):,} funds; {len(qualificados):,} funds "
          f"qualify at least once ({time.perf_counter() - inicio:.0f} s)", flush=True)
    retornos = retornos_dos_fundos(fundos_do_cda)
    cotados = set(retornos["cnpj"])
    print(f"Funds with a monthly return series: {len(cotados & set(fundos_do_cda)):,} of {len(fundos_do_cda):,} CDA "
          f"funds, {len(cotados & set(qualificados)):,} of {len(qualificados):,} that qualify at least once",
          flush=True)

    painel = painel_top_fundos(carteiras, retornos, universo, MESES)
    SAIDA.mkdir(parents=True, exist_ok=True)
    painel.to_parquet(PAINEL, index=False)
    hash_painel = hashlib.sha256(PAINEL.read_bytes()).hexdigest()
    (SAIDA / "SHA256SUMS").write_text(f"{hash_painel} *{PAINEL.name}\n", encoding="ascii", newline="\n")
    print(f"Panel {len(painel):,} rows, sha256 {hash_painel}")
    print("\nCoverage by year (medians over the months; shares of the qualifying funds):")
    print(cobertura_por_ano(painel).to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nMonths read: {len(meses_cheios(painel))} of {len(MESES)}")
    print(f"(total {time.perf_counter() - inicio:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
