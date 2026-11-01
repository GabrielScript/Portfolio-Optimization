"""
qml_painel.py - Builds the hybrid forecast's panel on the frozen data (docs/superpowers/specs/2026-11-01-qpo-hibrido-design.md).
Prints the inputs' hashes. No market result is computed here.

    python -X utf8 scripts/qml_painel.py b3
    python -X utf8 scripts/qml_painel.py us [exports folder]
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.crescimento_lucro import nota_de_crescimento  # noqa: E402
from core.qml_painel import baixa_volatilidade, pacote_do_painel, painel_de_atributos  # noqa: E402
from core.sinais import dividend_yield_12m, momentum_12_1  # noqa: E402
from core.valor import Balancos, NotaDoMes, formula_magica_no_mes, trending_value  # noqa: E402
from qpo_kaggle.qml_pacote import guardar_pacote  # noqa: E402

SAIDA = Path("D:/b3-datalake/exports/qml")
INICIO = {"b3": pd.Timestamp("2009-11-01"), "us": pd.Timestamp("2019-11-01")}


def notas_de_atributos(retornos: pd.DataFrame, proventos: pd.DataFrame, balancos: Balancos) -> Mapping[str, NotaDoMes]:
    """The Composite's inputs, each from the function the strategies use, with higher meaning better."""
    def so_dos(mes_para_nota):
        return lambda mes, candidatos: mes_para_nota(retornos[[c for c in candidatos if c in retornos.columns]], mes)

    return {"momentum": so_dos(momentum_12_1), "baixa_vol": so_dos(baixa_volatilidade),
            "dividendos": lambda mes, candidatos: dividend_yield_12m(proventos, mes, candidatos),
            "formula_magica": lambda mes, candidatos: formula_magica_no_mes(balancos, mes, candidatos),
            "trending_value": lambda mes, candidatos: trending_value(balancos, retornos, proventos, mes, candidatos),
            "crescimento": nota_de_crescimento(balancos.fundamentos)}


def _dados_b3() -> tuple[pd.DataFrame, pd.DataFrame, Balancos, pd.DataFrame, dict[str, str], NotaDoMes]:
    from scripts import avaliar_conservadora, avaliar_robustez, avaliar_valor
    from scripts.paper_v3 import nota_do_composto
    from scripts.qpo_montar import FOLDER

    avaliar_robustez.PASTA_PRECOS = avaliar_robustez.PASTA_UNIVERSO = avaliar_valor.PASTA_UNIVERSO = FOLDER
    dados = avaliar_robustez.carregar()
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, sha256 = avaliar_valor.carregar_balancos(FOLDER.parent / "2026-10-24-cvm")
    return (mensal.retornos, mensal.proventos, balancos, dados.universo, dados.sha256 | mensal.sha256 | sha256,
            nota_do_composto(mensal.retornos, mensal.proventos, balancos))


def _dados_us(pasta: Path) -> tuple[pd.DataFrame, pd.DataFrame, Balancos, pd.DataFrame, dict[str, str], NotaDoMes]:
    from scripts.paper_v3 import nota_do_composto
    from scripts.us_avaliar import carregar

    dados = carregar(pasta)
    sha256 = {arquivo.name: hashlib.sha256(arquivo.read_bytes()).hexdigest() for arquivo in sorted(pasta.glob("*.parquet"))}
    return (dados.retornos, dados.proventos, dados.balancos, dados.universo, sha256,
            nota_do_composto(dados.retornos, dados.proventos, dados.balancos))


def main(argv: list[str]) -> int:
    inicio = time.perf_counter()
    mercado = argv[0]
    retornos, proventos, balancos, universo, sha256, nota_composto = (
        _dados_b3() if mercado == "b3" else _dados_us(Path(argv[1]) if len(argv) > 1 else Path("D:/b3-datalake/exports/us_stocks")))
    print(f"Files (sha256): {sha256}")
    meses = pd.DatetimeIndex(sorted(pd.to_datetime(universo["month"]).unique()))
    painel = painel_de_atributos(notas_de_atributos(retornos, proventos, balancos), retornos, universo, meses)
    composto = pd.concat([nota_composto(mes, list(parte["security"])).rename("composto").rename_axis("security").reset_index()
                          .assign(month=mes) for mes, parte in painel.groupby("month") if mes >= INICIO[mercado]])
    painel = painel.merge(composto, on=["month", "security"], how="left")
    destino = SAIDA / mercado
    destino.mkdir(parents=True, exist_ok=True)
    pacote, chaves = pacote_do_painel(painel, meses[meses >= INICIO[mercado]])
    painel.to_parquet(destino / "painel.parquet", index=False)
    chaves.to_parquet(destino / "chaves.parquet", index=False)
    guardar_pacote(destino / "pacote.npz", pacote)
    print(f"{mercado}: {len(painel):,} rows, {painel['security'].nunique():,} names, {meses[0]:%Y-%m} to {meses[-1]:%Y-%m}, "
          f"{len(pacote.previstos)} months to forecast, training rows {len(pacote.treinos[0]):,} to "
          f"{len(pacote.treinos[-1]):,}; missing share of each feature: "
          f"{(painel[['momentum', 'baixa_vol', 'dividendos', 'formula_magica', 'trending_value', 'crescimento']] == 0).mean().round(2).to_dict()}; "
          f"done in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
