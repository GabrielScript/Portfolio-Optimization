"""
painel_amplo_montar.py - The broad panel's package (docs/superpowers/specs/2026-11-16-painel-amplo-design.md, sections 2
and 3): the 213 package's rows, target and training rows, with its six angles followed by the 23 new characteristics.
Writes ``painel_run/pacote_amplo.npz`` (the app's venv cannot write on D:). Prints the inputs' hashes.

    python -X utf8 scripts/painel_amplo_montar.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.painel_amplo import NOVAS, Dados, angulos_do_mes  # noqa: E402
from qpo_kaggle.qml_pacote import Pacote, guardar_pacote, ler_pacote  # noqa: E402
from scripts import avaliar_conservadora, avaliar_valor  # noqa: E402
from scripts.qpo_montar import FOLDER  # noqa: E402

PACOTE_213 = Path("D:/b3-datalake/exports/qml/b3")
SAIDA = Path(__file__).resolve().parent.parent / "painel_run"
SEIS = ("momentum", "baixa_vol", "dividendos", "formula_magica", "trending_value", "crescimento")


def main() -> int:
    inicio = time.perf_counter()
    avaliar_valor.PASTA_UNIVERSO = FOLDER
    mensal = avaliar_conservadora.carregar_mensal(FOLDER)
    balancos, sha256_cvm = avaliar_valor.carregar_balancos(FOLDER.parent / "2026-10-24-cvm")
    caps = pd.read_parquet(FOLDER.parent / "2026-10-24-cvm" / "market_caps.parquet",      # same hashed file, + shares
                           columns=["issuer", "month", "market_cap", "shares_today_units"])
    caps["month"] = pd.to_datetime(caps["month"])
    balancos = replace(balancos, valor_mercado=caps)
    fechamentos = pd.read_parquet(FOLDER / "universe_prices.parquet")
    fechamentos = fechamentos.set_index(pd.DatetimeIndex(pd.to_datetime(fechamentos.pop("date")))).astype("float64")
    pacote = ler_pacote(PACOTE_213 / "pacote.npz")
    chaves = pd.read_parquet(PACOTE_213 / "chaves.parquet")
    hashes = {nome: hashlib.sha256((PACOTE_213 / nome).read_bytes()).hexdigest()
              for nome in ("pacote.npz", "chaves.parquet")}
    print(f"Files (sha256): {mensal.sha256 | sha256_cvm | hashes}", flush=True)
    if len(chaves) != len(pacote.y):
        raise SystemExit("chaves and the 213 package differ in rows")
    numeros = pd.DatetimeIndex(chaves["month"]).year * 12 + pd.DatetimeIndex(chaves["month"]).month - 1
    if not np.array_equal(np.asarray(numeros), pacote.mes):
        raise SystemExit("chaves' months differ from the package's")

    dados = Dados(mensal.retornos, fechamentos, balancos)
    partes = []
    for posicao, (mes, linhas) in enumerate(chaves.groupby("month", sort=True)):
        tabela = angulos_do_mes(dados, pd.Timestamp(mes), list(linhas["security"]))
        partes.append(tabela.set_axis(linhas.index))
        if (posicao + 1) % 50 == 0:
            print(f"  {posicao + 1} months, {time.perf_counter() - inicio:.0f} s", flush=True)
    novas = pd.concat(partes).reindex(chaves.index)
    if novas.isna().any().any():
        raise SystemExit("a row lost its features")
    theta = np.column_stack([pacote.theta, novas.to_numpy(dtype=float)])
    SAIDA.mkdir(exist_ok=True)
    guardar_pacote(SAIDA / "pacote_amplo.npz", Pacote(theta, pacote.y, pacote.mes, pacote.previstos, pacote.treinos))
    presentes = (novas != 0.0).groupby(chaves["month"].dt.year).mean()
    print(f"{len(theta):,} rows x {theta.shape[1]} features ({', '.join((*SEIS, *NOVAS))})")
    print("Share of rows with each new feature, by year (descriptive):")
    print(presentes.loc[2005:].T.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"Done in {time.perf_counter() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
