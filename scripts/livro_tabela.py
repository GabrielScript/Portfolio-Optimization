"""
livro_tabela.py - The screen's table of the book's strategies, from each reading's saved curves and gates: one row per arm, plus the
references. Runs no strategy. A batch whose reading has not finished is left out, and the screen says so.

    python -X utf8 scripts/livro_tabela.py b3
    python -X utf8 scripts/livro_tabela.py us
"""

from __future__ import annotations

import pickle
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.metrics import drawdown, retornos_de_valores  # noqa: E402

DADOS = Path("docs/dados")
EXPORTS = Path("D:/b3-datalake/exports")
PERIODOS = {"b3": 252, "us": 12}
FASES = {"b3": (0, 21, 42), "us": (0, 1, 2)}
LOTES = {1: "todas", 2: "lote2", 3: "lote3"}
REFERENCIAS = ("Composto", "Conservadora")

Giro = Callable[[Any], float]
PorFase = Mapping[int, Any]


def _cagr(valores: pd.Series, periodos: int) -> float:
    return float((valores.iloc[-1] / valores.iloc[0]) ** (periodos / (len(valores) - 1)) - 1.0)


def _medidas(curvas: list[pd.Series], periodos: int) -> dict[str, float]:
    cagrs = [_cagr(curva, periodos) for curva in curvas]
    return {"cagr_min": min(cagrs), "cagr_max": max(cagrs),
            "vol": float(np.mean([retornos_de_valores(curva).std() * np.sqrt(periodos) for curva in curvas])),
            "queda": min(float(drawdown(curva)[1]) for curva in curvas)}


def linhas_do_lote(lote: int, resultados: Mapping[str, PorFase], portao: pd.DataFrame, periodos: int, giro: Giro) -> pd.DataFrame:
    """Each arm's worst and best phase, its mean volatility and turnover, its worst drawdown and most refusals, and the lowest of each gate
    measure: the gate asks for all of them in every phase, so the lowest is what decides."""
    do_portao = portao.set_index("braco")
    linhas = []
    for nome, por_fase in resultados.items():
        medidas = do_portao.loc[nome]
        rodadas = list(por_fase.values())
        linhas.append({"lote": lote, "braco": nome, **_medidas([resultado.valores.dropna() for resultado in rodadas], periodos),
                       "giro": float(np.mean([giro(resultado) for resultado in rodadas])),
                       "recusas": max(len(resultado.recusas) for resultado in rodadas),
                       "p_composto": float(np.min(medidas["p_composto"])), "p_conservadora": float(np.min(medidas["p_conservadora"])),
                       "p_referencia": float(np.min(medidas["p_referencia"])), "dsr": float(np.min(medidas["dsr"])),
                       "passa": bool(medidas["passa"])})
    return pd.DataFrame(linhas)


def _resultados_salvos(pasta: Path, fases: tuple[int, ...], nomes: list[str]) -> dict[str, dict[int, Any]]:
    """The readings' own pickles, never anything downloaded."""
    resultados: dict[str, dict[int, Any]] = {}
    for nome in nomes:
        resultados[nome] = {}
        for fase in fases:
            with (pasta / f"{nome.replace('/', '-')}-fase{fase}.pkl").open("rb") as arquivo:
                resultados[nome][fase] = pickle.load(arquivo)
    return resultados


def _referencias(mercado: str) -> pd.DataFrame:
    """The two references on the same measures. On the B3 from the published curves, on the US from the first batch's reading."""
    if mercado == "b3":
        quadro = pd.read_parquet(DADOS / "composto-curvas-2026-10-28.parquet")
        curvas = {nome: [grupo.set_index("date")[nome].dropna() for _, grupo in quadro.groupby("fase")] for nome in REFERENCIAS}
    else:
        salvos = _resultados_salvos(EXPORTS / "todas" / "us", FASES["us"], list(REFERENCIAS))
        curvas = {nome: [resultado.valores.dropna() for resultado in por_fase.values()] for nome, por_fase in salvos.items()}
    return pd.DataFrame([{"lote": 0, "braco": nome, **_medidas(series, PERIODOS[mercado]), "passa": False} for nome, series in curvas.items()])


def _giro(mercado: str) -> Giro:
    if mercado == "b3":
        return lambda resultado: resultado.turnover_anual_medio
    from core.us_teste import giro_anual_mensal
    return giro_anual_mensal


def main(argv: list[str]) -> int:
    mercado = argv[0]
    partes = [_referencias(mercado)]
    for lote, prefixo in LOTES.items():
        caminho = DADOS / f"{prefixo}-portao-{mercado}.parquet"
        if not caminho.exists():
            print(f"Batch {lote} has no gate on {mercado} yet: left out")
            continue
        portao = pd.read_parquet(caminho)
        resultados = _resultados_salvos(EXPORTS / prefixo / mercado, FASES[mercado], [nome for nome in portao["braco"] if (EXPORTS / prefixo / mercado / f"{nome.replace('/', '-')}-fase{FASES[mercado][0]}.pkl").exists()])
        partes.append(linhas_do_lote(lote, resultados, portao, PERIODOS[mercado], _giro(mercado)))
    tabela = pd.concat(partes, ignore_index=True)
    tabela.to_parquet(DADOS / f"livro-tabela-{mercado}.parquet", index=False)
    print(tabela.to_string(float_format=lambda valor: f"{valor:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
