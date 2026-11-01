"""
pbo_leitura.py - The descriptive reading promised in DECISOES 228: the probability of backtest overfitting on the curves that were saved.
No new trial and no gate: it runs no strategy and changes nothing. It sees only the curves given, not the configurations whose curves were
never kept, so it understates the search that really happened.

    python -X utf8 scripts/pbo_leitura.py b3 D:/b3-datalake/exports/todas/b3 D:/b3-datalake/exports/lote2/b3
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pbo import cscv  # noqa: E402

CURVAS = {"b3": (Path("docs/dados/composto-curvas-2026-10-28.parquet"), ("Composto", "Conservadora")),
          "us": (Path("docs/dados/us-curvas.parquet"), ("Composto", "Conservadora"))}
FASES = {"b3": (0, 21, 42), "us": (0, 1, 2)}
BLOCOS = 16


def curvas_salvas(pastas: list[Path], fase: int) -> dict[str, pd.Series]:
    """Each arm's curve at that phase. These pickles are the readings' own output, never anything downloaded."""
    curvas = {}
    for pasta in pastas:
        for caminho in sorted(pasta.glob(f"*-fase{fase}.pkl")):
            with caminho.open("rb") as arquivo:
                curvas[caminho.name.removesuffix(f"-fase{fase}.pkl")] = pickle.load(arquivo).valores.dropna()
    return curvas


def referencias(mercado: str, fase: int) -> dict[str, pd.Series]:
    caminho, nomes = CURVAS[mercado]
    quadro = pd.read_parquet(caminho)
    quadro = quadro[quadro["fase"] == fase].set_index("date")
    return {nome: quadro[nome].dropna() for nome in nomes}


def main(argv: list[str]) -> int:
    mercado, pastas = argv[0], [Path(pasta) for pasta in argv[1:]]
    for fase in FASES[mercado]:
        curvas = {**curvas_salvas(pastas, fase), **referencias(mercado, fase)}
        retornos = pd.DataFrame(curvas).pct_change(fill_method=None).dropna()
        sobra = {nome: len(curva) for nome, curva in curvas.items() if len(curva) != len(retornos) + 1}
        resultado = cscv(retornos, BLOCOS)
        print(f"{mercado} phase {fase}: {retornos.shape[1]} curves, {len(retornos)} rows, {resultado.divisoes} splits: PBO {resultado.pbo:.3f}, "
              f"median logit {pd.Series(resultado.logits).median():.2f}, most often chosen in sample: {resultado.mais_escolhida}; "
              f"curves cut to the common dates: {sobra or 'none'}")
    print("Curves:", ", ".join(curvas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
