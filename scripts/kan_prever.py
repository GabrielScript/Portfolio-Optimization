"""
kan_prever.py - KAN-1 and KAN-2 over the B3 package (docs/superpowers/specs/2026-11-07-kan-design.md, section 3), in
``.venv-qpo``. Adds their columns to the 213 ``previsoes.npz`` and checks, by hash, that the 213 columns did not move.
Also writes the curves of each seed's last-month KAN-1. Resumes where it stopped.

    PYTHONPATH=. ../.venv-qpo/Scripts/python.exe -X utf8 scripts/kan_prever.py D:/b3-datalake/exports/qml/b3
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qpo_kaggle.kan import Kan1, Kan2, curvas
from qpo_kaggle.qml_pacote import ler_pacote
from qpo_kaggle.qml_sinal import Treino, prever_torch

SEMENTES = (0, 1, 2)
FIOS = 3
MODELOS = {"KAN-1": Kan1, "KAN-2": Kan2}


def _hashes(colunas: dict[str, np.ndarray]) -> dict[str, str]:
    return {nome: hashlib.sha256(np.ascontiguousarray(valor).tobytes()).hexdigest() for nome, valor in colunas.items()}


def main(argv: list[str]) -> int:
    pasta = Path(argv[0])
    torch.set_num_threads(FIOS)
    pacote = ler_pacote(pasta / "pacote.npz")
    saida = pasta / "previsoes.npz"
    with np.load(saida) as antigos:
        resultados = dict(antigos)
    de_213 = {nome: valor for nome, valor in resultados.items() if not nome.startswith("KAN")}
    antes = _hashes(de_213)
    print(f"versions: torch {torch.__version__}; months {len(pacote.previstos)}; rows {len(pacote.y):,}; "
          f"213 columns {sorted(de_213)}", flush=True)
    for nome, classe in MODELOS.items():
        if nome in resultados:
            continue
        inicio = time.perf_counter()
        por_semente = [prever_torch(classe, pacote, semente, Treino(), pasta / f"estado_{nome}") for semente in SEMENTES]
        resultados[nome] = np.mean(por_semente, axis=0)
        for semente, previsao in zip(SEMENTES, por_semente, strict=True):
            resultados[f"{nome}_{semente}"] = previsao
        np.savez_compressed(saida, **resultados)
        print(f"{nome}: {time.perf_counter() - inicio:.0f} s", flush=True)
    with np.load(saida) as gravados:
        depois = _hashes({nome: gravados[nome] for nome in de_213})
    if depois != antes:
        raise SystemExit("a 213 column changed")
    print("213 columns unchanged (sha256)")
    tabelas = []
    for semente in SEMENTES:
        modelo = Kan1(semente=semente)
        estado = torch.load(pasta / "estado_KAN-1" / f"estado_{semente}.pt", map_location="cpu", weights_only=True)
        modelo.load_state_dict(estado["modelo"])
        tabelas.append(curvas(modelo).assign(semente=semente))
    pd.concat(tabelas, ignore_index=True).to_csv(pasta / "kan_curvas.csv", index=False)   # no pyarrow in .venv-qpo
    print("curves written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
