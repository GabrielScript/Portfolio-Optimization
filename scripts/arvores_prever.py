"""
arvores_prever.py - The tree forecasts over the B3 package (docs/superpowers/specs/2026-11-16-arvores-design.md,
sections 2-3), in ``.venv-qpo``.

    PYTHONPATH=. ../.venv-qpo/Scripts/python.exe -X utf8 scripts/arvores_prever.py D:/b3-datalake/exports/qml/b3 RF
    ... GBR / LGBM (one process per model, in parallel: each writes only its own ``arvores_<model>.npz``)
    ... juntar     (adds every model's columns to ``previsoes.npz``; the 213 and KAN columns checked by hash)
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

from qpo_kaggle.arvores import LIGHTGBM, MODELOS, prever_arvore
from qpo_kaggle.qml_pacote import ler_pacote

SEMENTES = (0, 1, 2)
ATRIBUTOS = ("momentum", "baixa_vol", "dividendos", "formula_magica", "trending_value", "crescimento")


def _hashes(colunas: dict[str, np.ndarray]) -> dict[str, str]:
    return {nome: hashlib.sha256(np.ascontiguousarray(valor).tobytes()).hexdigest() for nome, valor in colunas.items()}


def prever(pasta: Path, nome: str) -> None:
    pacote = ler_pacote(pasta / "pacote.npz")
    biblioteca = f"sklearn {sklearn.__version__}"
    if nome == "LGBM":
        import lightgbm
        biblioteca = f"lightgbm {lightgbm.__version__}" if LIGHTGBM else f"{biblioteca} (HistGradientBoosting fallback)"
    print(f"{nome}: {biblioteca}; months {len(pacote.previstos)}; rows {len(pacote.y):,}", flush=True)
    colunas: dict[str, np.ndarray] = {}
    inicio = time.perf_counter()
    for semente in SEMENTES:
        def progresso(posicao: int) -> None:
            if (posicao + 1) % 25 == 0:
                print(f"  seed {semente}: {posicao + 1} months, {time.perf_counter() - inicio:.0f} s", flush=True)
        colunas[f"{nome}_{semente}"] = prever_arvore(MODELOS[nome], pacote, semente, progresso)
    colunas[nome] = np.mean([colunas[f"{nome}_{s}"] for s in SEMENTES], axis=0)
    ultimo = MODELOS[nome](0).fit(pacote.theta[pacote.treinos[-1]], pacote.y[pacote.treinos[-1]])
    importancia = np.asarray(ultimo.feature_importances_, dtype=float)
    colunas[f"{nome}_importancia"] = importancia / importancia.sum()
    np.savez_compressed(pasta / f"arvores_{nome}.npz", **colunas)
    print(f"{nome}: done in {time.perf_counter() - inicio:.0f} s; last-month importance "
          + ", ".join(f"{a} {v:.3f}" for a, v in zip(ATRIBUTOS, colunas[f'{nome}_importancia'], strict=True)), flush=True)


def juntar(pasta: Path) -> None:
    saida = pasta / "previsoes.npz"
    with np.load(saida) as antigos:
        resultados = dict(antigos)
    antes = _hashes(resultados)
    for nome in MODELOS:
        with np.load(pasta / f"arvores_{nome}.npz") as novos:
            resultados |= {chave: novos[chave] for chave in novos.files if not chave.endswith("_importancia")}
    np.savez_compressed(saida, **resultados)
    with np.load(saida) as gravados:
        if _hashes({nome: gravados[nome] for nome in antes}) != antes:
            raise SystemExit("an earlier column changed")
    print(f"previsoes.npz: {len(antes)} earlier columns unchanged (sha256), tree columns added")


def main(argv: list[str]) -> int:
    pasta, alvo = Path(argv[0]), argv[1]
    juntar(pasta) if alvo == "juntar" else prever(pasta, alvo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
