"""The continuous out-of-sample curve, 2010 to 2026.

``drl.corrida`` returns one row per model, and each model is trained only on the five years before its test year, so the
seventeen rows are seventeen genuine out-of-sample segments. Read one by one they answer "was this year good"; chained
end to end they answer the only question that compares with the Composite: what does the strategy compound at, and at
what risk, over the whole period.

The one thing that is easy to get wrong: every model's curve starts at 1.0, so concatenating the segments as they come
resets the level every January and manufactures a crash that never happened. ``emendar`` rescales each segment onto the
level the previous one ended at."""

from __future__ import annotations

import numpy as np


def emendar(curvas: list[np.ndarray]) -> np.ndarray:
    """The continuous curve of per-year value curves, each starting at 1.0.

    Segment k is multiplied by the level segment k-1 ended at, then its first point (which repeats the previous
    segment's last value) is dropped."""
    if not curvas:
        return np.zeros(0)
    saida = np.asarray(curvas[0], dtype=float)
    for curva in curvas[1:]:
        c = np.asarray(curva, dtype=float)
        c = c * (saida[-1] / c[0])
        saida = np.concatenate([saida, c[1:]])
    return saida


def por_ano(linhas: list[dict], coluna: str = "algoritmo") -> dict[str, list[dict]]:
    """Group result rows by the model, each group ordered by its test year — the shape ``emendar`` wants."""
    grupos: dict[str, list[dict]] = {}
    for linha in linhas:
        grupos.setdefault(str(linha[coluna]) + "/" + str(linha.get("recompensa", "-")), []).append(linha)
    return {chave: sorted(grupo, key=lambda r: r["ano_teste"]) for chave, grupo in grupos.items()}
