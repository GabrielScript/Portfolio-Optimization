"""Solve the exported QUBOs with XQuad (spec 2026-11-11-xquad-quip-design.md). Runs in the Python 3.13 container.

    docker run --rm -m 1500m -v <quantum-lab>:/lab xquad-lab python -W ignore resolver.py cpu 0 21 42
    docker run ... -v <quantum-lab>/quip-home:/root/.quip xquad-lab python -W ignore resolver.py quip 0

For each decision of ``instancias/fase{f}.npz``: the binary model x'Qx + A (sum x - 15)^2 + A sum_s (sum_{i in s} x_i
+ slack_s - 4)^2, slack on 3 bits weighted 1, 2, 1; A = 2 x the largest absolute row sum of Q; every coefficient
scaled so the largest is 10^6 and rounded to an integer. The best sample is repaired to |x| = 15 and <= 4 per sector
if needed. Writes ``escolhas/{solver}-fase{f}.json`` (one record per decision, resumable).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from xquad.types import XqmxModel

K, TETO, ESCALA, LEITURAS = 15, 4, 1_000_000, 200
PESOS_FOLGA = (1, 2, 1)
RAIZ = Path(__file__).resolve().parent


def modelo(q: np.ndarray, grupos: np.ndarray) -> tuple[XqmxModel, int, float]:
    """The penalized model, its size and the scale applied."""
    n = len(q)
    s = (q + q.T) / 2.0
    a = 2.0 * float(np.abs(s).sum(axis=1).max())
    setores = sorted(set(grupos))
    total = n + len(PESOS_FOLGA) * len(setores)
    linear = np.zeros(total)
    quad: dict[tuple[int, int], float] = {}

    def soma(i: int, j: int, v: float) -> None:
        if i == j:
            linear[i] += v
        else:
            chave = (min(i, j), max(i, j))
            quad[chave] = quad.get(chave, 0.0) + v

    for i in range(n):
        for j in range(n):
            soma(i, j, s[i, j])

    def penalidade(termos: list[tuple[int, float]], alvo: float) -> None:
        # A (sum c_i x_i - alvo)^2 on binaries: c_i^2 - 2 alvo c_i on the diagonal, 2 c_i c_j off it
        for p, (i, ci) in enumerate(termos):
            soma(i, i, a * (ci * ci - 2.0 * alvo * ci))
            for j, cj in termos[p + 1:]:
                soma(i, j, a * 2.0 * ci * cj)

    penalidade([(i, 1.0) for i in range(n)], K)
    for posicao, setor in enumerate(setores):
        folgas = [(n + len(PESOS_FOLGA) * posicao + b, float(w)) for b, w in enumerate(PESOS_FOLGA)]
        penalidade([(i, 1.0) for i in range(n) if grupos[i] == setor] + folgas, TETO)
    maior = max(np.abs(linear).max(), max(abs(v) for v in quad.values()))
    fator = ESCALA / maior
    m = XqmxModel.binary(total)
    for i in range(total):
        if round(linear[i] * fator):
            m.set_linear(i, int(round(linear[i] * fator)))
    for (i, j), v in quad.items():
        if round(v * fator):
            m.set_quad(i, j, int(round(v * fator)))
    return m, total, fator


def energia(s: np.ndarray, picks: list[int]) -> float:
    return float(s[np.ix_(picks, picks)].sum())


def reparar(x: np.ndarray, s: np.ndarray, grupos: np.ndarray) -> tuple[list[int], bool]:
    """Feasible picks from the sample: drop the costliest pick while a sector is over the cap or there are more than
    K; add the cheapest admissible name while there are fewer. Returns the picks and whether anything changed."""
    picks = [int(i) for i in np.flatnonzero(x)]
    original = list(picks)

    def custo(i: int, conjunto: list[int]) -> float:
        return float(s[i, i] + 2.0 * s[i, conjunto].sum() - (2.0 * s[i, i] if i in conjunto else 0.0))

    def contagem(conjunto: list[int]) -> dict:
        return {g: sum(grupos[i] == g for i in conjunto) for g in set(grupos)}

    while True:
        c = contagem(picks)
        cheios = [g for g, v in c.items() if v > TETO]
        if not cheios and len(picks) <= K:
            break
        candidatos = [i for i in picks if not cheios or grupos[i] in cheios]
        picks.remove(max(candidatos, key=lambda i: custo(i, picks)))
    while len(picks) < K:
        c = contagem(picks)
        livres = [i for i in range(len(s)) if i not in picks and c[grupos[i]] < TETO]
        picks.append(min(livres, key=lambda i: custo(i, picks)))
    return sorted(picks), sorted(picks) != sorted(original)


def construir(nome: str):
    if nome == "cpu":
        from xqsa import SolverCPU
        return lambda semente: SolverCPU(num_reads=LEITURAS, seed=semente)
    from xqsa.quip import SolverQuip
    solver = SolverQuip.for_network("aglais", topology="native", autoconfirm=True, autofund=True, timeout=900)
    return lambda semente: solver


def main(argv: list[str]) -> int:
    nome, fases = argv[0], [int(f) for f in argv[1:]]
    fabrica = construir(nome)
    (RAIZ / "escolhas").mkdir(exist_ok=True)
    for fase in fases:
        dados = np.load(RAIZ / "instancias" / f"fase{fase}.npz")
        saida = RAIZ / "escolhas" / f"{nome}-fase{fase}.json"
        feitos = json.loads(saida.read_text()) if saida.exists() else {}
        for i, data in enumerate(dados["dates"]):
            if str(data) in feitos:
                continue
            q, grupos, de_56 = dados[f"q{i}"], dados[f"g{i}"], [int(v) for v in dados[f"p{i}"]]
            s = (q + q.T) / 2.0
            m, total, fator = modelo(q, grupos)
            semente = int(np.datetime64(str(data), "D").astype(int))
            inicio = time.time()
            registro = {"n": len(q), "vars": total}
            try:
                r = fabrica(semente).solve(m)
                x = np.array([int(v) for v in r.sample.values], dtype=int)[:len(q)]
                registro.update(solver=nome, raw_energy=float(r.energy), meta={k: v for k, v in r.metadata.items()
                                                                              if k in ("order_id", "num_submissions",
                                                                                       "quantization_error")})
            except Exception as erro:  # noqa: BLE001 - the spec's fallback: count it, solve on the CPU
                from xqsa import SolverCPU
                r = SolverCPU(num_reads=LEITURAS, seed=semente).solve(m)
                x = np.array([int(v) for v in r.sample.values], dtype=int)[:len(q)]
                registro.update(solver="cpu-fallback", erro=f"{type(erro).__name__}: {erro}"[:300])
            picks, reparado = reparar(x, s, grupos)
            registro.update(picks=picks, tickers=[str(dados[f"t{i}"][j]) for j in picks], repaired=reparado,
                            energy=energia(s, picks), energy_56=energia(s, de_56), same_as_56=picks == sorted(de_56),
                            seconds=round(time.time() - inicio, 1))
            feitos[str(data)] = registro
            saida.write_text(json.dumps(feitos, indent=1))
            print(f"phase {fase} {data} n {len(q)} {registro['solver']} repaired {reparado} same {registro['same_as_56']} "
                  f"dE {registro['energy'] - registro['energy_56']:+.3e} {registro['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
