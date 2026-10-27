# QpoFormal — Lean 4 proofs of the QPO's math

Lean 4 (`v4.36.0-rc1`) + Mathlib proofs of the algebra behind `qpo_kaggle/` and `core/qpo.py`, for every
`n`, every `Q` and every bitstring (the Python tests check random instances only).

| Module | What it proves | Mirrors |
|---|---|---|
| `Energy` | `x'Qx = x'((Q+Qᵀ)/2)x`; adding `diag(d)` adds `Σ dᵢxᵢ` on 0/1 vectors | `qubo_math.energy` |
| `Ising` | `qubo_to_ising` is exact on every bitstring | `qubo_math.qubo_to_ising` |
| `BuildQubo` | `x'Qx = -α'w + (λ/2)w'Cw` at `w = x/k` | `core.qpo.build_qubo` |
| `Turnover` | the turnover term is a diagonal change of `Q`; one-way turnover `(k - x'h)/k` | DECISOES 150 |
| `Minimizers` | a constant shift or positive scale moves no minimizer | `ising_scale`, constants |
| `Ratio` | approximation ratio in `[0, 1]`, `1` exactly at the optimum | `qubo_math.approximation_ratio` |
| `Sizes` | `C(20,15) = 15,504 ≤ 2^24 < C(28,15) = 37,442,160` | `MAX_SUBSPACE` guard |
| `Subspace` | Dicke start + diagonal cost layers + XY gates never leave weight `k` | `qaoa_subspace.py`, no penalty |

Status (session 54): every module builds. Proven: `Energy`, `BuildQubo`, `Turnover`, `Minimizers`, `Ratio`,
`Sizes`, and in `Subspace` the weight-k invariance (`qaoa_supported`, `qaoa_never_infeasible`). Still `sorry`:
`Ising` (`energy_symm_split`, `energy_eq_ising`, `sum_lt_eq_half`, `energy_eq_ising_triu`) and
`Subspace.card_weight_eq`.

Check (from this folder; first time `lake exe cache get` downloads Mathlib's build, ~5 GB):

```
lake exe cache get
lake build
```

A clean build with no `sorry` warning means every theorem is machine-checked (standard axioms only).
