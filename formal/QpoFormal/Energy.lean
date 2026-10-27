/-
The QUBO energy `x'Qx` of `qpo_kaggle/qubo_math.energy`, 0/1 vectors and the symmetric part of `Q`:
the shared definitions of the QPO formalization.
-/
import Mathlib

open Finset

namespace QpoFormal

variable {n : ℕ}

/-- `x'Qx` (`qubo_math.energy`). -/
def energy (Q : Fin n → Fin n → ℝ) (x : Fin n → ℝ) : ℝ :=
  ∑ i, ∑ j, x i * Q i j * x j

/-- A 0/1 vector: bit `i` is `1` when name `i` is picked. -/
def Binary (x : Fin n → ℝ) : Prop := ∀ i, x i = 0 ∨ x i = 1

lemma Binary.mul_self {x : Fin n → ℝ} (hx : Binary x) (i : Fin n) : x i * x i = x i := by
  rcases hx i with h | h <;> simp [h]

/-- The symmetric part `(Q + Qᵀ)/2` (`s` in `qubo_math.qubo_to_ising`). -/
noncomputable def symm (Q : Fin n → Fin n → ℝ) : Fin n → Fin n → ℝ := fun i j ↦ (Q i j + Q j i) / 2

/-- Only the symmetric part of `Q` matters: `x'Qx = x'((Q + Qᵀ)/2)x`, for any real `x`. -/
theorem energy_symm (Q : Fin n → Fin n → ℝ) (x : Fin n → ℝ) :
    energy Q x = energy (symm Q) x := by
  have hswap : ∑ i, ∑ j, x i * Q j i * x j = ∑ i, ∑ j, x i * Q i j * x j := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun j _ ↦ ?_
    ring
  unfold energy symm
  have : ∑ i, ∑ j, x i * ((Q i j + Q j i) / 2) * x j
      = (∑ i, ∑ j, x i * Q i j * x j + ∑ i, ∑ j, x i * Q j i * x j) / 2 := by
    rw [← Finset.sum_add_distrib, Finset.sum_div]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [← Finset.sum_add_distrib, Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ ↦ ?_
    ring
  rw [this, hswap]
  ring

/-- Adding `diag(d)` to `Q` adds `Σ dᵢ xᵢ` to the energy of a 0/1 vector. -/
lemma energy_add_diag (Q : Fin n → Fin n → ℝ) (d : Fin n → ℝ) {x : Fin n → ℝ} (hx : Binary x) :
    energy (fun i j ↦ Q i j + if i = j then d i else 0) x = energy Q x + ∑ i, d i * x i := by
  unfold energy
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  have : ∀ j, x i * (Q i j + if i = j then d i else 0) * x j
      = x i * Q i j * x j + if i = j then d i * (x i * x i) else 0 := by
    intro j
    by_cases hij : i = j
    · subst hij; simp; ring
    · simp [hij]
  rw [Finset.sum_congr rfl fun j _ ↦ this j, Finset.sum_add_distrib, Finset.sum_ite_eq]
  simp [hx.mul_self i]

end QpoFormal
