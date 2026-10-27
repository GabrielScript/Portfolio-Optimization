/-
`core.qpo.build_qubo`: `Q = (λ / (2k²)) (C + Cᵀ)/2 - diag(α)/k`. On a 0/1 vector `x` its energy is the negative
mean-variance utility of the equal-weight portfolio `w = x / k`: `x'Qx = -α'w + (λ/2) w'Cw`.
-/
import QpoFormal.Energy

open Finset

namespace QpoFormal

noncomputable section

variable {n : ℕ}

/-- The QUBO of `build_qubo` (alpha `α`, covariance `C`, cardinality `k`, risk aversion `lam`). -/
def buildQubo (α : Fin n → ℝ) (C : Fin n → Fin n → ℝ) (k lam : ℝ) : Fin n → Fin n → ℝ :=
  fun i j ↦ lam / (2 * k ^ 2) * symm C i j - (if i = j then α i / k else 0)

/-- `x'Qx = -α'w + (λ/2) w'Cw` at `w = x / k`, for every 0/1 vector `x` and every `k ≠ 0`. -/
theorem energy_build_qubo (α : Fin n → ℝ) (C : Fin n → Fin n → ℝ) {k : ℝ} (hk : k ≠ 0) (lam : ℝ)
    {x : Fin n → ℝ} (hx : Binary x) :
    energy (buildQubo α C k lam) x
      = -(∑ i, α i * (x i / k)) + lam / 2 * energy C (fun i ↦ x i / k) := by
  have hdiag : ∑ i, ∑ j, x i * (if i = j then α i / k else 0) * x j = ∑ i, α i * (x i / k) := by
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    have : ∀ j, x i * (if i = j then α i / k else 0) * x j
        = if i = j then α i / k * (x i * x i) else 0 := by
      intro j
      by_cases h : i = j
      · subst h; rw [if_pos rfl, if_pos rfl]; ring
      · rw [if_neg h, if_neg h]; ring
    rw [Finset.sum_congr rfl fun j _ ↦ this j, Finset.sum_ite_eq]
    simp only [Finset.mem_univ, ↓reduceIte, hx.mul_self i]
    ring
  have hrisk : energy C (fun i ↦ x i / k) = energy C x / k ^ 2 := by
    unfold energy
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl fun j _ ↦ ?_
    first | (field_simp; ring) | field_simp
  have hquad : ∑ i, ∑ j, x i * (lam / (2 * k ^ 2) * symm C i j) * x j
      = lam / (2 * k ^ 2) * energy (symm C) x := by
    unfold energy
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ ↦ by ring
  have hsplit : energy (buildQubo α C k lam) x
      = ∑ i, ∑ j, x i * (lam / (2 * k ^ 2) * symm C i j) * x j
        - ∑ i, ∑ j, x i * (if i = j then α i / k else 0) * x j := by
    unfold energy buildQubo
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ ↦ by ring
  rw [hsplit, hquad, hdiag, hrisk, energy_symm C x]
  first | (field_simp; ring) | field_simp | ring

end

end QpoFormal
