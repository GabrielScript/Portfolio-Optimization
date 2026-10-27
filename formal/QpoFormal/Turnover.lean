/-
The turnover term of DECISOES 150: `τ Σ (xᵢ - hᵢ)²` with `h` the incumbent picks. On 0/1 vectors it is a diagonal
change of `Q` plus a constant, and on the `|x| = k` slice the one-way turnover of equal weights is `(k - x'h)/k`.
-/
import QpoFormal.Energy

open Finset

namespace QpoFormal

noncomputable section

variable {n : ℕ}

/-- `Σ (xᵢ - hᵢ)² = Σ xᵢ (1 - 2hᵢ) + |h|` on 0/1 vectors: the penalty is linear in `x`. -/
theorem turnover_sum_sq {x h : Fin n → ℝ} (hx : Binary x) (hh : Binary h) :
    ∑ i, (x i - h i) ^ 2 = ∑ i, x i * (1 - 2 * h i) + ∑ i, h i := by
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  have h1 := hx.mul_self i
  have h2 := hh.mul_self i
  nlinarith [h1, h2]

/-- `x'Qx + τ Σ (xᵢ - hᵢ)² = x'(Q + τ diag(1 - 2h))x + τ |h|` on 0/1 vectors. -/
theorem turnover_qubo (Q : Fin n → Fin n → ℝ) (τ : ℝ) {x h : Fin n → ℝ} (hx : Binary x)
    (hh : Binary h) :
    energy Q x + τ * ∑ i, (x i - h i) ^ 2
      = energy (fun i j ↦ Q i j + if i = j then τ * (1 - 2 * h i) else 0) x + τ * ∑ i, h i := by
  rw [energy_add_diag Q _ hx, turnover_sum_sq hx hh, mul_add, Finset.mul_sum, Finset.mul_sum]
  have : ∀ i, τ * (x i * (1 - 2 * h i)) = τ * (1 - 2 * h i) * x i := fun i ↦ by ring
  simp only [this]
  ring

/-- On the slice `|x| = k`, `Q + τ diag(1 - 2h)` and `Q - 2τ diag(h)` differ by the constant `τ k`. -/
theorem turnover_slice (Q : Fin n → Fin n → ℝ) (τ : ℝ) {x h : Fin n → ℝ} (hx : Binary x) {k : ℝ}
    (hxk : ∑ i, x i = k) :
    energy (fun i j ↦ Q i j + if i = j then τ * (1 - 2 * h i) else 0) x
      = energy (fun i j ↦ Q i j + if i = j then -(2 * τ * h i) else 0) x + τ * k := by
  rw [energy_add_diag Q _ hx, energy_add_diag Q _ hx]
  have : ∑ i, τ * (1 - 2 * h i) * x i = ∑ i, -(2 * τ * h i) * x i + τ * ∑ i, x i := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ ↦ by ring
  rw [this, hxk]
  ring

/-- With `|x| = |h| = k > 0`, the one-way turnover `½ Σ |xᵢ/k - hᵢ/k|` of equal weights is `(k - x'h)/k`. -/
theorem turnover_one_way {x h : Fin n → ℝ} (hx : Binary x) (hh : Binary h) {k : ℝ} (hk : 0 < k)
    (hxk : ∑ i, x i = k) (hhk : ∑ i, h i = k) :
    (∑ i, |x i / k - h i / k|) / 2 = (k - ∑ i, x i * h i) / k := by
  have habs : ∀ i, |x i / k - h i / k| = (x i - h i) ^ 2 / k := by
    intro i
    rw [show x i / k - h i / k = (x i - h i) / k by ring, abs_div, abs_of_pos hk]
    congr 1
    rcases hx i with h1 | h1 <;> rcases hh i with h2 | h2 <;> norm_num [h1, h2]
  simp only [habs]
  rw [← Finset.sum_div]
  have hsq : ∑ i, (x i - h i) ^ 2 = 2 * k - 2 * ∑ i, x i * h i := by
    rw [turnover_sum_sq hx hh, hhk]
    have : ∑ i, x i * (1 - 2 * h i) = ∑ i, x i - 2 * ∑ i, x i * h i := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ ↦ by ring
    rw [this, hxk]
    ring
  rw [hsq]
  first | (field_simp; ring) | field_simp

end

end QpoFormal
