/-
STATUS (2026-10-28): written, NOT yet compiled (the compile was stopped for low system memory); not imported
by `QpoFormal.lean`, so `lake build` does not check it. Compile with `lake env lean QpoFormal/NormalScore.lean`.

`core.views.escore_normal`: a signal's raw values become `g((rank - 1/2) / n)` with `g = Φ⁻¹` and pandas' average
rank for ties. The properties that make it the right way to combine indicators of different units (a growth or
quality composite): it ignores any strictly increasing re-expression of the raw indicator (log, winsorizing, scaling),
every score sits in `[g(1/(2n)), g(1 - 1/(2n))]` so no outlier can dominate, and without ties the scores of a month
sum to zero. Proven for any increasing `g` with `g(1 - p) = -g(p)`, so they hold for `Φ⁻¹`.
-/
import Mathlib

open Finset

namespace QpoFormal

noncomputable section

variable {n : ℕ}

/-- Pandas' average rank (`Series.rank()`, 1-based): the names strictly below, plus the mean position of the tied
block `(#ties + 1) / 2`. -/
def avgRank (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  (#{j | x j < x i} : ℝ) + ((#{j | x j = x i} : ℝ) + 1) / 2

/-- `escore_normal` with a generic quantile function `g`. -/
def normalScore (g : ℝ → ℝ) (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  g ((avgRank x i - 1 / 2) / n)

/-- A strictly increasing re-expression of the raw indicator moves no rank. -/
theorem avgRank_comp {f : ℝ → ℝ} (hf : StrictMono f) (x : Fin n → ℝ) (i : Fin n) :
    avgRank (f ∘ x) i = avgRank x i := by
  simp only [avgRank, Function.comp_apply, hf.lt_iff_lt, hf.injective.eq_iff]

/-- ... so it moves no normal score: the choice between, say, `ΔNI / assets` and its log or a winsorized version is
irrelevant to the composite. -/
theorem normalScore_comp (g : ℝ → ℝ) {f : ℝ → ℝ} (hf : StrictMono f) (x : Fin n → ℝ) (i : Fin n) :
    normalScore g (f ∘ x) i = normalScore g x i := by
  simp only [normalScore, avgRank_comp hf]

/-- Every average rank lies in `[1, n]`. -/
theorem avgRank_mem (x : Fin n → ℝ) (i : Fin n) : 1 ≤ avgRank x i ∧ avgRank x i ≤ n := by
  have hE : 1 ≤ #{j | x j = x i} := Finset.card_pos.mpr ⟨i, by simp⟩
  have hsum : #{j | x j < x i} + #{j | x j = x i} ≤ n := by
    rw [← Finset.card_union_of_disjoint]
    · simpa using Finset.card_le_univ _
    · exact Finset.disjoint_filter.mpr fun j _ hlt heq ↦ (ne_of_lt hlt) heq
  have hE' : (1 : ℝ) ≤ #{j | x j = x i} := by exact_mod_cast hE
  have hsum' : (#{j | x j < x i} : ℝ) + #{j | x j = x i} ≤ n := by exact_mod_cast hsum
  have hL : (0 : ℝ) ≤ #{j | x j < x i} := Nat.cast_nonneg _
  unfold avgRank
  constructor <;> linarith

/-- No outlier dominates: every score lies in `[-B, B]`, `B = g(1 - 1/(2n))` (`Φ⁻¹(1 - 1/(2n))`: 2.58 at
`n = 100`, 2.77 at `n = 150`), however extreme the raw value. -/
theorem normalScore_bounded {g : ℝ → ℝ} (hg : Monotone g) (hsym : ∀ p, g (1 - p) = -g p)
    (x : Fin n → ℝ) (i : Fin n) :
    |normalScore g x i| ≤ g (1 - 1 / (2 * n)) := by
  have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  obtain ⟨h1, hn'⟩ := avgRank_mem x i
  have hlo : 1 / (2 * n) ≤ (avgRank x i - 1 / 2) / n := by
    rw [div_le_div_iff₀ (by positivity) hn]; nlinarith
  have hhi : (avgRank x i - 1 / 2) / n ≤ 1 - 1 / (2 * n) := by
    rw [div_le_iff₀ hn]; field_simp; nlinarith
  have hsym' := hsym (1 / (2 * n))
  have hmono := hg hlo
  unfold normalScore
  rw [abs_le]
  exact ⟨by linarith, hg hhi⟩

/-- Centring without ties: the probability points `(i + 1/2)/n` are symmetric about `1/2`, so for any `g` with
`g(1 - p) = -g(p)` the month's scores sum to zero. -/
theorem sum_symmetric_points_eq_zero {g : ℝ → ℝ} (hsym : ∀ p, g (1 - p) = -g p) (n : ℕ) :
    ∑ i ∈ range n, g ((i + 1 / 2) / n) = 0 := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp
  have hn' : (n : ℝ) ≠ 0 := by positivity
  have hreflect : ∑ i ∈ range n, g ((i + 1 / 2) / n)
      = ∑ i ∈ range n, g ((((n - 1 - i : ℕ) : ℝ) + 1 / 2) / n) :=
    (Finset.sum_range_reflect (fun i ↦ g ((i + 1 / 2) / n)) n).symm
  have hpoint : ∀ i ∈ range n, g ((((n - 1 - i : ℕ) : ℝ) + 1 / 2) / n) = -g ((i + 1 / 2) / n) := by
    intro i hi
    have hi' : i ≤ n - 1 := Nat.le_sub_one_of_lt (Finset.mem_range.mp hi)
    have hcast : ((n - 1 - i : ℕ) : ℝ) = n - 1 - i := by
      rw [Nat.cast_sub hi', Nat.cast_sub (by omega)]; simp
    rw [hcast, ← hsym]
    congr 1
    field_simp
    ring
  rw [Finset.sum_congr rfl hpoint, Finset.sum_neg_distrib] at hreflect
  linarith

end

end QpoFormal
