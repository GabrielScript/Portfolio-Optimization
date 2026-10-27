/-
`qpo_kaggle/qubo_math.approximation_ratio`: `(E_worst - E) / (E_worst - E_best)`, and `1` when
every feasible subset has the same energy. For a feasible pick (`E_best ≤ E ≤ E_worst`) it lies in
`[0, 1]`, and it is `1` exactly at the optimum.
-/
import Mathlib

namespace QpoFormal

noncomputable section

/-- The approximation ratio of `qubo_math.approximation_ratio`. -/
noncomputable def approxRatio (e eBest eWorst : ℝ) : ℝ :=
  if eWorst = eBest then 1 else (eWorst - e) / (eWorst - eBest)

theorem approxRatio_mem_Icc {e eBest eWorst : ℝ} (h1 : eBest ≤ e) (h2 : e ≤ eWorst) :
    approxRatio e eBest eWorst ∈ Set.Icc (0 : ℝ) 1 := by
  unfold approxRatio
  split_ifs with h
  · simp
  · have hpos : 0 < eWorst - eBest := sub_pos.mpr (lt_of_le_of_ne (h1.trans h2) (Ne.symm h))
    exact ⟨div_nonneg (by linarith) hpos.le, (div_le_one hpos).mpr (by linarith)⟩

theorem approxRatio_eq_one_iff {e eBest eWorst : ℝ} (hlt : eBest < eWorst) :
    approxRatio e eBest eWorst = 1 ↔ e = eBest := by
  have hne : eWorst ≠ eBest := hlt.ne'
  unfold approxRatio
  rw [if_neg hne, div_eq_one_iff_eq (sub_ne_zero.mpr hne)]
  constructor <;> intro h <;> linarith

theorem approxRatio_eq_zero_iff {e eBest eWorst : ℝ} (hlt : eBest < eWorst) :
    approxRatio e eBest eWorst = 0 ↔ e = eWorst := by
  have hne : eWorst ≠ eBest := hlt.ne'
  unfold approxRatio
  rw [if_neg hne, div_eq_zero_iff]
  constructor
  · rintro (h | h)
    · linarith
    · exact absurd h (sub_ne_zero.mpr hne)
  · intro h
    left
    linarith

end

end QpoFormal
