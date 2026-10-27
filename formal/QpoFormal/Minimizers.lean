/-
Transformations that move no minimizer: a constant shift (the Ising constant, the turnover term's
`τ|h|`) and a positive scale (the cost Hamiltonian is the Ising form divided by
`ising_scale > 0`), on any feasible set, e.g. the slice `|x| = k`.
-/
import Mathlib

namespace QpoFormal

/-- A constant shift moves no minimizer. -/
theorem isMinOn_add_const {α : Type*} (f : α → ℝ) (c : ℝ) (S : Set α) (a : α) :
    IsMinOn (fun y ↦ f y + c) S a ↔ IsMinOn f S a := by
  simp only [isMinOn_iff, add_le_add_iff_right]

/-- A positive scale moves no minimizer. -/
theorem isMinOn_div_pos {α : Type*} (f : α → ℝ) {c : ℝ} (hc : 0 < c) (S : Set α) (a : α) :
    IsMinOn (fun y ↦ f y / c) S a ↔ IsMinOn f S a := by
  simp only [isMinOn_iff, div_le_div_iff_of_pos_right hc]

/-- Both at once: `(f + c) / s` with `s > 0` has the minimizers of `f`. -/
theorem isMinOn_affine {α : Type*} (f : α → ℝ) (c : ℝ) {s : ℝ} (hs : 0 < s) (S : Set α) (a : α) :
    IsMinOn (fun y ↦ (f y + c) / s) S a ↔ IsMinOn f S a :=
  (isMinOn_div_pos (fun y ↦ f y + c) hs S a).trans (isMinOn_add_const f c S a)

end QpoFormal
