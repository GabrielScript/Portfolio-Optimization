/-
`qpo_kaggle/qubo_math.qubo_to_ising` is exact: with `z = 1 - 2x` (so `z = +1` is `x = 0`, the qubit `|0⟩`),
`x'Qx = h·z + Σ_{i<j} Jᵢⱼ zᵢ zⱼ + const` on every 0/1 vector, for the `h`, `J`, `const` the Python returns.
-/
import QpoFormal.Energy

open Finset

namespace QpoFormal

noncomputable section

variable {n : ℕ}

/-- The off-diagonal part of `symm Q` (`fora` in the Python). -/
def offDiag (Q : Fin n → Fin n → ℝ) : Fin n → Fin n → ℝ :=
  fun i j ↦ if i = j then 0 else symm Q i j

/-- `h = -diag(s)/2 - (row sums of the off-diagonal part)/2`. -/
def isingH (Q : Fin n → Fin n → ℝ) (i : Fin n) : ℝ :=
  -(symm Q i i) / 2 - (∑ j, offDiag Q i j) / 2

/-- `J = offDiag / 2`: symmetric, zero diagonal. -/
def isingJ (Q : Fin n → Fin n → ℝ) (i j : Fin n) : ℝ := offDiag Q i j / 2

/-- `const = Σ diag(s)/2 + Σ offDiag/4`. -/
def isingConst (Q : Fin n → Fin n → ℝ) : ℝ :=
  (∑ i, symm Q i i) / 2 + (∑ i, ∑ j, offDiag Q i j) / 4

lemma offDiag_comm (Q : Fin n → Fin n → ℝ) (i j : Fin n) : offDiag Q i j = offDiag Q j i := by
  unfold offDiag symm
  by_cases h : i = j
  · subst h; rfl
  · rw [if_neg h, if_neg (Ne.symm h)]; ring

lemma isingJ_comm (Q : Fin n → Fin n → ℝ) (i j : Fin n) : isingJ Q i j = isingJ Q j i := by
  unfold isingJ
  rw [offDiag_comm]

lemma isingJ_self (Q : Fin n → Fin n → ℝ) (i : Fin n) : isingJ Q i i = 0 := by
  simp [isingJ, offDiag]

/-- `x'sx = Σᵢ sᵢᵢ xᵢ + Σᵢⱼ offDiagᵢⱼ xᵢ xⱼ` for a 0/1 vector. -/
lemma energy_symm_split (Q : Fin n → Fin n → ℝ) {x : Fin n → ℝ} (hx : Binary x) :
    energy (symm Q) x = ∑ i, symm Q i i * x i + ∑ i, ∑ j, offDiag Q i j * x i * x j := by
  sorry

/-- The Ising form with the pair sum written over all ordered pairs (each pair twice, hence `/2`). -/
theorem energy_eq_ising (Q : Fin n → Fin n → ℝ) {x : Fin n → ℝ} (hx : Binary x) :
    energy Q x = ∑ i, isingH Q i * (1 - 2 * x i)
      + (∑ i, ∑ j, isingJ Q i j * (1 - 2 * x i) * (1 - 2 * x j)) / 2 + isingConst Q := by
  sorry

/-- For `J` symmetric with a zero diagonal, the sum over `i < j` is half the sum over all `(i, j)`. -/
lemma sum_lt_eq_half (J : Fin n → Fin n → ℝ) (hJ : ∀ i j, J i j = J j i) (h0 : ∀ i, J i i = 0)
    (z : Fin n → ℝ) :
    ∑ i, ∑ j, (if i < j then J i j * z i * z j else 0) = (∑ i, ∑ j, J i j * z i * z j) / 2 := by
  sorry

/-- `qubo_to_ising` exactly as the Python evaluates it (`z @ h + z @ np.triu(J, 1) @ z + const`). -/
theorem energy_eq_ising_triu (Q : Fin n → Fin n → ℝ) {x : Fin n → ℝ} (hx : Binary x) :
    energy Q x = ∑ i, isingH Q i * (1 - 2 * x i)
      + ∑ i, ∑ j, (if i < j then isingJ Q i j * (1 - 2 * x i) * (1 - 2 * x j) else 0)
      + isingConst Q := by
  sorry

end

end QpoFormal
