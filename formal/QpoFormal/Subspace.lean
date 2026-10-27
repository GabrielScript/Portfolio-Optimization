/-
Why the QPO's QAOA needs no cardinality penalty, and why `qpo_kaggle/qaoa_subspace.py` may simulate only the
`C(n, k)` amplitudes of Hamming weight `k`: the Dicke start lives on the weight-`k` bitstrings, the cost layer is
diagonal, and each XY-mixer gate only mixes `|01⟩` and `|10⟩` of its pair, which swaps two bits and keeps the
weight. So every state of the circuit, and every measured bitstring, has exactly `k` ones.

A state is an amplitude function on bitstrings `Fin n → Bool` (bit `i` = name `i` picked); no tensor products are
needed for a support statement. PennyLane's `IsingXY(φ)` is the gate `xyGate a b c s` with `c = cos(φ/2)` and
`s = i sin(φ/2)`; the theorems hold for every `c, s`.
-/
import Mathlib

open Finset

namespace QpoFormal

noncomputable section

variable {n : ℕ}

/-- An `n`-qubit state: one complex amplitude per bitstring. -/
abbrev State (n : ℕ) := (Fin n → Bool) → ℂ

/-- Hamming weight: the number of picked names. -/
def weight (y : Fin n → Bool) : ℕ := (univ.filter fun i ↦ y i = true).card

/-- `ψ` has no amplitude outside the weight-`k` bitstrings. -/
def SupportedOn (k : ℕ) (ψ : State n) : Prop := ∀ y, weight y ≠ k → ψ y = 0

/-- `f` maps states supported on weight `k` to states supported on weight `k`. -/
def Preserves (k : ℕ) (f : State n → State n) : Prop := ∀ ψ, SupportedOn k ψ → SupportedOn k (f ψ)

/-- The bitstring with bits `a` and `b` exchanged. -/
def swapBits (a b : Fin n) (y : Fin n → Bool) : Fin n → Bool := fun i ↦ y (Equiv.swap a b i)

/-- Exchanging two bits keeps the Hamming weight. -/
theorem weight_swapBits (a b : Fin n) (y : Fin n → Bool) : weight (swapBits a b y) = weight y := by
  unfold weight swapBits
  exact Finset.card_bij' (fun i _ ↦ Equiv.swap a b i) (fun i _ ↦ Equiv.swap a b i)
    (by simp) (by simp) (by simp) (by simp)

/-- A diagonal layer (the cost layer `exp(-iγ H_C)` is one: `d y = exp(-iγ E(y))`). -/
def diagLayer (d : (Fin n → Bool) → ℂ) (ψ : State n) : State n := fun y ↦ d y * ψ y

/-- The XY gate on the pair `(a, b)`: identity on `|00⟩, |11⟩`; on `{|01⟩, |10⟩}` the matrix `[[c, s], [s, c]]`. -/
def xyGate (a b : Fin n) (c s : ℂ) (ψ : State n) : State n :=
  fun y ↦ if y a = y b then ψ y else c * ψ y + s * ψ (swapBits a b y)

theorem diagLayer_preserves (k : ℕ) (d : (Fin n → Bool) → ℂ) : Preserves k (diagLayer d) := by
  intro ψ hψ y hy
  simp [diagLayer, hψ y hy]

theorem xyGate_preserves (k : ℕ) (a b : Fin n) (c s : ℂ) : Preserves k (xyGate a b c s) := by
  intro ψ hψ y hy
  unfold xyGate
  split_ifs
  · exact hψ y hy
  · simp [hψ y hy, hψ (swapBits a b y) (by rwa [weight_swapBits])]

theorem Preserves.comp {k : ℕ} {f g : State n → State n} (hf : Preserves k f) (hg : Preserves k g) :
    Preserves k (f ∘ g) := by
  intro ψ hψ
  exact hf _ (hg ψ hψ)

/-- Any circuit built from weight-preserving gates (applied left to right) keeps the support. -/
theorem foldl_preserves (k : ℕ) (gates : List (State n → State n))
    (hg : ∀ g ∈ gates, Preserves k g) :
    Preserves k (fun ψ ↦ gates.foldl (fun φ g ↦ g φ) ψ) := by
  unfold Preserves
  induction gates with
  | nil => intro ψ hψ; simpa using hψ
  | cons g gs ih =>
    intro ψ hψ
    simp only [List.foldl_cons]
    exact ih (fun g' hg' ↦ hg g' (List.mem_cons_of_mem g hg')) (g ψ)
      (hg g List.mem_cons_self ψ hψ)

/-- The Dicke state `|D(n, k)⟩`: uniform on the weight-`k` bitstrings. -/
noncomputable def dicke (n k : ℕ) : State n :=
  fun y ↦ if weight y = k then ((Real.sqrt (Nat.choose n k : ℝ))⁻¹ : ℂ) else 0

theorem dicke_supported (k : ℕ) : SupportedOn k (dicke n k) := by
  intro y hy
  simp [dicke, hy]

/-- The QAOA of the QPO: from `|D(n, k)⟩`, any sequence of diagonal layers and XY gates (any angles, any ring
order, any depth `p`) ends on a state with no amplitude outside weight `k`. -/
theorem qaoa_supported (k : ℕ) (gates : List (State n → State n))
    (hg : ∀ g ∈ gates, (∃ d, g = diagLayer d) ∨ ∃ a b c s, g = xyGate a b c s) :
    SupportedOn k (gates.foldl (fun φ g ↦ g φ) (dicke n k)) := by
  refine foldl_preserves k gates (fun g hg' ↦ ?_) _ (dicke_supported k)
  rcases hg g hg' with ⟨d, rfl⟩ | ⟨a, b, c, s, rfl⟩
  · exact diagLayer_preserves k d
  · exact xyGate_preserves k a b c s

/-- Hence a measurement never returns an infeasible portfolio: the Born probability `|ψ(y)|²` of every bitstring
with `weight y ≠ k` is `0` (the `n_infeasible = 0` the solvers check). -/
theorem qaoa_never_infeasible (k : ℕ) (gates : List (State n → State n))
    (hg : ∀ g ∈ gates, (∃ d, g = diagLayer d) ∨ ∃ a b c s, g = xyGate a b c s)
    (y : Fin n → Bool) (hy : weight y ≠ k) :
    Complex.normSq (gates.foldl (fun φ g ↦ g φ) (dicke n k) y) = 0 := by
  rw [qaoa_supported k gates hg y hy]
  simp

/-- There are `C(n, k)` bitstrings of weight `k`: the subspace the solver simulates. -/
theorem card_weight_eq (n k : ℕ) :
    (univ.filter fun y : Fin n → Bool ↦ weight y = k).card = Nat.choose n k := by
  sorry

end

end QpoFormal
