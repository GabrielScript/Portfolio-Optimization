/-
Sizes behind the solver choices (DECISOES 145, 148): the weight-k subspace of the reading (`n = 20, k = 15`) has
`C(20, 15) = 15,504` states, within `torch.multinomial`'s `2^24` categories (`MAX_SUBSPACE` in
`qpo_kaggle/qaoa_subspace.py`), while `n = 28, k = 15` has `C(28, 15) = 37,442,160 > 2^24`. The feasible
portfolios are exactly the `k`-subsets, so there are `C(n, k)` of them.
-/
import Mathlib

namespace QpoFormal

theorem choose_20_15 : Nat.choose 20 15 = 15504 := by
  rw [Nat.choose_eq_factorial_div_factorial (by norm_num)]
  norm_num [Nat.factorial]

theorem choose_28_15 : Nat.choose 28 15 = 37442160 := by
  rw [Nat.choose_eq_factorial_div_factorial (by norm_num)]
  norm_num [Nat.factorial]

theorem reading_fits_sampler : Nat.choose 20 15 ≤ 2 ^ 24 := by
  rw [choose_20_15]; norm_num

theorem n28_exceeds_sampler : 2 ^ 24 < Nat.choose 28 15 := by
  rw [choose_28_15]; norm_num

/-- The feasible portfolios (`k` of `n` names) are counted by `C(n, k)`. -/
theorem card_feasible (n k : ℕ) :
    ((Finset.univ : Finset (Fin n)).powersetCard k).card = Nat.choose n k := by
  rw [Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin]

end QpoFormal
