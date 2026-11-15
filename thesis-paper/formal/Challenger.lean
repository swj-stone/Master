import Std

namespace MinPI.Challenger

/-- A candidate's worst count starts at the panel count and falls as we inspect contexts. -/
theorem remaining_cannot_raise (remaining : List Nat) (observed : Nat) :
    remaining.foldl Nat.min observed ≤ observed := by
  induction remaining generalizing observed with
  | nil => exact Nat.le_refl _
  | cons vote more ih =>
    exact Nat.le_trans (ih (min observed vote)) (Nat.min_le_left _ _)

/-- Under a strict improvement rule, a tied partial minimum can also be pruned.
    The incumbent already wins the tie by the fixed order. -/
theorem pruned_cannot_win (remaining : List Nat) (observed incumbent : Nat)
    (stop : observed ≤ incumbent) :
    ¬ incumbent < remaining.foldl Nat.min observed := by
  have := remaining_cannot_raise remaining observed
  omega

#print axioms pruned_cannot_win
end MinPI.Challenger
