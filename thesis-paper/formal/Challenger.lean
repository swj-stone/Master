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

/-- The margin of each measured context can be replaced by its rank among the finitely
many measured margins, assigning equal ranks to ties. Both coordinates prefer larger values. -/
def better (vote margin refVote refMargin : Nat) : Prop :=
  refVote < vote ∨ (refVote = vote ∧ refMargin < margin)

/-- Refining the largest vote by a continuous minimum still permits exact early elimination.
The incumbent wins an equality under the fixed visitation order. -/
theorem pruned_refinement_cannot_win
    (votes margins : List Nat) (vote margin refVote refMargin : Nat)
    (stop : ¬ better vote margin refVote refMargin) :
    ¬ better (votes.foldl Nat.min vote) (margins.foldl Nat.min margin) refVote refMargin := by
  have hv := remaining_cannot_raise votes vote
  have hm := remaining_cannot_raise margins margin
  unfold better at stop ⊢
  omega

#print axioms pruned_refinement_cannot_win
end MinPI.Challenger
