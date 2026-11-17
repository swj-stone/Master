import Std

namespace MinPI.AnchoredGuard

/-- The upper endpoint is the last accepted width. A stalled midpoint makes no change. -/
def step (u : Nat → Rat) (floor : Rat) (bounds : Nat × Nat) : Nat × Nat :=
  let (lo, hi) := bounds
  let mid := (lo + hi) / 2
  if mid ≤ lo then (lo, hi) else if floor ≤ u mid then (lo, mid) else (mid, hi)

/-- A rejected or stalled proposal retains the incumbent. An accepted proposal lies strictly
inside the current interval. No monotonicity of the observed utility is needed. -/
theorem guarded_step (u : Nat → Rat) (floor : Rat) (lo hi : Nat)
    (ordered : lo < hi) (feasible : floor ≤ u hi) :
    let next := step u floor (lo, hi)
    next.1 < next.2 ∧ next.2 ≤ hi ∧ floor ≤ u next.2 := by
  have mid_lt : (lo+hi)/2 < hi := by omega
  simp only [step]
  split <;> simp_all
  split <;> simp_all <;> omega

/-- Acceptance strictly shortens the prefix of the same ranking. In particular this
applies to the second proposal conditional on the first one having passed. -/
theorem accepted_prefix {α : Type} (ranked : List α) (u : Nat → Rat) (floor : Rat)
    (lo hi : Nat) (inside : hi ≤ ranked.length)
    (moves : lo < (lo+hi)/2) (accept : floor ≤ u ((lo+hi)/2)) :
    let next := (step u floor (lo, hi)).2
    (ranked.take next).length < (ranked.take hi).length ∧
      ranked.take next ⊆ ranked.take hi := by
  have strict : (lo+hi)/2 < hi := by omega
  simp only [step, ite_eq_right (Nat.not_le_of_lt moves), ite_eq_left accept]
  exact ⟨by rw [List.length_take_of_le (by omega), List.length_take_of_le inside]; omega,
    List.take_subset_take_left _ (by omega)⟩

/-- Model the effective floating-point cutoff as its stored rational value. At entry the
entire ranked list is a permutation of the feasible anchor. The utility is the observed
utility for the sorted feature set of each prefix (including the full anchor). -/
theorem two_proposals {α : Type} (ranked anchor : List α) (order : ranked.Perm anchor)
    (u : Nat → Rat) (floor : Rat) (nonempty : 0 < ranked.length)
    (feasible : floor ≤ u ranked.length) :
    let first := step u floor (0, ranked.length)
    let final := (step u floor first).2
    floor ≤ u final ∧ 0 < final ∧ (ranked.take final).length ≤ anchor.length ∧
      ranked.take final ⊆ anchor := by
  dsimp only
  obtain ⟨ordered, bounded, passed⟩ := guarded_step u floor 0 ranked.length nonempty feasible
  generalize h : step u floor (0, ranked.length) = pair at *
  obtain ⟨lo, hi⟩ := pair
  obtain ⟨inside, shrinks, valid⟩ := guarded_step u floor lo hi ordered passed
  refine ⟨valid, by omega, ?_, ?_⟩
  · exact Nat.le_trans (List.length_take_le' _ _) (Nat.le_of_eq order.length_eq)
  · exact List.Subset.trans (List.take_subset _ _) order.subset

#print axioms guarded_step
#print axioms accepted_prefix
#print axioms two_proposals

end MinPI.AnchoredGuard
