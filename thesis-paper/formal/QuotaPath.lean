import Std

/-!
# Anchored quota votes on a finite gain grid

One integer grid step can represent any fixed unit of loss improvement. The reference level
`anchor` offsets negative gains. At threshold `t` a context with level `z` casts the
marginal ballot `1[t ≤ z] − 1[t ≤ anchor]`. Minimizing the ballot across contexts and
summing over thresholds recovers the smallest signed gain on the grid. This depends on using
a common anchor. Coalition-specific unshifted reference levels give a different game.
-/
namespace MinPI.QuotaPath

def centeredBallot (threshold level anchor : Nat) : Int :=
  (if threshold ≤ level then 1 else 0) - (if threshold ≤ anchor then 1 else 0)

/-- The value at an actual minimizing context is also the least centered ballot. -/
theorem ballot_minimum {H : Type} (level : H → Nat) (worst : H)
    (minimal : ∀ h, level worst ≤ level h) (threshold anchor : Nat) :
    ∀ h, centeredBallot threshold (level worst) anchor ≤ centeredBallot threshold (level h) anchor := by
  intro h
  have := minimal h
  unfold centeredBallot
  split <;> split <;> split <;> omega

def countVotes (level : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => countVotes level n + (if n + 1 ≤ level then 1 else 0)

theorem count_votes (level width : Nat) : countVotes level width = min level width := by
  induction width with
  | zero => simp [countVotes]
  | succ w ih =>
    rw [countVotes, ih]
    split <;> omega

def sumBallots (level anchor : Nat) : Nat → Int
  | 0 => 0
  | n + 1 => sumBallots level anchor n + centeredBallot (n + 1) level anchor

theorem sum_ballots (level anchor width : Nat) :
    sumBallots level anchor width = (countVotes level width : Int) - countVotes anchor width := by
  induction width with
  | zero => rfl
  | succ w ih =>
    rw [sumBallots, ih]
    simp only [countVotes, centeredBallot]
    split <;> split <;> omega

/-- At every sampled quota the same minimizing context attains the minimum. Their sum is
the signed minimum gain, measured in grid steps, provided both levels lie inside the grid. -/
theorem anchored_grid {H : Type} (level : H → Nat) (worst : H)
    (minimal : ∀ h, level worst ≤ level h) (anchor width : Nat)
    (level_in : level worst ≤ width) (anchor_in : anchor ≤ width) :
    (∀ t h, centeredBallot t (level worst) anchor ≤ centeredBallot t (level h) anchor) ∧
    (sumBallots (level worst) anchor width = (level worst : Int) - anchor) := by
  constructor
  · intro t
    exact ballot_minimum level worst minimal t anchor
  · simp [sum_ballots, count_votes, Nat.min_eq_left level_in, Nat.min_eq_left anchor_in]

/-- Larger scores win. The smaller feature ID wins an exact tie, regardless of visit order. -/
def beats (score : Int) (id : Nat) (otherScore : Int) (otherId : Nat) : Prop :=
  otherScore < score ∨ (otherScore = score ∧ id < otherId)

/-- The upper bound furnished by a queried context may be tested against a fully known winner.
This applies to arbitrary panel visit orders, including a late candidate with an earlier ID. -/
theorem prune_with_priority (score upper incumbent : Int) (id incumbentId : Nat)
    (observed : score ≤ upper) (stop : ¬ beats upper id incumbent incumbentId) :
    ¬ beats score id incumbent incumbentId := by
  unfold beats at *
  omega

/-- A partially read candidate cannot be discarded at an equal upper bound if it wins ties:
an unobserved completion can attain the bound, even if other completions cannot. -/
theorem equality_can_win (level : Int) (id incumbentId : Nat) (earlier : id < incumbentId) :
    beats level id level incumbentId := Or.inr ⟨rfl, earlier⟩

def prefixSum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => prefixSum f n + f (n + 1)

private theorem sum_nonpos (f : Nat → Int) (n : Nat) (h : ∀ t, 0 < t → t ≤ n → f t ≤ 0) :
    prefixSum f n ≤ 0 := by
  induction n with
  | zero => simp [prefixSum]
  | succ n ih =>
    have prev := ih fun t ht htn ↦ h t ht (by omega)
    have last := h (n+1) (by omega) (by omega)
    simp [prefixSum]
    omega

private theorem sum_neg (f : Nat → Int) (n : Nat) (h : ∀ t, 0 < t → t ≤ n → f t ≤ 0)
    (hit : ∃ t, 0 < t ∧ t ≤ n ∧ f t < 0) : prefixSum f n < 0 := by
  induction n with
  | zero => obtain ⟨_, _, _, _⟩ := hit; omega
  | succ n ih =>
    have prev := sum_nonpos f n fun t ht htn ↦ h t ht (by omega)
    have last := h (n+1) (by omega) (by omega)
    obtain ⟨t, ht, htn, neg⟩ := hit
    by_cases eq : t = n+1
    · subst t; simp only [prefixSum]; omega
    · have negPrev := ih (fun s hs hsn ↦ h s hs (by omega)) ⟨t, ht, by omega, neg⟩
      simp only [prefixSum]
      omega

/-- For arbitrary base levels, one declining context suffices for a strictly negative total
vote. The conclusion persists under adding any number of further contexts. -/
theorem downward_witness {H : Type} (before after : H → Nat) (worst : Nat → Int)
    (upper : ∀ t h, worst t ≤ centeredBallot t (after h) (before h))
    (width : Nat) (w : H) (drop : after w < before w) (within : before w ≤ width) :
    prefixSum worst width < 0 := by
  apply sum_neg
  · intro t _ _
    have bound := upper t w
    unfold centeredBallot at bound
    split at bound <;> split at bound <;> omega
  · refine ⟨after w + 1, by omega, by omega, ?_⟩
    have bound := upper (after w + 1) w
    unfold centeredBallot at bound
    split at bound <;> split at bound <;> omega

private theorem sum_nonneg (f : Nat → Int) (n : Nat) (h : ∀ t, 0 < t → t ≤ n → 0 ≤ f t) :
    0 ≤ prefixSum f n := by
  induction n with
  | zero => simp [prefixSum]
  | succ n ih =>
    have prev := ih fun t ht htn ↦ h t ht (by omega)
    have last := h (n+1) (by omega) (by omega)
    simp only [prefixSum]
    omega

/-- For integer endpoints on the grid, the negative decision is equivalent to the existence
of a single strictly deteriorating context. The minimum must be attained at each threshold. -/
theorem negative_iff_decline {H : Type} (before after : H → Nat) (worst : Nat → Int)
    (upper : ∀ t h, worst t ≤ centeredBallot t (after h) (before h))
    (reached : ∀ t, ∃ h, worst t = centeredBallot t (after h) (before h))
    (width : Nat) (within : ∀ h, before h ≤ width) :
    prefixSum worst width < 0 ↔ ∃ h, after h < before h := by
  constructor
  · intro negative
    apply Classical.byContradiction
    intro none
    have pos := sum_nonneg worst width fun t _ _ ↦ by
      obtain ⟨h, same⟩ := reached t
      have rising : before h ≤ after h := by
        have nodrop : ¬ after h < before h := fun q ↦ none ⟨h, q⟩
        omega
      rw [same]
      unfold centeredBallot
      split <;> split <;> omega
    omega
  · rintro ⟨h, drop⟩
    exact downward_witness before after worst upper width h drop (within h)

#print axioms anchored_grid
#print axioms prune_with_priority
#print axioms equality_can_win
#print axioms negative_iff_decline
end MinPI.QuotaPath
