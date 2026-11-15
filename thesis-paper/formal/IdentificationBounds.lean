import Std

/-!
Sharp bounds for a two-score refit observed only through its unpaired positive-score mass.
The rational statements apply directly to empirical frequencies on a finite evaluation set.
-/

namespace MinPI.IdentificationBounds

/-- The training proportions yield a pure split on the first coordinate and a mixed split
on the second. Impurity uses the positive-class fraction in a binary leaf. -/
def gini (x : Rat) : Rat := 2 * x * (1 - x)

theorem training_splits :
    (gini 0 + gini 1) / 2 < (gini (1/4) + gini (3/4)) / 2 ∧
    (gini (1/4) + gini (3/4)) / 2 < gini (1/2) := by
  unfold gini
  grind

/-- A frozen stump using only the first coordinate has identical scores for every
assignment to the second, including when either input is masked. -/
def frozen (showD showR : Bool) (background d r : Bool) : Bool :=
  let masked := (if showD then d else background, if showR then r else background)
  masked.1

theorem pointwise_mask_invariance (showD showR background d r s : Bool) :
    frozen showD showR background d r = frozen showD showR background d s := rfl

/-- The positive-label margin is p, the large-score margin is q, and their overlap is u. -/
def HasTable (p q u : Rat) : Prop :=
  ∃ fn fp tn : Rat, 0 ≤ u ∧ 0 ≤ fn ∧ 0 ≤ fp ∧ 0 ≤ tn ∧
    u + fn = p ∧ u + fp = q ∧ u + fn + fp + tn = 1

theorem sharp_overlap (p q u : Rat) :
    HasTable p q u ↔ max 0 (p + q - 1) ≤ u ∧ u ≤ min p q := by
  constructor
  · rintro ⟨fn, fp, tn, _, _, _, _, _, _, _⟩
    grind
  · intro h
    exact ⟨p-u, q-u, 1-p-q+u, by grind⟩

/-- Twice the unconditioned credit on a positive-negative pair: twice the wins
plus ties. Divide by twice the positive-negative mass to obtain AUC. -/
def pairCredit (p q u : Rat) : Rat :=
  let fn := p-u
  let fp := q-u
  let tn := 1-p-q+u
  2*u*tn + u*fp + fn*tn

def correct (p q u : Rat) : Rat := u + (1-p-q+u)

theorem affine_metrics (p q u : Rat) :
    pairCredit p q u = p*(1-p) + u - p*q ∧ correct p q u = 1-p-q+2*u := by
  unfold pairCredit correct
  grind

/-- Joint pass with two possibly different floors. For 0 < p < 1 the pair inequality
is the AUC floor after multiplication by twice the positive-negative mass. -/
def Pass (p q u aucFloor accFloor : Rat) : Prop :=
  2*p*(1-p)*aucFloor ≤ pairCredit p q u ∧ accFloor ≤ correct p q u

theorem pass_monotone {p q u v a c : Rat} (huv : u ≤ v) (hu : Pass p q u a c) :
    Pass p q v a c := by
  unfold Pass at *
  obtain ⟨hp, hc⟩ := affine_metrics p q u
  obtain ⟨hp', hc'⟩ := affine_metrics p q v
  grind

/-- The lower endpoint decides universal passage and the upper endpoint decides possible
passage, including the equality boundary in the tolerance. -/
theorem sharp_decisions (p q a c : Rat) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) :
    ((∀ u, HasTable p q u → Pass p q u a c) ↔ Pass p q (max 0 (p+q-1)) a c) ∧
    ((∃ u, HasTable p q u ∧ Pass p q u a c) ↔ Pass p q (min p q) a c) := by
  have endpoints : max 0 (p+q-1) ≤ min p q := by grind
  have lo : HasTable p q (max 0 (p+q-1)) := (sharp_overlap ..).2 ⟨by grind, endpoints⟩
  have hi : HasTable p q (min p q) := (sharp_overlap ..).2 ⟨endpoints, by grind⟩
  refine ⟨⟨fun h ↦ h _ lo, ?_⟩, ⟨?_, fun h ↦ ⟨_, hi, h⟩⟩⟩
  · intro h u hu
    exact pass_monotone ((sharp_overlap ..).1 hu).1 h
  · rintro ⟨u, hu, h⟩
    exact pass_monotone ((sharp_overlap ..).1 hu).2 h

theorem balanced_range (s : Rat) (hs : 0 ≤ s ∧ s ≤ 1) :
    HasTable (1/2) (1/2) (s/2) ∧
    pairCredit (1/2) (1/2) (s/2) = s/2 ∧ correct (1/2) (1/2) (s/2) = s := by
  rw [sharp_overlap]
  unfold pairCredit correct
  grind

/-- On an ambiguous observation, vetoing with probability z incurs one of these two
expected costs. Equalizing them attains the minimax bound for positive error costs. -/
theorem minimax_at_equalizer (alarm miss z : Rat) (ha : 0 < alarm) (hm : 0 < miss) :
    let e := miss / (alarm+miss)
    0 ≤ e ∧ e ≤ 1 ∧ alarm*e = miss*(1-e) ∧
      alarm*e ≤ max (alarm*z) (miss*(1-z)) := by
  dsimp
  have heq := Rat.div_mul_cancel (a := miss) (b := alarm+miss) (by grind)
  have pos : 0 < miss / (alarm+miss) := (Rat.lt_div_iff (by grind)).2 (by grind)
  have upper : miss / (alarm+miss) < 1 := (Rat.div_lt_iff (by grind)).2 (by grind)
  have equal : alarm * (miss/(alarm+miss)) = miss * (1-miss/(alarm+miss)) := by grind
  refine ⟨by grind, by grind, equal, ?_⟩
  by_cases hz : z < miss/(alarm+miss)
  · have lt := Rat.mul_lt_mul_of_pos_left (show 1-miss/(alarm+miss) < 1-z by grind) hm
    grind
  · have le := Rat.mul_le_mul_of_nonneg_left (show miss/(alarm+miss) ≤ z by grind)
      (show 0 ≤ alarm by grind)
    grind

#print axioms training_splits
#print axioms pointwise_mask_invariance
#print axioms sharp_overlap
#print axioms affine_metrics
#print axioms sharp_decisions
#print axioms balanced_range
#print axioms minimax_at_equalizer

end MinPI.IdentificationBounds
