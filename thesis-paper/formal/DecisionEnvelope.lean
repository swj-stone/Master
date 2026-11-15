import Std

/-!
The choices, savings and weights are fixed before an outer scorer inspects labels.
Equal returned supports on one split are the same fitted model with the same outer passage.
The bounds allow the passage bits of distinct supports to vary independently. Shared outer
labels can further restrict those bits.
-/

namespace MinPI.DecisionEnvelope

structure Fold (α : Type) where
  left : α
  right : α
  weight : Rat
  weight_nonneg : 0 ≤ weight
  save : α → Rat
  save_nonneg : ∀ s, 0 ≤ save s
  pass : α → Bool

def utility (f : Fold α) (s : α) : Rat := if f.pass s then f.save s else 0

def diff (f : Fold α) : Rat := f.weight * (utility f f.left - utility f f.right)

def lower [DecidableEq α] (f : Fold α) : Rat :=
  if f.left = f.right then 0 else -(f.weight * f.save f.right)

def upper [DecidableEq α] (f : Fold α) : Rat :=
  if f.left = f.right then 0 else f.weight * f.save f.left

theorem utility_bounds (f : Fold α) (s : α) :
    0 ≤ utility f s ∧ utility f s ≤ f.save s := by
  unfold utility
  split <;> grind [f.save_nonneg s]

theorem fold_bounds [DecidableEq α] (f : Fold α) :
    lower f ≤ diff f ∧ diff f ≤ upper f := by
  unfold lower upper diff
  by_cases h : f.left = f.right
  · simp only [h, ite_true]
    grind
  · simp only [h, ite_false]
    have l := utility_bounds f f.left
    have r := utility_bounds f f.right
    have lo := Rat.mul_le_mul_of_nonneg_left
      (show -(f.save f.right) ≤ utility f f.left - utility f f.right by grind) f.weight_nonneg
    have hi := Rat.mul_le_mul_of_nonneg_left
      (show utility f f.left - utility f f.right ≤ f.save f.left by grind) f.weight_nonneg
    grind

theorem study_bounds [DecidableEq α] (folds : List (Fold α)) :
    (folds.map lower).sum ≤ (folds.map diff).sum ∧
    (folds.map diff).sum ≤ (folds.map upper).sum := by
  induction folds with
  | nil => simp
  | cons f fs ih =>
    have hf := fold_bounds f
    simp only [List.map_cons, List.sum_cons]
    grind

#print axioms study_bounds

/-- One coefficient for each fold and distinct returned support:
weight times savings times the difference in return probabilities. -/
def lotteryDiff (atoms : List (Rat × Bool)) : Rat :=
  (atoms.map fun (c, passed) ↦ if passed then c else 0).sum

theorem lottery_bounds (atoms : List (Rat × Bool)) :
    (atoms.map fun x ↦ min 0 x.1).sum ≤ lotteryDiff atoms ∧
    lotteryDiff atoms ≤ (atoms.map fun x ↦ max 0 x.1).sum := by
  induction atoms with
  | nil => simp [lotteryDiff]
  | cons head tail ih =>
    rcases head with ⟨c, b⟩
    simp only [lotteryDiff, List.map_cons, List.sum_cons] at *
    cases b <;> simp <;> grind

#print axioms lottery_bounds

end MinPI.DecisionEnvelope
