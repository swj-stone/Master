import Std

namespace MinPI.SplitVote

def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n (k+1) + choose n k

/-- Mark the other columns that represent one fixed voter. Each list entry represents a
distinct physical column, so repeated Boolean labels retain their positional multiplicity. -/
def panels : List Bool → Nat → List (List Bool)
  | _, 0 => [[]]
  | [], _ + 1 => []
  | b :: bs, k + 1 => panels bs (k+1) ++ (panels bs k).map (b :: ·)

def missed (xs : List Bool) : Bool := !(xs.any id)

/-- Number of equally likely size-k panels after conditioning on a candidate's absence. -/
theorem panel_count (xs : List Bool) (k : Nat) : (panels xs k).length = choose xs.length k := by
  induction xs generalizing k with
  | nil => cases k <;> rfl
  | cons b bs ih =>
    cases k with
    | zero => rfl
    | succ k => simp [panels, choose, ih]

/-- A representing candidate is pivotal exactly on panels missing all its substitutes. -/
theorem pivotal_count (xs : List Bool) (k : Nat) :
    (panels xs k).countP missed = choose (xs.countP (!·)) k := by
  induction xs generalizing k with
  | nil => cases k <;> rfl
  | cons b bs ih =>
    cases k with
    | zero => cases b <;> simp [panels, missed, choose]
    | succ k =>
      cases b <;> simp [panels, Function.comp_def, missed, choose]
      · change (panels bs (k+1)).countP missed + (panels bs k).countP missed = _
        rw [ih, ih, Nat.add_comm]
      · exact ih (k+1)

/-- In a five-column game A has a private voter of weight 5, B and C share one of weight
11, and D and E each have a private voter of weight 1. The shared voter's chance of being
pivotal omits its sibling from a size-k selection among four alternatives. -/
def shares (k : Nat) : Rat × Rat :=
  let common : Rat := (choose 3 k : Rat) / (choose 4 k : Rat)
  let total : Rat := 7 + 2 * 11 * common
  (5 / total, 11 * common / total)

theorem pooled_min_reversal :
    let (a₁,b₁) := shares 1
    let (a₂,b₂) := shares 2
    let (a₃,b₃) := shares 3
    min a₁ (min a₂ a₃) < min b₁ (min b₂ b₃) ∧ b₁+b₂+b₃ < a₁+a₂+a₃ := by decide +kernel

#print axioms panel_count
#print axioms pivotal_count
#print axioms pooled_min_reversal

end MinPI.SplitVote
