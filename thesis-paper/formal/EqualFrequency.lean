import Std

namespace MinPI.EqualFrequency

/- Each row contains the two features' bits in one restoration.
   The two off-diagonal counts measure exposures in each direction. -/
def ones (bits : List (Bool × Bool)) (read : Bool × Bool → Bool) : Nat :=
  match bits with
  | [] => 0
  | x :: xs => (if read x then 1 else 0) + ones xs read

theorem balance (bits : List (Bool × Bool)) :
    ones bits Prod.fst + bits.count (false, true) =
    ones bits Prod.snd + bits.count (true, false) := by
  induction bits with
  | nil => rfl
  | cons row rows ih =>
    obtain ⟨a, b⟩ := row
    cases a <;> cases b <;> simp [ones] at * <;> omega

theorem equal_margins (bits : List (Bool × Bool))
    (same : ones bits Prod.fst = ones bits Prod.snd) :
    bits.count (true, false) = bits.count (false, true) := by
  have h := balance bits
  omega

theorem pair_missed_iff_same_signature (bits : List (Bool × Bool))
    (same : ones bits Prod.fst = ones bits Prod.snd) :
    (true, false) ∉ bits ↔ ∀ row ∈ bits, row.1 = row.2 := by
  constructor
  · intro no row hr
    have hz : bits.count (true, false) = 0 := List.count_eq_zero.mpr no
    have other : bits.count (false, true) = 0 := (equal_margins bits same).symm.trans hz
    have hn : (false, true) ∉ bits := List.count_eq_zero.mp other
    obtain ⟨a, b⟩ := row
    cases a <;> cases b <;> simp_all
  · intro h hr
    have contra := h (true, false) hr
    cases contra

#print axioms pair_missed_iff_same_signature
end MinPI.EqualFrequency
