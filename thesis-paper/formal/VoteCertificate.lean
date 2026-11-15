import Std

/-! Certificates for the earliest maximizer of a table of bounded vote counts.
The completion model permits the unqueried counts to vary independently. -/

namespace MinPI.VoteCertificate

/-- The winner must beat every earlier row strictly and may tie a later row. -/
def FirstMaximum {n : Nat} (score : Fin n → Nat) (w : Fin n) : Prop :=
  (∀ i, i < w → score i < score w) ∧ (∀ i, w < i → score i ≤ score w)

/-- All row minima may take values in their observed intervals independently. -/
def Compatible {n : Nat} (lower upper score : Fin n → Nat) : Prop :=
  ∀ i, lower i ≤ score i ∧ score i ≤ upper i

/-- With any ballot unread, zero remains attainable. An exhausted row has its exact score. -/
def lowerBound (upper unread : Nat) : Nat := if unread = 0 then upper else 0

private theorem minimum_le_observed (entries : List Nat) (u : Nat) :
    entries.foldl Nat.min u ≤ u := by
  induction entries generalizing u with
  | nil => exact Nat.le_refl _
  | cons x xs ih => exact Nat.le_trans (ih (min u x)) (Nat.min_le_left _ _)

private theorem constant_minimum (k x : Nat) :
    (List.replicate k x).foldl Nat.min x = x := by
  induction k with
  | zero => rfl
  | succ k ih => simpa [List.replicate_succ] using ih

/-- `upper` is the minimum observed so far, initialized to the panel count for an unread row. -/
def Attainable {n : Nat} (panels : Nat) (upper unread score : Fin n → Nat) : Prop :=
  ∃ entries : Fin n → List Nat, ∀ i,
    (entries i).length = unread i ∧ (∀ x ∈ entries i, x ≤ panels) ∧
    (entries i).foldl Nat.min (upper i) = score i

/-- The interval for every partially observed row is exact, simultaneously over all rows. -/
theorem attainable_iff {n : Nat} (panels : Nat) (upper unread score : Fin n → Nat)
    (bounded : ∀ i, upper i ≤ panels) :
    Attainable panels upper unread score ↔
      Compatible (fun i => lowerBound (upper i) (unread i)) upper score := by
  constructor
  · rintro ⟨entries, valid⟩ i
    obtain ⟨length_eq, _, value_eq⟩ := valid i
    constructor
    · by_cases h : unread i = 0
      · have empty : entries i = [] := List.length_eq_zero_iff.mp (length_eq.trans h)
        simpa [lowerBound, h, empty] using (Nat.le_of_eq value_eq)
      · simp [lowerBound, h]
    · rw [← value_eq]
      exact minimum_le_observed _ _
  · intro h
    refine ⟨fun i => List.replicate (unread i) (score i), ?_⟩
    intro i
    obtain ⟨lo, hi⟩ := h i
    refine ⟨List.length_replicate .., ?_, ?_⟩
    · intro x hx
      have same : x = score i := (List.mem_replicate.mp hx).2
      exact same ▸ Nat.le_trans hi (bounded i)
    · change (List.replicate (unread i) (score i)).foldl Nat.min (upper i) = score i
      cases hlen : unread i with
      | zero =>
        have rev : upper i ≤ score i := by simpa [lowerBound, hlen] using lo
        exact Nat.le_antisymm rev hi
      | succ k =>
        simpa [List.replicate_succ, Nat.min_eq_right hi] using constant_minimum k (score i)

/-- A partial table forces a winner exactly when even its most unfavorable completion does. -/
theorem certificate_iff {n : Nat} (lower upper : Fin n → Nat) (w : Fin n)
    (ordered : ∀ i, lower i ≤ upper i) :
    (∀ score, Compatible lower upper score → FirstMaximum score w) ↔
      (∀ i, i < w → upper i < lower w) ∧ (∀ i, w < i → upper i ≤ lower w) := by
  constructor
  · intro forced
    let adverse : Fin n → Nat := fun i => if i = w then lower i else upper i
    have allowed : Compatible lower upper adverse := by
      intro i
      by_cases hi : i = w <;> simp [adverse, hi, ordered]
    obtain ⟨earlier, later⟩ := forced adverse allowed
    constructor
    · intro i hi
      have hne : i ≠ w := fun h => by subst i; exact (Nat.lt_irrefl _) hi
      simpa [adverse, hne] using earlier i hi
    · intro i hi
      have hne : i ≠ w := fun h => by subst i; exact (Nat.lt_irrefl _) hi
      simpa [adverse, hne] using later i hi
  · rintro ⟨earlier, later⟩ score allowed
    constructor
    · intro i hi
      exact Nat.lt_of_lt_of_le (Nat.lt_of_le_of_lt (allowed i).2 (earlier i hi)) (allowed w).1
    · intro i hi
      exact Nat.le_trans (Nat.le_trans (allowed i).2 (later i hi)) (allowed w).1

#print axioms certificate_iff

/-- The ballot-level statement follows by filling each unread row with a constant vote count. -/
theorem ballot_certificate_iff {n : Nat} (panels : Nat) (upper unread : Fin n → Nat) (w : Fin n)
    (bounded : ∀ i, upper i ≤ panels) :
    (∀ score, Attainable panels upper unread score → FirstMaximum score w) ↔
      (∀ i, i < w → upper i < lowerBound (upper w) (unread w)) ∧
      (∀ i, w < i → upper i ≤ lowerBound (upper w) (unread w)) := by
  have ordered (i : Fin n) : lowerBound (upper i) (unread i) ≤ upper i := by
    by_cases h : unread i = 0 <;> simp [lowerBound, h]
  simpa only [attainable_iff panels upper unread _ bounded] using
    certificate_iff (fun i => lowerBound (upper i) (unread i)) upper w ordered

#print axioms ballot_certificate_iff

end MinPI.VoteCertificate
