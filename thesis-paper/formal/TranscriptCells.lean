import Std

/-!
Row compatibility is the image of a fixed refit on the input fibre of a transcript.
Rowwise witnesses can be assembled even when an additional constraint is imposed on
the complete score vector. The final statements compare twice the number of concordant
pairs, assigning one point to ties. No tree fitting or solver is formalized here.
-/

namespace MinPI.TranscriptCells

def Local {ι X T S : Type} (legal : ι → X → Prop) (observe : ι → X → T)
    (transcript : ι → T) (refit : X → S) (s : ι → S) : Prop :=
  ∀ i, ∃ x, legal i x ∧ observe i x = transcript i ∧ refit x = s i

/-- Exact equality of the identified set with the image of rowwise compatible scores.
The side observation may specify, for example, the histogram of all refit scores. -/
theorem exact_image {ι X T S M : Type} (legal : ι → X → Prop) (observe : ι → X → T)
    (transcript : ι → T) (refit : X → S) (side : (ι → S) → Prop) (metric : (ι → S) → M)
    (m : M) :
    (∃ x : ι → X, (∀ i, legal i (x i) ∧ observe i (x i) = transcript i) ∧
      side (fun i ↦ refit (x i)) ∧ metric (fun i ↦ refit (x i)) = m) ↔
    (∃ s : ι → S, Local legal observe transcript refit s ∧ side s ∧ metric s = m) := by
  constructor
  · rintro ⟨x, hx, hs, hm⟩
    exact ⟨fun i ↦ refit (x i), fun i ↦ ⟨x i, (hx i).1, (hx i).2, rfl⟩, hs, hm⟩
  · rintro ⟨s, h, hs, hm⟩
    let x := fun i ↦ Classical.choose (h i)
    have hx (i : ι) : legal i (x i) ∧ observe i (x i) = transcript i ∧ refit (x i) = s i :=
      Classical.choose_spec (h i)
    have he : (fun i ↦ refit (x i)) = s := funext (fun i ↦ (hx i).2.2)
    exact ⟨x, fun i ↦ ⟨(hx i).1, (hx i).2.1⟩, he ▸ hs, he ▸ hm⟩

def twiceCredit (p n : Nat) : Nat := if n < p then 2 else if p = n then 1 else 0

theorem credit_monotone {p n p' n' : Nat} (hp : p ≤ p') (hn : n' ≤ n) :
    twiceCredit p n ≤ twiceCredit p' n' := by
  unfold twiceCredit
  grind

def creditTotal {ι : Type} (s : ι → Nat) (pairs : List (ι × ι)) : Nat :=
  (pairs.map fun (p, n) ↦ twiceCredit (s p) (s n)).sum

/-- The same favourable score assignment maximizes every positive-negative pair credit. -/
theorem auc_monotone {ι : Type} (s t : ι → Nat) (pairs : List (ι × ι))
    (h : ∀ pair ∈ pairs, s pair.1 ≤ t pair.1 ∧ t pair.2 ≤ s pair.2) :
    creditTotal s pairs ≤ creditTotal t pairs := by
  induction pairs with
  | nil => simp [creditTotal]
  | cons a rest ih =>
    have head := h a (by simp)
    have tail := ih (fun pair hp ↦ h pair (by simp [hp]))
    have c := credit_monotone head.1 head.2
    simpa [creditTotal, List.sum_cons] using Nat.add_le_add c tail

#print axioms exact_image
#print axioms auc_monotone

end MinPI.TranscriptCells

