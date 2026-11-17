import Std

/-! Each weight can include a common denominator for a size-conditioned
coefficient. Path-group union weights may have either sign. -/
namespace MinPI.TreeProperQuota

/-- Apply once per voting coordinate and available mask, then sum. -/
theorem pointwise_excess (z kept full : Int) :
    (z-kept)*(z-kept)-(z-full)*(z-full) =
    2*z*(full-kept)-(full*full-kept*kept) := by grind

def total {U : Type} (us : List U) (w : U → Int) (test : U → Bool) : Int :=
  (us.map (fun u ↦ if test u then w u else 0)).sum

private theorem new_one (live hit : Bool) (v : Int) :
    (if !(live && !hit) then v else 0) =
    (if !live then v else 0) + (if live && hit then v else 0) := by
  cases live <;> cases hit <;> simp

private theorem remaining_one (live removed candidate : Bool) (v : Int) :
    (if live && !removed && candidate then v else 0) =
    (if live && candidate then v else 0) - (if live && removed && candidate then v else 0) := by
  cases live <;> cases removed <;> cases candidate <;> simp

/-- Newly disabled unions are exactly the live unions incident to the deletion.
The terms with repeated or overlapping indices have no extra multiplicity. -/
theorem new_risk {U : Type} (us : List U) (w : U → Int) (live hit : U → Bool) :
    total us w (fun u ↦ !(live u && !hit u)) =
    total us w (fun u ↦ !live u) + total us w (fun u ↦ live u && hit u) := by
  induction us with
  | nil => simp [total]
  | cons u us ih =>
    simp only [total, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih, new_one (live u) (hit u) (w u)]
    omega

/-- The prospective cost of deleting a surviving candidate loses exactly the
live unions that contained both the eliminated column and this candidate. -/
theorem update {U : Type} (us : List U) (w : U → Int) (live removed candidate : U → Bool) :
    total us w (fun u ↦ live u && !removed u && candidate u) =
    total us w (fun u ↦ live u && candidate u) -
    total us w (fun u ↦ live u && removed u && candidate u) := by
  induction us with
  | nil => simp [total]
  | cons u us ih =>
    simp only [total, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih, remaining_one (live u) (removed u) (candidate u) (w u)]
    omega

#print axioms pointwise_excess
#print axioms new_risk
#print axioms update
end MinPI.TreeProperQuota
