import Std

/-!
# Feature-specific projection veto for a greedy deletion

Integer coordinates suffice for exact rational data by clearing common denominators.
`den` below scales every coefficient of the projection, so that a cell with original
effect h uses coefficient c/den. Current deletion effects can have either sign.
-/
namespace MinPI.TreeProjection

def dot {J : Type} (x y : J → Int) (xs : List J) : Int := (xs.map (fun j ↦ x j*y j)).sum

private theorem norm_nonneg {J : Type} (x : J → Int) (xs : List J) : 0 ≤ dot x x xs := by
  induction xs with
  | nil => simp [dot]
  | cons j js ih =>
    have one : 0 ≤ x j*x j := by
      by_cases h : 0 ≤ x j
      · exact Int.mul_nonneg h h
      · exact Int.mul_nonneg_of_nonpos_of_nonpos (by omega) (by omega)
    simp only [dot, List.map_cons, List.sum_cons] at ih ⊢
    omega

theorem residual_cross {J : Type} (r d : J → Int) (xs : List J) :
    dot (fun j ↦ r j+d j) (fun j ↦ r j+d j) xs =
    dot r r xs + 2*dot r d xs + dot d d xs := by
  induction xs with
  | nil => simp [dot]
  | cons j js ih =>
    simp only [dot, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih]
    grind

private theorem residual_square {J : Type} (d h : J → Int) (xs : List J) (den c : Int) :
    dot (fun j ↦ den*d j-c*h j) (fun j ↦ den*d j-c*h j) xs =
    den*den*dot d d xs - 2*den*c*dot d h xs + c*c*dot h h xs := by
  induction xs with
  | nil => simp [dot]
  | cons j js ih =>
    simp only [dot, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih]
    grind

/-- On one bin the optimal coefficient solves its scalar normal equation.
This projection identity includes a zero original vector (take c=0 there). -/
theorem cell_identity {J : Type} (d h : J → Int) (xs : List J) (den c : Int)
    (normal : den*dot d h xs = c*dot h h xs) :
    den*den*dot d d xs = c*c*dot h h xs +
    dot (fun j ↦ den*d j-c*h j) (fun j ↦ den*d j-c*h j) xs := by
  rw [residual_square]
  grind

/-- A single cell contributes at most its exact deletion energy. -/
theorem cell_lower {J : Type} (d h : J → Int) (xs : List J) (den c : Int)
    (normal : den*dot d h xs = c*dot h h xs) :
    c*c*dot h h xs ≤ den*den*dot d d xs := by
  have := cell_identity d h xs den c normal
  have := norm_nonneg (fun j ↦ den*d j-c*h j) xs
  omega

/-- The coordinate bins are disjoint blocks; summing their lower bounds preserves the
inequality. Their concatenation is the complete voting-coordinate vector. -/
theorem cells_lower {J : Type} (d h : J → Int) (bins : List (Int × List J)) (den : Int)
    (normal : ∀ b ∈ bins, den*dot d h b.2 = b.1*dot h h b.2) :
    (bins.map (fun b ↦ b.1*b.1*dot h h b.2)).sum ≤
    (bins.map (fun b ↦ den*den*dot d d b.2)).sum := by
  induction bins with
  | nil => simp
  | cons b bs ih =>
    have one := cell_lower d h b.2 den b.1 (normal b (by simp))
    have tail := ih fun v hv ↦ normal v (by simp [hv])
    simp only [List.map_cons, List.sum_cons]
    omega

/-- Scores are minimized; smaller priority wins ties. A strict veto therefore never
discards a candidate that could beat or tie the incumbent, regardless of priority. -/
def better (score : Int) (priority : Nat) (incumbent : Int) (incPriority : Nat) : Prop :=
  score < incumbent ∨ (score = incumbent ∧ priority < incPriority)

/-- For any finite family of coalition sizes, an exact feasible deletion passes every vote.
The validity hypotheses follow from `cell_lower` on the full list of coordinates. -/
theorem passes_every_size (fixed norm budget : Int) (bounds : List Int)
    (valid : ∀ q ∈ bounds, q ≤ norm) (feasible : fixed + norm ≤ budget) :
    ∀ q ∈ bounds, fixed + q ≤ budget := by
  intro q h
  have := valid q h
  omega

theorem strict_veto (fixed norm bound incumbent : Int) (priority incPriority : Nat)
    (lower : bound ≤ norm) (veto : incumbent < fixed+bound) :
    ¬ better (fixed+norm) priority incumbent incPriority := by
  unfold better
  omega

/-- Full one-step agreement: every unresolved candidate has been strictly vetoed,
and every resolved candidate loses to the chosen incumbent under the common tie rule. -/
theorem full_winner {J : Type} (score lower : J → Int) (priority : J → Nat) (resolved : J → Bool)
    (chosen : J) (valid : ∀ j, lower j ≤ score j)
    (unread : ∀ j, resolved j = false → score chosen < lower j)
    (best_read : ∀ j, resolved j = true → ¬ better (score j) (priority j) (score chosen) (priority chosen)) :
    ∀ j, ¬ better (score j) (priority j) (score chosen) (priority chosen) := by
  intro j
  cases h : resolved j with
  | true => exact best_read j h
  | false =>
    have := unread j h
    have := valid j
    unfold better
    omega

#print axioms residual_cross
#print axioms cell_identity
#print axioms cells_lower
#print axioms passes_every_size
#print axioms strict_veto
#print axioms full_winner
end MinPI.TreeProjection
