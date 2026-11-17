import Std

/-!
# Positive MinPI in a binary game

For a feature to have minimum marginal one, both endpoints of *every* edge adding it
are forced: the base loses and the extension wins. Every coalition is one such endpoint.
Consequently a coalition wins exactly when it contains this feature. Conversely this
membership rule makes every such marginal one. Toggling a different feature then leaves
every vote unchanged.

A column eligible for this rule must occur in every observed winning support and in no
observed losing support. When a winning support has an observed losing superset, no
column can satisfy both constraints. These are statements about the complete game even
when the observations are incomplete. They require neither monotonicity nor a learner.
-/

namespace MinPI.Dictator

variable {ι : Type} [DecidableEq ι]

abbrev Coalition (ι : Type) := ι → Bool
abbrev Game (ι : Type) := Coalition ι → Bool

def put (s : Coalition ι) (i : ι) (b : Bool) : Coalition ι :=
  fun j ↦ if j = i then b else s j

def difference (a b : Bool) : Int := (if b then 1 else 0) - (if a then 1 else 0)

def edge (v : Game ι) (i : ι) (s : Coalition ι) : Int :=
  difference (v (put s i false)) (v (put s i true))

/-- A lower bound attained at an edge is its exact minimum over all supports. -/
def IsMinimum (v : Game ι) (i : ι) (m : Int) : Prop :=
  (∀ s, m ≤ edge v i s) ∧ ∃ s, edge v i s = m

theorem minimum_unique {v : Game ι} {i : ι} {a b : Int}
    (ha : IsMinimum v i a) (hb : IsMinimum v i b) : a = b := by
  obtain ⟨la, sa, ra⟩ := ha
  obtain ⟨lb, sb, rb⟩ := hb
  exact Int.le_antisymm (rb ▸ la sb) (ra ▸ lb sa)

theorem put_existing (s : Coalition ι) (i : ι) : put s i (s i) = s := by
  funext j
  by_cases h : j = i <;> simp [put, h]

theorem one_le_difference (a b : Bool) : 1 ≤ difference a b ↔ a = false ∧ b = true := by
  cases a <;> cases b <;> decide

theorem positive_iff_membership (v : Game ι) (i : ι) :
    IsMinimum v i 1 ↔ ∀ s, v s = s i := by
  constructor
  · intro h s
    have ends := (one_le_difference _ _).1 (h.1 s)
    cases hi : s i with
    | false =>
      have same : put s i false = s := by simpa [hi] using put_existing s i
      simpa [same] using ends.1
    | true =>
      have same : put s i true = s := by simpa [hi] using put_existing s i
      simpa [same] using ends.2
  · intro d
    have e (s : Coalition ι) : edge v i s = 1 := by simp [edge, d, put, difference]
    exact ⟨fun s ↦ Int.le_of_eq (e s).symm, ⟨fun _ ↦ false, e _⟩⟩

theorem other_minimum_zero {v : Game ι} {i j : ι}
    (hi : IsMinimum v i 1) (ne : i ≠ j) : IsMinimum v j 0 := by
  have d := (positive_iff_membership v i).1 hi
  have e (s : Coalition ι) : edge v j s = 0 := by simp [edge, d, put, ne, difference]
  exact ⟨fun s ↦ Int.le_of_eq (e s).symm, ⟨fun _ ↦ false, e _⟩⟩

theorem positive_unique {v : Game ι} {i j : ι}
    (hi : IsMinimum v i 1) (hj : IsMinimum v j 1) : i = j := by
  by_cases eq : i = j
  · exact eq
  · have := minimum_unique (other_minimum_zero hi eq) hj
    omega

/-- Membership in the intersection of observed wins minus the union of observed losses. -/
def Compatible (v : Game ι) (seen : List (Coalition ι)) (i : ι) : Prop :=
  ∀ s ∈ seen, v s = s i

theorem exclude_from_observations {v : Game ι} {seen : List (Coalition ι)} {i : ι}
    (out : ¬Compatible v seen i) : ¬ IsMinimum v i 1 := by
  intro h
  exact out (fun s _ ↦ (positive_iff_membership v i).1 h s)

theorem excluded_upper_bound {v : Game ι} {seen : List (Coalition ι)} {i : ι} {m : Int}
    (h : IsMinimum v i m) (out : ¬Compatible v seen i) : m ≤ 0 := by
  have never : m ≠ 1 := fun e ↦ exclude_from_observations out (e ▸ h)
  obtain ⟨s, hs⟩ := h.2
  have bound (a b : Bool) : difference a b ≤ 1 := by cases a <;> cases b <;> decide
  have top : edge v i s ≤ 1 := bound _ _
  omega

omit [DecidableEq ι] in
/-- One observed winning support contained in a losing support empties the candidate set. -/
theorem negative_pair_excludes_all {v : Game ι} {seen : List (Coalition ι)}
    {s t : Coalition ι} (hs : s ∈ seen) (ht : t ∈ seen)
    (sub : ∀ j, s j = true → t j = true) (win : v s = true) (lose : v t = false) :
    ∀ i, ¬ Compatible v seen i := by
  intro i h
  have a : s i = true := (h s hs).symm.trans win
  have b : t i = false := (h t ht).symm.trans lose
  have clash := (sub i a).symm.trans b
  contradiction

#print axioms positive_iff_membership
#print axioms other_minimum_zero
#print axioms positive_unique
#print axioms exclude_from_observations
#print axioms excluded_upper_bound
#print axioms negative_pair_excludes_all

end MinPI.Dictator
