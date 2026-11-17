import Std

/-!
# Sharp upper bound from observed coalition votes

Fix a feature. An edge index specifies the membership of all *other* features. The two
endpoints are obtained by setting the fixed feature to false or true. Thus the representation
below assigns every coalition to exactly one pair. A completion fills each missing vote
independently; the known votes, including any known quota endpoints, remain fixed.

To maximize all edge increments for this feature simultaneously, fill missing base votes
with false and missing extension votes with true. This constructs a single feasible game
for this feature. Taking its attained minimum gives the sharp upper bound over all Boolean
completions. Different features may use different maximizing completions.
-/

namespace MinPI.PartialUpper

abbrev Transcript (E : Type) := E → Bool → Option Bool
abbrev Game (E : Type) := E → Bool → Bool

def difference (base extension : Bool) : Int :=
  (if extension then 1 else 0) - (if base then 1 else 0)

def edge {E : Type} (g : Game E) (s : E) : Int := difference (g s false) (g s true)

def agrees (o : Option Bool) (v : Bool) : Prop := o = none ∨ o = some v

def Feasible {E : Type} (p : Transcript E) (g : Game E) : Prop :=
  ∀ s b, agrees (p s b) (g s b)

def optimistic {E : Type} (p : Transcript E) : Game E :=
  fun s b ↦ (p s b).getD b

theorem feasible_optimistic {E : Type} (p : Transcript E) : Feasible p (optimistic p) := by
  intro s b
  cases o : p s b <;> simp [agrees, optimistic, o]

theorem edge_upper (a b : Option Bool) (x y : Bool) (hx : agrees a x) (hy : agrees b y) :
    difference x y ≤ difference (a.getD false) (b.getD true) := by
  rcases hx with rfl | rfl <;> rcases hy with rfl | rfl <;>
    cases x <;> cases y <;> decide

theorem pointwise_upper {E : Type} {p : Transcript E} {g : Game E} (h : Feasible p g) (s : E) :
    edge g s ≤ edge (optimistic p) s := edge_upper _ _ _ _ (h s false) (h s true)

def IsMinimum {E : Type} (g : Game E) (m : Int) : Prop :=
  (∀ s, m ≤ edge g s) ∧ ∃ s, edge g s = m

theorem ternary (a b : Bool) : -1 ≤ difference a b ∧ difference a b ≤ 1 := by
  cases a <;> cases b <;> decide

def Decline {E : Type} (p : Transcript E) : Prop :=
  ∃ s, p s false = some true ∧ p s true = some false

def Compatible {E : Type} (p : Transcript E) : Prop :=
  ∀ s, p s false ≠ some true ∧ p s true ≠ some false

noncomputable def bound {E : Type} (p : Transcript E) : Int := by
  classical
  exact if Decline p then -1 else if Compatible p then 1 else 0

private theorem negative_case (a b : Option Bool) :
    difference (a.getD false) (b.getD true) = -1 ↔ a = some true ∧ b = some false := by
  rcases a with _ | (_ | _) <;> rcases b with _ | (_ | _) <;> decide

private theorem positive_case (a b : Option Bool) :
    difference (a.getD false) (b.getD true) = 1 ↔ a ≠ some true ∧ b ≠ some false := by
  rcases a with _ | (_ | _) <;> rcases b with _ | (_ | _) <;> decide

theorem optimistic_attains {E : Type} [Nonempty E] (p : Transcript E) :
    IsMinimum (optimistic p) (bound p) := by
  classical
  have limits (s : E) : -1 ≤ edge (optimistic p) s ∧ edge (optimistic p) s ≤ 1 :=
    ternary _ _
  by_cases down : Decline p
  · rw [bound, ite_eq_left down]
    obtain ⟨s, hs⟩ := down
    exact ⟨fun s ↦ (limits s).1, ⟨s, (negative_case _ _).2 hs⟩⟩
  · rw [bound, ite_eq_right down]
    by_cases ok : Compatible p
    · rw [ite_eq_left ok]
      have allOne (s : E) : edge (optimistic p) s = 1 := (positive_case _ _).2 (ok s)
      exact ⟨fun s ↦ Int.le_of_eq (allOne s).symm, ⟨Classical.ofNonempty, allOne _⟩⟩
    · rw [ite_eq_right ok]
      have lower (s : E) : 0 ≤ edge (optimistic p) s := by
        have h := (limits s).1
        have ne : edge (optimistic p) s ≠ -1 :=
          fun e ↦ down ⟨s, (negative_case _ _).1 e⟩
        omega
      obtain ⟨s, hs⟩ := Classical.not_forall.mp ok
      have ne : edge (optimistic p) s ≠ 1 := fun e ↦ hs ((positive_case _ _).1 e)
      have h := limits s
      have h' := lower s
      exact ⟨lower, ⟨s, by omega⟩⟩

/-- There is a feasible game attaining the upper bound and no feasible minimum can exceed it. -/
theorem sharp {E : Type} [Nonempty E] (p : Transcript E) :
    (∃ g, Feasible p g ∧ IsMinimum g (bound p)) ∧
    (∀ g m, Feasible p g → IsMinimum g m → m ≤ bound p) := by
  have h := optimistic_attains p
  constructor
  · exact ⟨optimistic p, feasible_optimistic p, h⟩
  · intro g m hmatch hm
    obtain ⟨s, equal⟩ := h.2
    exact equal ▸ Int.le_trans (hm.1 s) (pointwise_upper hmatch s)

#print axioms pointwise_upper
#print axioms optimistic_attains
#print axioms sharp

end MinPI.PartialUpper
