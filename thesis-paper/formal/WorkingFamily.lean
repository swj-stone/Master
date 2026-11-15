import Std

/- The deletion oracle and the allowed (nonempty, one-column smaller)
   neighbours are parameters. These claims concern the returned support. -/
namespace MinPI.WorkingFamily

def Pass {S C : Type} (ballot : S → C → Prop) (family : C → Prop) (s : S) : Prop :=
  ∀ c, family c → ballot s c

def Local {S C : Type} (ballot : S → C → Prop) (family : C → Prop)
    (neighbor : S → S → Prop) (s : S) : Prop :=
  Pass ballot family s ∧ ∀ t, neighbor s t → ¬ Pass ballot family t

theorem working_to_full {S C : Type} (ballot : S → C → Prop)
    (active full : C → Prop) (neighbor : S → S → Prop) (s : S)
    (included : ∀ c, active c → full c) (small : Local ballot active neighbor s)
    (finalCheck : Pass ballot full s) : Local ballot full neighbor s := by
  refine ⟨finalCheck, ?_⟩
  intro t hn hfull
  apply small.2 t hn
  intro c hc
  exact hfull c (included c hc)

theorem veto_is_new {S C : Type} (ballot : S → C → Prop) (active : C → Prop)
    (s : S) (c : C) (accepted : Pass ballot active s) (veto : ¬ ballot s c) :
    ¬ active c := by
  intro hc
  exact veto (accepted c hc)

theorem order_irrelevant {S C : Type} (ballot : S → C → Prop) (left right : List C)
    (same : ∀ c, c ∈ left ↔ c ∈ right) (s : S) :
    Pass ballot (fun c => c ∈ left) s ↔ Pass ballot (fun c => c ∈ right) s := by
  constructor
  · intro h c hc
    exact h c ((same c).2 hc)
  · intro h c hc
    exact h c ((same c).1 hc)

#print axioms working_to_full
#print axioms veto_is_new
#print axioms order_irrelevant
end MinPI.WorkingFamily
