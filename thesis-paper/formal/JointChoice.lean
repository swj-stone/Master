import Std

namespace MinPI.JointChoice

/-- Regret in units of an equal positive deletion fraction. The policy draws the first
candidate with probability q before the common latent world is observed. -/
def regret (q : Rat) (first second : Bool) : Rat :=
  if first && !second then 1-q else if second && !first then q else 0

/-- When the second candidate never passes alone, always choosing the first incurs no regret. -/
theorem dominance {Ω : Type} (a b : Ω → Bool) (h : ∀ w, b w = true → a w = true)
    (w : Ω) : regret 1 (a w) (b w) = 0 := by
  have hw := h w
  cases ha : a w <;> cases hb : b w <;> simp_all [regret] <;> grind

/-- A half lottery has regret at most one half in every world. If each strict order is
attainable under the common observer, some feasible world charges any lottery at least half. -/
theorem two_sided {Ω : Type} (a b : Ω → Bool)
    (left : ∃ w, a w = true ∧ b w = false) (right : ∃ w, a w = false ∧ b w = true) :
    (∀ w, regret (1/2) (a w) (b w) ≤ 1/2) ∧ (∀ q, ∃ w, 1/2 ≤ regret q (a w) (b w)) := by
  constructor
  · intro w
    cases a w <;> cases b w <;> simp [regret] <;> grind
  · intro q
    by_cases h : q ≤ 1/2
    · obtain ⟨w, ha, hb⟩ := left
      exact ⟨w, by simp [ha, hb, regret]; grind⟩
    · obtain ⟨w, ha, hb⟩ := right
      exact ⟨w, by simp [ha, hb, regret]; grind⟩

#print axioms dominance
#print axioms two_sided
end MinPI.JointChoice
