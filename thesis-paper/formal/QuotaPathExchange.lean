import Std

/-! Fixed-width exchanges. The two focal columns are excluded from the row of interactions.
All integer scores may be rescaled by one common positive factor. -/
namespace MinPI.Exchange

def maskedSum {J : Type} (g : J → Int) (take : J → Bool) : List J → Int
  | [] => 0
  | j :: js => (if take j then g j else 0) + maskedSum g take js

private theorem center_sum {J : Type} (g : J → Int) (c : Int) (take : J → Bool) (xs : List J) :
    maskedSum (fun j ↦ g j - c) take xs =
      maskedSum g take xs - (xs.countP take : Int) * c := by
  induction xs with
  | nil => simp [maskedSum]
  | cons j js ih =>
    cases h : take j <;> simp [maskedSum, h, ih, Int.add_mul] <;> omega

private theorem cutoff_upper {J : Type} (g : J → Int) (top take : J → Bool)
    (c : Int) (separates : ∀ j, (top j = true → c ≤ g j) ∧ (top j = false → g j ≤ c))
    (xs : List J) : maskedSum (fun j ↦ g j - c) take xs ≤ maskedSum (fun j ↦ g j - c) top xs := by
  induction xs with
  | nil => simp [maskedSum]
  | cons j js ih =>
    have h := separates j
    cases a : take j <;> cases b : top j <;>
      simp [maskedSum, a, b] at h ⊢ <;> omega

/-- A selection of k−1 largest row differences maximizes the interaction sum. A cutoff
allows arbitrary splits of ties. Its mask itself attains the bound. -/
theorem top_sum {J : Type} (g : J → Int) (c : Int) (top take : J → Bool)
    (separates : ∀ j, (top j = true → c ≤ g j) ∧ (top j = false → g j ≤ c))
    (xs : List J) (same : xs.countP take = xs.countP top) :
    maskedSum g take xs ≤ maskedSum g top xs := by
  have h := cutoff_upper g top take c separates xs
  rw [center_sum, center_sum, same] at h
  omega

/-- In a common context of the given width, replacing j by i has advantage at least the
affine margin at the k−1 greatest interaction differences. -/
theorem exchange_margin {J : Type} (d : Int) (g : J → Int) (c : Int) (top take : J → Bool)
    (separates : ∀ j, (top j = true → c ≤ g j) ∧ (top j = false → g j ≤ c))
    (xs : List J) (same : xs.countP take = xs.countP top) :
    d - 2 * maskedSum g top xs ≤ d - 2 * maskedSum g take xs := by
  have := top_sum g c top take separates xs same
  omega

/-- All optima respect any strict exchange edge from an unselected to a selected column.
The functions `hasI`, `hasJ` refer to a specific pair; `C` is the finite set of feasible
width-k supports and `swap` maps a feasible support to a feasible support. -/
theorem optimal_closed {C : Type} (score : C → Int) (hasI hasJ : C → Bool) (swap : C → C)
    (improves : ∀ s, hasI s = false → hasJ s = true → score s < score (swap s))
    (chosen : C) (optimal : ∀ s, score s ≤ score chosen) :
    hasJ chosen = true → hasI chosen = true := by
  intro hj
  cases hi : hasI chosen with
  | true => rfl
  | false =>
    have := improves chosen hi hj
    have := optimal (swap chosen)
    omega

private theorem count_monotone {J : Type} (p q : J → Bool) (xs : List J)
    (contained : ∀ j, p j = true → q j = true) : xs.countP p ≤ xs.countP q := by
  induction xs with
  | nil => simp
  | cons j js ih =>
    have h := contained j
    cases a : p j <;> cases b : q j <;> simp [a, b] at h ⊢ <;> omega

/-- xs lists the p−1 other columns. A width-k support containing j has only k−1
other selected columns. k distinct dominators cannot all fit. -/
theorem excluded_by_count {J : Type} (xs : List J) (dom picked : J → Bool) (focal : Bool) (k : Nat)
    (width : xs.countP picked + (if focal then 1 else 0) = k)
    (closed : focal = true → ∀ j, dom j = true → picked j = true)
    (many : k ≤ xs.countP dom) : focal = false := by
  cases h : focal with
  | false => rfl
  | true =>
    simp only [h, ↓reduceIte] at width
    have := count_monotone dom picked xs (closed h)
    omega

/-- Among the p−1 columns besides an absent i, only p−k−1 are unselected.
p−k strictly dominated columns could not all be unselected. -/
theorem forced_by_count {J : Type} (xs : List J) (dom skipped : J → Bool) (focal : Bool)
    (complementWidth : Nat)
    (width : xs.countP skipped + (if focal then 0 else 1) = complementWidth)
    (closed : focal = false → ∀ j, dom j = true → skipped j = true)
    (many : complementWidth ≤ xs.countP dom) : focal = true := by
  cases h : focal with
  | true => rfl
  | false =>
    simp only [h, Bool.false_eq_true, ↓reduceIte] at width
    have := count_monotone dom skipped xs (closed h)
    omega

/-- Three named columns and an arbitrary common set s of outside columns. In the four
exchanges in the premises, the total width is always q(s)+1+bit. This gives transitivity
even when the intermediate column belongs to the common context of the outer pair. -/
theorem strict_exchange_transitive {C : Type} (q : C → Nat) (k : Nat)
    (u : C → Bool → Bool → Bool → Int)
    (ij : ∀ s b, q s + 1 + (if b then 1 else 0) = k → u s false true b < u s true false b)
    (jh : ∀ s b, q s + 1 + (if b then 1 else 0) = k → u s b false true < u s b true false)
    (s : C) (b : Bool) (width : q s + 1 + (if b then 1 else 0) = k) :
    u s false b true < u s true b false := by
  cases b with
  | false => exact Int.lt_trans (jh s false width) (ij s false width)
  | true => exact Int.lt_trans (ij s true width) (jh s true width)

#print axioms exchange_margin
#print axioms optimal_closed
#print axioms excluded_by_count
#print axioms forced_by_count
#print axioms strict_exchange_transitive
end MinPI.Exchange
