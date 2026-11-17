import Std

/-! The scalar coefficients below are inner products with the group-order
vectors, divided by choose(p-1,q). Clearing common denominators puts all
coefficients on an integer grid. Vanishing of every tested marginal is
equivalent to vanishing of every order coefficient. -/
namespace MinPI.TreeSizeSpan

def binomial : Nat → Nat → Int
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => binomial n k + binomial n (k+1)

theorem above (s q : Nat) (h : s < q) : binomial s q = 0 := by
  induction s generalizing q with
  | zero => cases q <;> simp_all [binomial]
  | succ s ih =>
    cases q with
    | zero => omega
    | succ q =>
      rw [binomial, ih q (by omega), ih (q+1) (by omega)]
      decide

theorem diagonal (q : Nat) : binomial q q = 1 := by
  induction q with
  | zero => rfl
  | succ q ih => simp [binomial, ih, above q (q+1) (by omega)]

def row (a : Nat → Int) (s : Nat) : Nat → Int
  | 0 => 0
  | q+1 => row a s q + binomial s q * a q

theorem zero_sum (a : Nat → Int) (s n : Nat) (h : ∀ q < n, a q = 0) :
    row a s n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp [row, ih (fun q hq ↦ h q (by omega)), h n (by omega)]

/-- The square Pascal matrix on rows and columns 0,...,D-1 has trivial kernel.
Above-diagonal entries vanish, hence row s need only be summed through s. -/
theorem first_rows_determine (a : Nat → Int) (D : Nat)
    (h : ∀ s < D, row a s (s+1) = 0) : ∀ q < D, a q = 0 := by
  induction D with
  | zero => intro q hq; omega
  | succ n ih =>
    have earlier := ih fun s hs ↦ h s (by omega)
    intro q hq
    by_cases same : q = n
    · subst q
      have last := h n (by omega)
      rw [row, zero_sum a n n earlier, diagonal] at last
      simpa using last
    · exact earlier q (by omega)

/-- Once the first D rows vanish, every one of the p sizes vanishes. -/
theorem all_from_first (a : Nat → Int) (D s : Nat)
    (h : ∀ t < D, row a t (t+1) = 0) : row a s D = 0 :=
  zero_sum a s D (first_rows_determine a D h)

def dot {J : Type} (x y : J → Int) (js : List J) : Int :=
  (js.map (fun j ↦ x j * y j)).sum

private theorem nonneg {J : Type} (x : J → Int) (js : List J) : 0 ≤ dot x x js := by
  induction js with
  | nil => simp [dot]
  | cons j js ih =>
    have square : 0 ≤ x j * x j := by
      by_cases h : 0 ≤ x j
      · exact Int.mul_nonneg h h
      · exact Int.mul_nonneg_of_nonpos_of_nonpos (by omega) (by omega)
    simp only [dot, List.map_cons, List.sum_cons] at ih ⊢
    omega

private theorem difference {J : Type} (x y : J → Int) (js : List J) :
    dot (fun j ↦ x j-y j) (fun j ↦ x j-y j) js =
    dot x x js - 2*dot y x js + dot y y js := by
  induction js with
  | nil => simp [dot]
  | cons j js ih =>
    simp only [dot, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih]
    grind

/-- The feasible in-span representative y attains the norm lower bound.
Membership in the span makes its own inner product a linear combination of
the measured inner products, hence fixed over all compatible x. -/
theorem minimum_norm {J : Type} (feasible : (J → Int) → Prop) (y : J → Int)
    (js : List J) (attained : feasible y)
    (normal : ∀ x, feasible x → dot y x js = dot y y js) :
    feasible y ∧ ∀ x, feasible x → dot y y js ≤ dot x x js := by
  refine ⟨attained, ?_⟩
  intro x h
  have eq := difference x y js
  rw [normal x h] at eq
  have := nonneg (fun j ↦ x j-y j) js
  omega

#print axioms first_rows_determine
#print axioms all_from_first
#print axioms minimum_norm
end MinPI.TreeSizeSpan
