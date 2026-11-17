import Std

/-!
# Exact worst marginal of a quadratic coalition game

Fix one target feature. Each other active column has an integer interaction weight `g j`.
For a squared-loss game it is the voting inner product of the two columns' prediction vectors.
The one-feature marginal is affine in the inclusion bits. The identities below operate on that
affine marginal. Rational inputs reduce to these identities by one common positive scaling.
-/
namespace MinPI.Quadratic

def total {J : Type} (f : J → Int) (xs : List J) : Int := (xs.map f).sum

def active {J : Type} (g : J → Int) (take : J → Bool) (j : J) : Int :=
  if take j then g j else 0

def pos (x : Int) : Int := max 0 x
def neg (x : Int) : Int := max 0 (-x)

def marginal {J : Type} (d : Int) (g : J → Int) (xs : List J) (take : J → Bool) : Int :=
  d - 2 * total (active g take) xs

def worst {J : Type} (d : Int) (g : J → Int) (xs : List J) : Int :=
  d - 2 * total (fun j ↦ pos (g j)) xs

private theorem positive_sum_upper {J : Type} (g : J → Int) (take : J → Bool) (xs : List J) :
    total (active g take) xs ≤ total (fun j ↦ pos (g j)) xs := by
  induction xs with
  | nil => simp [total]
  | cons j xs ih =>
    have one : active g take j ≤ pos (g j) := by
      cases h : take j <;> simp [active, pos, h] <;> omega
    simp only [total, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- The lower bound is attained by including exactly the positively interacting columns.
Zero interactions can be included or excluded with the same result. -/
theorem exact_minimum {J : Type} (d : Int) (g : J → Int) (xs : List J) :
    (∀ take, worst d g xs ≤ marginal d g xs take) ∧
    marginal d g xs (fun j ↦ decide (0 < g j)) = worst d g xs := by
  constructor
  · intro take
    have := positive_sum_upper g take xs
    unfold worst marginal
    omega
  · have same : active g (fun j ↦ decide (0 < g j)) = (fun j ↦ pos (g j)) := by
      funext j
      by_cases h : 0 < g j
      · simp [active, h, pos, Int.max_eq_right (Int.le_of_lt h)]
      · simp [active, h, pos, Int.max_eq_left (by omega : g j ≤ 0)]
    simp [marginal, worst, same]

/-- Removing column h increases each surviving worst score by twice its positive interaction. -/
theorem remove_update {J : Type} (d : Int) (g : J → Int) (h : J) (left right : List J) :
    worst d g (left ++ right) = worst d g (left ++ h :: right) + 2 * pos (g h) := by
  simp [worst, total, List.sum_append]
  omega

private theorem positive_split {J : Type} (g : J → Int) (xs : List J) :
    2 * total (fun j ↦ pos (g j)) xs = total g xs +
      total (fun j ↦ pos (g j) + neg (g j)) xs := by
  induction xs with
  | nil => simp [total]
  | cons j xs ih =>
    have one : 2 * pos (g j) = g j + (pos (g j) + neg (g j)) := by
      unfold pos neg
      omega
    simp only [total, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- Subtracting the total absolute interaction from half-inclusion mean gives the minimum. -/
theorem mean_minus_radius {J : Type} (d : Int) (g : J → Int) (xs : List J) :
    worst d g xs = (d - total g xs) - total (fun j ↦ pos (g j) + neg (g j)) xs := by
  have := positive_split g xs
  unfold worst
  omega

/-- Any positive worst marginal certifies that adding i improves every compared coalition. -/
theorem positive_for_all {J : Type} (d : Int) (g : J → Int) (xs : List J)
    (positive : 0 < worst d g xs) (take : J → Bool) : 0 < marginal d g xs take :=
  Int.lt_of_lt_of_le positive ((exact_minimum d g xs).1 take)

def best {J : Type} (d : Int) (g : J → Int) (xs : List J) : Int :=
  d + 2 * total (fun j ↦ neg (g j)) xs

private theorem negative_sum_lower {J : Type} (g : J → Int) (take : J → Bool) (xs : List J) :
    -(total (fun j ↦ neg (g j)) xs) ≤ total (active g take) xs := by
  induction xs with
  | nil => simp [total]
  | cons j xs ih =>
    have one : -(neg (g j)) ≤ active g take j := by
      cases h : take j <;> simp [active, neg, h] <;> omega
    simp only [total, List.map_cons, List.sum_cons] at ih ⊢
    omega

private theorem minimum_sum {J : Type} (g : J → Int) (xs : List J) :
    total (active g (fun j ↦ decide (g j < 0))) xs = -(total (fun j ↦ neg (g j)) xs) := by
  induction xs with
  | nil => simp [total]
  | cons j xs ih =>
    have one : active g (fun j ↦ decide (g j < 0)) j = -(neg (g j)) := by
      by_cases h : g j < 0 <;> simp [active, neg, h] <;> omega
    simp only [total, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- Every coalition lies below the upper endpoint, attained at the negative interactions. -/
theorem exact_maximum {J : Type} (d : Int) (g : J → Int) (xs : List J) :
    (∀ take, marginal d g xs take ≤ best d g xs) ∧
    marginal d g xs (fun j ↦ decide (g j < 0)) = best d g xs := by
  constructor
  · intro take
    have := negative_sum_lower g take xs
    unfold best marginal
    omega
  · unfold marginal best
    rw [minimum_sum]
    omega

/-- The competitive terms for all other variables can be arbitrary. Only the objective
difference on toggling the focal variable needs to match its penalized marginal. -/
def Optimal {J : Type} (reward : Bool → (J → Bool) → Int) (on : Bool) (rest : J → Bool) : Prop :=
  ∀ b bits, reward b bits ≤ reward on rest

/-- A strict margin fixes the focal variable in *every* global optimum. A single interval of
penalties [lo,hi] is covered by comparing the endpoints to hi and lo respectively. -/
theorem fix_throughout_interval {J : Type} (d : Int) (g : J → Int) (xs : List J)
    (lo penalty hi : Int) (lower : lo ≤ penalty) (upper : penalty ≤ hi)
    (reward : Bool → (J → Bool) → Int)
    (difference : ∀ bits, reward true bits - reward false bits = marginal d g xs bits - penalty)
    (on : Bool) (rest : J → Bool) (opt : Optimal reward on rest) :
    (hi < worst d g xs → on = true) ∧ (best d g xs < lo → on = false) := by
  have low := (exact_minimum d g xs).1 rest
  have high := (exact_maximum d g xs).1 rest
  have flip := difference rest
  constructor
  · intro above
    cases on with
    | true => rfl
    | false =>
      have contra := opt true rest
      omega
  · intro below
    cases on with
    | false => rfl
    | true =>
      have contra := opt false rest
      omega

#print axioms exact_minimum
#print axioms remove_update
#print axioms mean_minus_radius
#print axioms positive_for_all
#print axioms exact_maximum
#print axioms fix_throughout_interval
end MinPI.Quadratic
