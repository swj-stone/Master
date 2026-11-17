import Std

/-!
# Directional fidelity votes

Directional weights and quotas are integers after any common positive scaling. For physical
prediction vectors the quota is u·F−R‖u‖. The branch certificate only uses the resulting scalar
inequality. The last lemma applies the exact dual norm identity to any finite Euclidean vector
of integer coordinates, without division or a square root.
-/
namespace MinPI.Directional

def weight {I : Type} (w : I → Int) (on : I → Bool) (xs : List I) : Int :=
  (xs.map (fun i ↦ if on i then w i else 0)).sum

def excess {I : Type} (w : I → Int) (c : Int) (xs : List I) : Int :=
  (xs.map (fun i ↦ max 0 (w i-c))).sum

private theorem affine_upper {I : Type} (w : I → Int) (c : Int) (on : I → Bool) (xs : List I) :
    weight w on xs ≤ (xs.countP on : Int)*c + excess w c xs := by
  induction xs with
  | nil => simp [weight, excess]
  | cons i xs ih =>
    cases h : on i <;> simp [weight, excess, h, Int.add_mul] at ih ⊢ <;> omega

/-- A threshold c≥0 prices each available slot. The sum of all excess weights plus Kc is
an upper bound on every support using at most K slots, including negative ballot weights. -/
theorem capacity_upper {I : Type} (w : I → Int) (c : Int) (on : I → Bool) (xs : List I)
    (capacity : Nat) (nonneg : 0 ≤ c) (fits : xs.countP on ≤ capacity) :
    weight w on xs ≤ (capacity : Int)*c + excess w c xs := by
  have hf : (xs.countP on : Int) ≤ (capacity : Int) := by exact_mod_cast fits
  have mono := Int.mul_le_mul_of_nonneg_right hf nonneg
  have := affine_upper w c on xs
  omega

/-- A strict failure of an upper vote rules out every completion. Equality remains feasible
under the closed-ball definition. Forced-in coordinates contribute a constant `known`. -/
theorem forbid_branch {I : Type} (w : I → Int) (c : Int) (xs : List I)
    (capacity : Nat) (known quota : Int) (nonneg : 0 ≤ c)
    (fails : known + (capacity : Int)*c + excess w c xs < quota)
    (on : I → Bool) (fits : xs.countP on ≤ capacity) : known + weight w on xs < quota := by
  have := capacity_upper w c on xs capacity nonneg fits
  omega

private theorem threshold_exact {I : Type} (w : I → Int) (c : Int) (top : I → Bool)
    (separates : ∀ i, (top i = true → c ≤ w i) ∧ (top i = false → w i ≤ c)) (xs : List I) :
    weight w top xs = (xs.countP top : Int)*c + excess w c xs := by
  induction xs with
  | nil => simp [weight, excess]
  | cons i xs ih =>
    have h := separates i
    cases b : top i <;> simp [weight, excess, b, Int.add_mul] at h ih ⊢ <;> omega

/-- A top-positive mask attains the bound, provided its cutoff is zero or the capacity
is filled. This includes a capacity larger than the number of positive weights. -/
theorem capacity_exact {I : Type} (w : I → Int) (c : Int) (top : I → Bool)
    (separates : ∀ i, (top i = true → c ≤ w i) ∧ (top i = false → w i ≤ c)) (xs : List I)
    (capacity : Nat) (filled : c = 0 ∨ xs.countP top = capacity) :
    weight w top xs = (capacity : Int)*c + excess w c xs := by
  rw [threshold_exact w c top separates xs]
  rcases filled with h | h <;> simp [h]


private theorem square_nonneg (a : Int) : 0 ≤ a*a := by
  by_cases h : 0 ≤ a
  · exact Int.mul_nonneg h h
  · exact Int.mul_nonneg_of_nonpos_of_nonpos (by omega) (by omega)


#print axioms forbid_branch
#print axioms capacity_exact

def dot {J : Type} (a b : J → Int) (xs : List J) : Int := (xs.map (fun i ↦ a i * b i)).sum

private theorem length_nonneg {J : Type} (a : J → Int) (xs : List J) : 0 ≤ dot a a xs := by
  induction xs with
  | nil => simp [dot]
  | cons i xs ih =>
    have hh := square_nonneg (a i)
    simp only [dot, List.map_cons, List.sum_cons] at ih ⊢
    omega

private theorem null_dot {J : Type} (a b : J → Int) (xs : List J) (zero : dot b b xs = 0) :
    dot a b xs = 0 := by
  induction xs with
  | nil => simp [dot]
  | cons i xs ih =>
    have positive := length_nonneg b xs
    have sq := square_nonneg (b i)
    change b i * b i + dot b b xs = 0 at zero
    have prod : b i * b i = 0 := by omega
    have bi : b i = 0 := by rcases Int.mul_eq_zero.mp prod with v | v <;> exact v
    change a i * b i + dot a b xs = 0
    simp [bi,ih (by omega)]

private theorem sum_quadratic {J : Type} (a b : J → Int) (xs : List J) (s t : Int) :
    dot (fun i ↦ s*a i - t*b i) (fun i ↦ s*a i - t*b i) xs =
    s*s*dot a a xs - 2*s*t*dot a b xs + t*t*dot b b xs := by
  induction xs with
  | nil => simp [dot]
  | cons i xs ih =>
    simp only [dot, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih]
    grind

private theorem cauchy {J : Type} (a b : J → Int) (xs : List J) :
    dot a b xs * dot a b xs ≤ dot a a xs * dot b b xs := by
  have nn := length_nonneg b xs
  by_cases h : dot b b xs = 0
  · simp [h, null_dot a b xs h]
  · have strict : 0 < dot b b xs := by omega
    have pos := length_nonneg (fun i ↦ dot b b xs*a i-dot a b xs*b i) xs
    rw [sum_quadratic] at pos
    have factor :
        dot b b xs*dot b b xs*dot a a xs-2*dot b b xs*dot a b xs*dot a b xs+
        dot a b xs*dot a b xs*dot b b xs =
        dot b b xs*(dot a a xs*dot b b xs-dot a b xs*dot a b xs) := by grind
    rw [factor] at pos
    apply Classical.byContradiction
    intro wrong
    have hh := Int.mul_neg_of_pos_of_neg strict
      (by omega : dot a a xs*dot b b xs-dot a b xs*dot a b xs < 0)
    omega

/-- Finite-dimensional directional duality for exact integer vectors. A failing direction
can always be chosen equal to the residual vector a; no division is needed. -/
theorem finite_ball {J : Type} (a : J → Int) (xs : List J) (s : Int) (radius : 0 ≤ s) :
    dot a a xs ≤ s ↔ ∀ v : J → Int, dot a v xs ≤ 0 ∨
        dot a v xs * dot a v xs ≤ s * dot v v xs := by
  constructor
  · intro inside v
    exact Or.inr (Int.le_trans (cauchy a v xs)
      (Int.mul_le_mul_of_nonneg_right inside (length_nonneg v xs)))
  · intro votes
    rcases votes a with nonpos | squared
    · omega
    · apply Classical.byContradiction
      intro wrong
      have strict := Int.mul_lt_mul_of_pos_right (by omega : s < dot a a xs)
        (by omega : 0 < dot a a xs)
      omega

#print axioms finite_ball
end MinPI.Directional
