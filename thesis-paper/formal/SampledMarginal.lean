import Std

/-!
The rational-arithmetic lemmas below verify the bordered ridge normal equations and the change
in a held-out row's squared residual. The binary-game lemmas are independent of the learner.
-/
namespace MinPI.SampledMarginal

def dot {ι : Type} (indices : List ι) (x y : ι → Rat) : Rat :=
  (indices.map fun i ↦ x i * y i).sum

theorem dot_sub_scaled {ι : Type} (indices : List ι) (x b c : ι → Rat) (g : Rat) :
    dot indices x (fun i ↦ b i - c i * g) = dot indices x b - dot indices x c * g := by
  induction indices with
  | nil => simp [dot]; grind
  | cons i tail ih =>
    simpa only [dot, List.map_cons, List.sum_cons] using
      (show x i * (b i - c i * g) + dot tail x (fun j ↦ b j - c j * g) =
        (x i * b i + dot tail x b) - (x i * c i + dot tail x c) * g by rw [ih]; grind)

/-- A column correction satisfying the scalar Schur equation solves both enlarged block rows. -/
theorem bordered_normal {ι : Type} (indices : List ι)
    (A : ι → ι → Rat) (b c q rhs : ι → Rat) (d t g : Rat)
    (old : ∀ j, dot indices (A j) b = rhs j) (projection : ∀ j, dot indices (A j) c = q j)
    (scalar : (d - dot indices q c) * g = t - dot indices q b) :
    (∀ j, dot indices (A j) (fun i ↦ b i - c i * g) + q j * g = rhs j) ∧
      dot indices q (fun i ↦ b i - c i * g) + d * g = t := by
  constructor
  · intro j
    rw [dot_sub_scaled, old, projection]
    grind
  · rw [dot_sub_scaled]
    grind

theorem validation_prediction {ι : Type} (indices : List ι) (z b c : ι → Rat) (x g : Rat) :
    dot indices z (fun i ↦ b i - c i * g) + x * g =
      dot indices z b + (x - dot indices z c) * g := by
  rw [dot_sub_scaled]
  grind

theorem squared_residual (e r g : Rat) :
    (e - r * g) * (e - r * g) - e * e = r * r * g * g - 2 * r * g * e := by grind

def marginal (before after : Bool) : Int :=
  (if after then 1 else 0) - (if before then 1 else 0)

theorem negative_iff (a b : Bool) : marginal a b = -1 ↔ a = true ∧ b = false := by
  cases a <;> cases b <;> decide

theorem absorb (a b : Bool) : min (-1 : Int) (marginal a b) = -1 := by
  cases a <;> cases b <;> decide

theorem skip_losing_base_after_zero (b : Bool) : min (0 : Int) (marginal false b) = 0 := by
  cases b <;> decide

def sampledMinimum : List (Bool × Bool) → Int
  | [] => 1
  | (a,b) :: tail => min (marginal a b) (sampledMinimum tail)

theorem sampled_lower (pairs : List (Bool × Bool)) : -1 ≤ sampledMinimum pairs := by
  induction pairs with
  | nil => decide
  | cons pair tail ih =>
    obtain ⟨a,b⟩ := pair
    cases a <;> cases b <;> simp [sampledMinimum, marginal, Int.min_def] <;> omega

theorem negative_witness (pairs : List (Bool × Bool)) :
    sampledMinimum pairs = -1 ↔ (true,false) ∈ pairs := by
  induction pairs with
  | nil => decide
  | cons pair tail ih =>
    obtain ⟨a,b⟩ := pair
    have bound := sampled_lower tail
    rw [List.mem_cons, ← ih]
    cases a <;> cases b <;> simp [sampledMinimum, marginal, Int.min_def] <;> (try split) <;> omega

theorem witnessed_value_exact {sample all : List (Bool × Bool)}
    (contained : ∀ x ∈ sample, x ∈ all) (found : sampledMinimum sample = -1) :
    sampledMinimum all = -1 :=
  (negative_witness all).2 (contained _ ((negative_witness sample).1 found))

#print axioms witnessed_value_exact
#print axioms bordered_normal
#print axioms validation_prediction
#print axioms squared_residual
#print axioms absorb
#print axioms skip_losing_base_after_zero

end MinPI.SampledMarginal
