import Std

/-! Finite mask averages before division by their common denominator.
The one-coordinate union-Gram identity adds over coordinates without any
independence hypothesis. Group indices may repeat. -/
namespace MinPI.TreeMaskQuota

def total {I : Type} (xs : List I) (f : I → Int) := (xs.map f).sum

private theorem cons_total {I : Type} (x : I) (xs : List I) (f : I → Int) :
    total (x::xs) f = f x + total xs f := by simp [total]

private theorem congr_total {I : Type} (xs : List I) (f g : I → Int)
    (h : ∀ x, f x = g x) : total xs f = total xs g := by
  have eq : f = g := funext h
  rw [eq]

private theorem zero_total {I : Type} (xs : List I) : total xs (fun _ ↦ 0) = 0 := by
  induction xs <;> simp_all [total]

private theorem add_total {I : Type} (xs : List I) (f g : I → Int) :
    total xs (fun i ↦ f i + g i) = total xs f + total xs g := by
  induction xs with
  | nil => simp [total]
  | cons x xs ih => simp only [cons_total, ih]; omega

private theorem mul_total {I : Type} (xs : List I) (k : Int) (g : I → Int) :
    total xs (fun i ↦ k*g i) = k*total xs g := by
  induction xs with
  | nil => simp [total]
  | cons x xs ih => simp only [cons_total, ih]; grind

private theorem commute {I J : Type} (xs : List I) (ys : List J) (f : I → J → Int) :
    total xs (fun x ↦ total ys (f x)) = total ys (fun y ↦ total xs (fun x ↦ f x y)) := by
  induction xs with
  | nil => simp only [total, List.map_nil, List.sum_nil]; exact (zero_total ys).symm
  | cons x xs ih => simp only [cons_total, ih, add_total]

private theorem expand {I J : Type} (xs : List I) (ys : List J) (a : I → Int) (b : J → Int) :
    total xs a * total ys b = total xs (fun i ↦ total ys (fun j ↦ a i*b j)) := by
  induction xs with
  | nil => simp [total]
  | cons x xs ih => rw [cons_total, cons_total, Int.add_mul, ih, mul_total]

/-- Exact finite mask loss.  With a zero-one incidence function m, the innermost
sum on the right counts masks that contain both of the two required sets. -/
theorem union_gram {I M : Type} (groups : List I) (masks : List M) (a : I → Int)
    (m : M → I → Int) :
    total masks (fun t ↦ total groups (fun i ↦ a i*m t i) * total groups (fun i ↦ a i*m t i)) =
    total groups (fun i ↦ total groups (fun j ↦ a i*a j*total masks (fun t ↦ m t i*m t j))) := by
  calc
    _ = total masks (fun t ↦ total groups (fun i ↦ total groups (fun j ↦ (a i*m t i)*(a j*m t j)))) := by
      apply congr_total; intro t; exact expand groups groups _ _
    _ = total groups (fun i ↦ total masks (fun t ↦ total groups (fun j ↦ (a i*m t i)*(a j*m t j)))) :=
      commute masks groups _
    _ = total groups (fun i ↦ total groups (fun j ↦ total masks (fun t ↦ (a i*m t i)*(a j*m t j)))) := by
      apply congr_total; intro i; exact commute masks groups _
    _ = _ := by
      apply congr_total; intro i
      apply congr_total; intro j
      rw [← mul_total]
      apply congr_total; intro t; ac_rfl

/-- Any mixture of integer-weighted size votes is bounded below by the least
single-size vote, scaled by the total weight. Concentrating weight at an actual
minimizer attains this bound. -/
theorem mixture_lower {I : Type} (sizes : List I) (w slack : I → Int) (least : Int)
    (positive : ∀ s, 0 ≤ w s) (bound : ∀ s, least ≤ slack s) :
    least*total sizes w ≤ total sizes (fun s ↦ w s*slack s) := by
  induction sizes with
  | nil => simp [total]
  | cons s sizes ih =>
    have one := Int.mul_le_mul_of_nonneg_left (bound s) (positive s)
    simp only [cons_total]
    rw [Int.mul_add, Int.mul_comm least (w s)]
    omega

#print axioms union_gram
#print axioms mixture_lower
end MinPI.TreeMaskQuota
