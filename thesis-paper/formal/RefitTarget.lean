import Std

namespace MinPI.RefitTarget

/-- Conditional class frequencies out of 100 for concordant and discordant R. -/
def atoms (p : Nat) (y : Int) : List (Int × Nat) := [(y, p), (-y, 100-p)]

def credit (a b : Int) : Nat := if a > b then 2 else if a = b then 1 else 0

/-- Twice the number of correctly ordered positive-negative pairs, of 10,000. -/
def auc2 (p : Nat) (f : Int → Int → Int) : Nat :=
  ((atoms p 1).map fun (r, n) =>
    ((atoms p (-1)).map fun (s, m) => n*m*credit (f 1 r) (f (-1) s)).sum).sum

/-- Number of correct classifications in 200 cases, using the positive class on ties. -/
def acc (p : Nat) (f : Int → Int → Int) : Nat :=
  ((atoms p 1).map fun (r,n) => if f 1 r ≥ 0 then n else 0).sum +
  ((atoms p (-1)).map fun (s,m) => if f (-1) s < 0 then m else 0).sum

def masked (showD showR : Bool) (d r : Int) : Int :=
  (if showD then d else 0) * (2 + if showR then r else 0)

theorem every_mask_agrees (d r : Bool) :
    (auc2 99 (masked d r), acc 99 (masked d r)) =
    (auc2 50 (masked d r), acc 50 (masked d r)) := by
  cases d <;> cases r <;> decide

theorem surviving_column_scores :
    (auc2 99 (fun _ r => r), acc 99 (fun _ r => r)) = (19800,198) ∧
    (auc2 50 (fun _ r => r), acc 50 (fun _ r => r)) = (10000,100) := by decide

#print axioms every_mask_agrees
#print axioms surviving_column_scores
end MinPI.RefitTarget
