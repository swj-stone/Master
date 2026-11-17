import Std

/-!
# A two-endpoint exchange bound for fixed cardinality

For one voting coordinate, u and v are target minus prediction at the two supports. Each
element e of `differences` is one removed contribution minus one added contribution, in
any one-to-one pairing. Their sum is v-u. Sum the coordinate identity over rows and outputs
to obtain the squared Euclidean loss identity. No positive definiteness is needed for the
following integer identities. The lower bound is expressed in doubled utility units.
-/
namespace MinPI.Radius

def fromA (u e : Int) : Int := (u + e) * (u + e) - u*u
def fromT (v e : Int) : Int := v*v - (v - e) * (v - e)

theorem two_ends (u v e : Int) : fromA u e + fromT v e = 2 * (u+v) * e := by
  unfold fromA fromT
  grind

private theorem sum_ends (u v : Int) (es : List Int) :
    (es.map (fun e ↦ fromA u e + fromT v e)).sum = 2*(u+v)*es.sum := by
  induction es with
  | nil => simp
  | cons e es ih =>
    change (fromA u e + fromT v e) + (es.map (fun v' ↦ fromA u v' + fromT v v')).sum =
      2 * (u+v) * (e + es.sum)
    rw [ih, two_ends]
    exact (Int.mul_add _ _ _).symm

/-- Each pairing has the same total two-endpoint vote, twice the true utility gap. -/
theorem polarization (u v : Int) (differences : List Int) (balance : differences.sum = v-u) :
    (differences.map (fun e ↦ fromA u e + fromT v e)).sum = 2*(v*v-u*u) := by
  rw [sum_ends, balance]
  grind

private theorem sum_lower {I : Type} (cost gain : I → Int) (pairs : List I)
    (valid : ∀ i ∈ pairs, cost i ≤ gain i) : (pairs.map cost).sum ≤ (pairs.map gain).sum := by
  induction pairs with
  | nil => simp
  | cons i is ih =>
    have one := valid i (by simp)
    have tail := ih (fun x hx ↦ valid x (by simp [hx]))
    simp only [List.map_cons, List.sum_cons]
    omega

/-- Any lower bound that applies to every possible size-r matching applies in particular
to the chosen pairing. The local vote is evaluated; the second vote may be bounded. -/
theorem matching_bound {I : Type} (endA endT lowT : I → Int) (pairing : List I)
    (gap bound : Int) (lower : ∀ i ∈ pairing, lowT i ≤ endT i)
    (valid : bound ≤ (pairing.map (fun i ↦ endA i + lowT i)).sum)
    (identity : (pairing.map (fun i ↦ endA i + endT i)).sum = 2*gap) :
    bound ≤ 2*gap := by
  have per i (hi : i ∈ pairing) : endA i + lowT i ≤ endA i + endT i := by
    have := lower i hi
    omega
  have := sum_lower (fun i ↦ endA i + lowT i) (fun i ↦ endA i + endT i) pairing per
  omega

theorem certified_optimal {C : Type} [DecidableEq C] (score : C → Int)
    (radius : C → Nat) (bound : Nat → Int) (a : C)
    (valid : ∀ t, t ≠ a → bound (radius t) ≤ 2*(score a-score t))
    (certified : ∀ t, t ≠ a → 0 ≤ bound (radius t)) :
    ∀ t, score t ≤ score a := by
  intro t
  by_cases same : t = a
  · simp [same]
  · have := valid t same
    have := certified t same
    omega

#print axioms polarization
#print axioms matching_bound
#print axioms certified_optimal
end MinPI.Radius
