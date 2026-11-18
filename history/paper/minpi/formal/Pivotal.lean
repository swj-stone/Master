import Std

/-! Exact counting for the ternary witness game, using only the Lean standard library. -/
namespace MinPI

/-- The three possible signs of a coordinate's hit/miss comparison. -/
inductive Vote where
  | plus | minus | abstain
  deriving DecidableEq

/-- Counts of positive, negative and abstaining peers in one coalition. -/
structure Profile where
  pos : Nat
  neg : Nat
  zero : Nat
  deriving DecidableEq

/-- Insert the current peer into the profile of a coalition. -/
def insertVote (v : Vote) (t : Profile) : Profile :=
  match v with
  | .plus => {t with pos := t.pos + 1}
  | .minus => {t with neg := t.neg + 1}
  | .abstain => {t with zero := t.zero + 1}

/-- Tally the whole peer population. -/
def tally : List Vote → Profile
  | [] => ⟨0, 0, 0⟩
  | v :: vs => insertVote v (tally vs)

/-- Enumerate indexed subsets by omitting or including successive peers.
Profiles may repeat; each repetition represents another subset of indices. -/
def subsetProfiles : List Vote → List Profile
  | [] => [⟨0, 0, 0⟩]
  | v :: vs =>
    let old := subsetProfiles vs
    old ++ old.map (insertVote v)

/-- Count occurrences of a coalition profile, including repeats. -/
def occurrences (target : Profile) (xs : List Profile) : Nat :=
  xs.countP (fun x ↦ decide (x = target))

/-- Pascal's recursion, with value zero outside the feasible range. -/
def choose : Nat → Nat → Nat
  | 0, 0 => 1
  | 0, _ + 1 => 0
  | _ + 1, 0 => 1
  | n + 1, k + 1 => choose n (k + 1) + choose n k

@[simp] private theorem choose_zero (n : Nat) : choose n 0 = 1 := by
  cases n <;> rfl

private theorem map_hits (xs : List Profile) (f : Profile → Profile) (to_ from_ : Profile)
    (h : ∀ x, f x = to_ ↔ x = from_) :
    occurrences to_ (xs.map f) = occurrences from_ xs := by
  simp [occurrences, List.countP_map, Function.comp_def, h]

private theorem map_misses (xs : List Profile) (f : Profile → Profile) (t : Profile)
    (h : ∀ x, f x ≠ t) : occurrences t (xs.map f) = 0 := by
  simp [occurrences, List.countP_map, Function.comp_def, h]

private theorem split_count (x : Profile) (l r : List Profile) :
    occurrences x (l ++ r) = occurrences x l + occurrences x r := by
  simp [occurrences]

/-- The multiplicity of an exact profile in the enumeration is the three-binomial product. -/
theorem profileMultiplicity (peers : List Vote) (a b c : Nat) :
    occurrences ⟨a, b, c⟩ (subsetProfiles peers) =
    choose (tally peers).pos a * choose (tally peers).neg b * choose (tally peers).zero c := by
  induction peers generalizing a b c with
  | nil =>
    cases a <;> cases b <;> cases c <;> simp [occurrences, subsetProfiles, tally, choose]
  | cons v vs ih =>
    rw [subsetProfiles, split_count, ih]
    cases v with
    | plus =>
      cases a with
      | zero =>
        rw [map_misses _ _ _ (by intro x; simp [insertVote, Profile.mk.injEq])]
        simp [tally, insertVote]
      | succ a =>
        rw [map_hits _ _ _ ⟨a, b, c⟩ (by intro x; cases x; simp [insertVote]), ih]
        simp [tally, insertVote, choose, Nat.add_mul]
    | minus =>
      cases b with
      | zero =>
        rw [map_misses _ _ _ (by intro x; simp [insertVote, Profile.mk.injEq])]
        simp [tally, insertVote]
      | succ b =>
        rw [map_hits _ _ _ ⟨a, b, c⟩ (by intro x; cases x; simp [insertVote]), ih]
        simp [tally, insertVote, choose, Nat.mul_add, Nat.add_mul]
    | abstain =>
      cases c with
      | zero =>
        rw [map_misses _ _ _ (by intro x; simp [insertVote, Profile.mk.injEq])]
        simp [tally, insertVote]
      | succ c =>
        rw [map_hits _ _ _ ⟨a, b, c⟩ (by intro x; cases x; simp [insertVote]), ih]
        simp [tally, insertVote, choose, Nat.mul_add]

/-- Cardinality of the subset described by a profile. -/
def size (t : Profile) : Nat := t.pos + t.neg + t.zero

private theorem size_insert (v : Vote) (x : Profile) :
    size (insertVote v x) = size x + 1 := by
  cases v <;> simp [size, insertVote] <;> omega

/-- There are exactly (number of peers choose k) indexed coalitions of size k. -/
theorem sizeMultiplicity (peers : List Vote) (k : Nat) :
    (subsetProfiles peers).countP (fun x ↦ decide (size x = k)) = choose peers.length k := by
  induction peers generalizing k with
  | nil => cases k <;> simp [subsetProfiles, size, choose]
  | cons v vs ih =>
    cases k <;>
      simp [subsetProfiles, List.countP_map, Function.comp_def, size_insert, ih, choose]

/-- Signed majority outcome, with ties assigned zero. -/
def outcome (z : Int) : Int := if z < 0 then -1 else if z = 0 then 0 else 1

/-- The change caused by adding a positive ballot occurs exactly at margins -1 and 0. -/
theorem positivePivot (z : Int) :
    outcome (z + 1) - outcome z = (if z = -1 ∨ z = 0 then 1 else 0) := by
  unfold outcome
  repeat' split
  all_goals omega

/-- The change caused by adding a negative ballot occurs exactly at margins 0 and 1. -/
theorem negativePivot (z : Int) :
    outcome (z - 1) - outcome z = (if z = 0 ∨ z = 1 then -1 else 0) := by
  unfold outcome
  repeat' split
  all_goals omega

/-- An abstention contributes zero for every peer margin. -/
theorem zeroPivot (z : Int) : outcome (z + 0) - outcome z = 0 := by simp

/-- Twice the payoff increment from a focal binary vote against one binary peer. -/
theorem binaryPair (v u : Int) (hv : v = 1 ∨ v = -1) (hu : u = 1 ∨ u = -1) :
    2 * (outcome (u + v) - outcome u) = v - u := by
  rcases hv with rfl | rfl <;> rcases hu with rfl | rfl <;> decide

/-- Summing over all one-peer coalitions yields p*v minus the full row margin,
where p is one plus the length of the peer list. -/
theorem binaryOnePeer (v : Int) (peers : List Int) (hv : v = 1 ∨ v = -1)
    (hpeers : ∀ u ∈ peers, u = 1 ∨ u = -1) :
    2 * (peers.map (fun u ↦ outcome (u + v) - outcome u)).sum =
      (peers.length : Int) * v - peers.sum := by
  induction peers with
  | nil => simp
  | cons u us ih =>
    have pair := binaryPair v u hv (hpeers u (by simp))
    have tail := ih (fun x hx ↦ hpeers x (by simp [hx]))
    simp only [List.map_cons, List.sum_cons, List.length_cons, Int.natCast_succ, Int.add_mul,
      Int.one_mul, Int.mul_add]
    omega

/-- A strict order between the selected lower key and the outside upper key transfers
to the two true scores, including the smaller-index rule at equality. -/
theorem indexedSeparation (lo s t hi : Int) (j i : Nat) (lower : lo ≤ s) (upper : t ≤ hi)
    (wins : hi < lo ∨ (hi = lo ∧ j < i)) : t < s ∨ (t = s ∧ j < i) := by omega

/-- If at least h scores meet a threshold, screening smaller upper bounds keeps the entire
first h entries of any descending score order, including its fixed order among tied scores. -/
theorem screenedPrefix {ι α : Type} [LE α] [DecidableLE α] [Std.IsPreorder α]
    (score upper : ι → α) (τ : α) (bound : ∀ i, score i ≤ upper i)
    (ids : List ι) (h : Nat) (sorted : ids.Pairwise (fun i j ↦ score j ≤ score i))
    (enough : h ≤ ids.countP (fun i ↦ decide (τ ≤ score i))) :
    (ids.filter (fun i ↦ decide (τ ≤ upper i))).take h = ids.take h := by
  induction ids generalizing h with
  | nil => simp
  | cons i rest ih =>
    cases h with
    | zero => simp
    | succ h =>
      have head : τ ≤ score i := by
        by_cases hit : τ ≤ score i
        · exact hit
        · have none : rest.countP (fun j ↦ decide (τ ≤ score j)) = 0 := by
            apply List.countP_eq_zero.mpr
            intro j hj
            simp only [decide_eq_true_eq]
            exact fun hjτ ↦ hit (Std.le_trans hjτ ((List.pairwise_cons.mp sorted).1 j hj))
          simp [hit, none] at enough
      have keep := Std.le_trans head (bound i)
      have tailCount : h ≤ rest.countP (fun j ↦ decide (τ ≤ score j)) := by
        simpa [List.countP_cons, head] using enough
      simp [keep, ih h (List.pairwise_cons.mp sorted).2 tailCount]

/-- With positive denominator, flooring a signed scaled numerator loses less
than one denominator. A divisible numerator loses nothing. -/
theorem gridTerm (n d : Int) (hd : 0 < d) :
    n / d * d ≤ n ∧ n ≤ n / d * d + (if n % d = 0 then 0 else d) := by
  have h₁ := Int.ediv_mul_add_emod n d
  have h₂ := Int.emod_nonneg n (by omega : d ≠ 0)
  have h₃ := Int.emod_lt_of_pos n hd
  split <;> omega

/-- A layer sums signed numerators with one common denominator. The sum of
the individual integer enclosures encloses its exact numerator. -/
theorem gridLayer (xs : List Int) (d : Int) (hd : 0 < d) :
    (xs.map (fun n ↦ n / d * d)).sum ≤ xs.sum ∧
      xs.sum ≤ (xs.map (fun n ↦ n / d * d + (if n % d = 0 then 0 else d))).sum := by
  induction xs with
  | nil => simp
  | cons n ns ih =>
    have h := gridTerm n d hd
    simp only [List.map_cons, List.sum_cons]
    omega

/-- Strict reversal of two anchor scores is equivalent to a larger difference
in their penalties. Rational scores can be expressed in common integer units. -/
theorem penaltyFlip (a b u v : Int) : a - u < b - v ↔ a - b < u - v := by omega

/-- The h-th largest score, counting multiplicity and allowing a tie at the cutoff. -/
def IsRankCut {α : Type} (ids : List α) (f : α → Int) (h : Nat) (t : Int) : Prop :=
  h ≤ ids.countP (fun j ↦ decide (t ≤ f j)) ∧ ids.countP (fun j ↦ decide (t < f j)) < h

/-- A pointwise one-sided error bound carries through to the h-th largest score. -/
theorem rankCutBound {α : Type} (ids : List α) (a b : α → Int) (h : Nat) (ca cb d : Int)
    (ra : IsRankCut ids a h ca) (rb : IsRankCut ids b h cb)
    (bounded : ∀ j ∈ ids, a j ≤ b j + d) : ca ≤ cb + d := by
  by_cases good : ca ≤ cb + d
  · exact good
  exfalso
  have count : ids.countP (fun j ↦ decide (ca ≤ a j)) ≤
      ids.countP (fun j ↦ decide (cb < b j)) := by
    apply List.countP_mono_left
    intro j hj ha
    simp only [decide_eq_true_eq] at ha ⊢
    have := bounded j hj
    omega
  exact (Nat.not_lt_of_ge (Nat.le_trans ra.1 count)) rb.2


/-- In common integer units, both endpoint cutoffs and the individual
enclosure width control the distance of an unresolved score to rank h. -/
theorem unresolvedBand (score theta lower upper lowCut highCut eta : Int)
    (enclosed : lower ≤ score ∧ score ≤ upper) (width : upper - lower ≤ eta)
    (cutoffs : theta - eta ≤ lowCut ∧ highCut ≤ theta + eta)
    (unresolved : lowCut ≤ upper ∧ lower ≤ highCut) :
    theta - 2*eta ≤ score ∧ score ≤ theta + 2*eta := by omega

/-- Scaling a signed numerator by its nonnegative common-denominator multiplier bounds
the exact product by the common denominator in both directions. -/
theorem scaledRowBound (n d factor : Int) (hn : -d ≤ n ∧ n ≤ d) (hf : 0 ≤ factor) :
    -(d * factor) ≤ n * factor ∧ n * factor ≤ d * factor := by
  constructor
  · simpa only [Int.neg_mul] using Int.mul_le_mul_of_nonneg_right hn.1 hf
  · exact Int.mul_le_mul_of_nonneg_right hn.2 hf

private theorem boundedSum (xs : List Int) (d : Int) (terms : ∀ z ∈ xs, -d ≤ z ∧ z ≤ d) :
    -((xs.length : Int) * d) ≤ xs.sum ∧ xs.sum ≤ (xs.length : Int) * d := by
  induction xs with
  | nil => simp
  | cons a xs ih =>
    have head := terms a (by simp)
    have tail := ih (fun z hz ↦ terms z (by simp [hz]))
    simp only [List.length_cons, List.sum_cons, Int.natCast_succ, Int.add_mul, Int.one_mul]
    omega

/-- Every subsum of bounded scaled row contributions lies in the symmetric signed-word
range. This includes separate sign accumulators and leaves their final score safe to negate. -/
theorem nativePartialSum (xs ys : List Int) (d : Int) (hd : 0 ≤ d) (part : ys.Sublist xs)
    (terms : ∀ z ∈ xs, -d ≤ z ∧ z ≤ d) (capacity : (xs.length : Int) * d ≤ 2^63 - 1) :
    -(2^63 - 1) ≤ ys.sum ∧ ys.sum ≤ 2^63 - 1 := by
  have size : (ys.length : Int) ≤ (xs.length : Int) := by exact_mod_cast part.length_le
  have bound := Int.mul_le_mul_of_nonneg_right size hd
  have partialBounds := boundedSum ys d (fun z hz ↦ terms z (part.subset hz))
  omega

/-- An actual survivor of both cutoff tests lies in the twice-width band. The cutoff
hypotheses identify order statistics with ties, without assuming any endpoint error bound. -/
theorem unresolvedAtRank {α : Type} (ids : List α) (s lo hi : α → Int) (h : Nat)
    (theta lowCut highCut eta : Int) (j : α) (member : j ∈ ids)
    (enclosures : ∀ i ∈ ids, lo i ≤ s i ∧ s i ≤ hi i ∧ hi i - lo i ≤ eta)
    (rs : IsRankCut ids s h theta) (rl : IsRankCut ids lo h lowCut)
    (rh : IsRankCut ids hi h highCut) (survives : lowCut ≤ hi j ∧ lo j ≤ highCut) :
    theta - 2*eta ≤ s j ∧ s j ≤ theta + 2*eta := by
  have lower := rankCutBound ids s lo h theta lowCut eta rs rl (by
    intro i hi
    obtain ⟨_, hs, hw⟩ := enclosures i hi
    omega)
  have upper := rankCutBound ids hi s h highCut theta eta rh rs (by
    intro i hi
    obtain ⟨hs, _, hw⟩ := enclosures i hi
    omega)
  obtain ⟨hlo, hhi, hw⟩ := enclosures j member
  exact unresolvedBand _ _ _ _ _ _ _ ⟨hlo, hhi⟩ hw ⟨by omega, upper⟩ survives

#print axioms binaryOnePeer
#print axioms indexedSeparation
#print axioms unresolvedAtRank

#print axioms scaledRowBound
#print axioms nativePartialSum

#print axioms unresolvedBand

#print axioms gridTerm
#print axioms gridLayer
#print axioms penaltyFlip
#print axioms profileMultiplicity
#print axioms sizeMultiplicity
#print axioms positivePivot
#print axioms negativePivot
#print axioms zeroPivot
#print axioms screenedPrefix

end MinPI
