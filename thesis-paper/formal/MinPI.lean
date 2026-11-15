import Std

/-!
Deterministic guarantees for context-voted deletion. A support is an ordered list
of feature identifiers (the application supplies distinct identifiers). The fixed Boolean
predicate includes all panel-specific and metric-specific tests. Prediction, scoring and
data partitioning are outside the formal interface.
-/

namespace MinPI

/-- One strict majority vetoes as soon as its context is visited. -/
def scanVotes (voters : Nat) : List Nat → Bool
  | [] => true
  | q :: qs => decide (voters < 2 * q) && scanVotes voters qs

/-- The initial value is neutral for counts in `[0, voters]`. -/
def minimumCount (voters : Nat) : List Nat → Nat
  | [] => voters
  | q :: qs => min q (minimumCount voters qs)

theorem scan_eq_min (voters : Nat) (h : 0 < voters) (qs : List Nat) :
    scanVotes voters qs = decide (voters < 2 * minimumCount voters qs) := by
  induction qs with
  | nil =>
    have hp : voters < 2 * voters := by omega
    simp [scanVotes, minimumCount, hp]
  | cons q qs ih =>
    apply Bool.eq_iff_iff.mpr
    simp only [scanVotes, ih, Bool.and_eq_true, decide_eq_true_eq]
    change (_ ∧ _) ↔ voters < 2 * min q (minimumCount voters qs)
    by_cases hlow : q ≤ minimumCount voters qs
    · simp only [Nat.min_eq_left hlow]; omega
    · simp only [Nat.min_eq_right (Nat.le_of_lt (Nat.lt_of_not_ge hlow))]; omega

/-- Each visited context consumes one oracle query before memoization. -/
def visits (voters : Nat) : List Nat → Nat
  | [] => 0
  | q :: qs => 1 + if voters < 2 * q then visits voters qs else 0

theorem visits_le (voters : Nat) (qs : List Nat) : visits voters qs ≤ qs.length := by
  induction qs with
  | nil => simp [visits]
  | cons q qs ih =>
    simp only [visits, List.length_cons]
    split <;> omega

abbrev Support := List Nat

/-- A mask splits uniquely into its anchor and external components. This is the
    disjoint-counting step in the exact cache cost formula. -/
theorem mask_injective {α : Type} {anchor s t e f : α → Prop}
    (hs : ∀ x, s x → anchor x) (ht : ∀ x, t x → anchor x)
    (he : ∀ x, e x → ¬anchor x) (hf : ∀ x, f x → ¬anchor x)
    (masks : ∀ x, (s x ∨ e x) ↔ (t x ∨ f x)) :
    (∀ x, s x ↔ t x) ∧ (∀ x, e x ↔ f x) := by
  constructor <;> intro x <;> specialize masks x
  · constructor
    · intro hx
      exact (masks.mp (.inl hx)).elim id (fun h ↦ False.elim (hf x h (hs x hx)))
    · intro hx
      exact (masks.mpr (.inl hx)).elim id (fun h ↦ False.elim (he x h (ht x hx)))
  · constructor
    · intro hx
      exact (masks.mp (.inr hx)).elim (fun h ↦ False.elim (he x hx (ht x h))) id
    · intro hx
      exact (masks.mpr (.inr hx)).elim (fun h ↦ False.elim (hf x hx (hs x h))) id

/-- A partial panel count fixes a majority precisely when an endpoint of its completion
    interval fixes it. For four panels, three revealed yes or two revealed no suffice. -/
theorem partial_votes (voters yes unknown : Nat) :
    ((∀ extra ≤ unknown, voters < 2 * (yes + extra)) ↔ voters < 2 * yes) ∧
    ((∀ extra ≤ unknown, 2 * (yes + extra) ≤ voters) ↔ 2 * (yes + unknown) ≤ voters) := by
  constructor
  · constructor
    · intro h; simpa using h 0 (Nat.zero_le _)
    · intro h extra _; omega
  · constructor
    · intro h; exact h unknown (Nat.le_refl _)
    · intro h extra he; omega

/-- If the weakest context has `q` positives, fewer than this many yes-to-no
    flips preserve every majority; spending exactly this many there defeats its majority. -/
theorem accepted_vote_radius (voters q : Nat) (pass : voters < 2 * q) :
    0 < q - voters / 2 ∧
    (∀ count r : Nat, q ≤ count → r < q - voters / 2 → voters < 2 * (count - r)) ∧
    2 * (q - (q - voters / 2)) ≤ voters := by
  refine ⟨by omega, ?_, ?_⟩
  · intro count r hq hr
    omega
  · omega

def canDelete (ok : Support → Bool) (s : Support) (j : Nat) : Bool :=
  decide (1 < s.length ∧ j ∈ s) && ok (s.erase j)

def step (ok : Support → Bool) (s : Support) (j : Nat) : Support :=
  if canDelete ok s j then s.erase j else s

theorem step_size (ok : Support → Bool) (s : Support) (j : Nat) :
    (step ok s j).length ≤ s.length := by
  simp only [step]; split
  next h =>
    have hm : j ∈ s := (of_decide_eq_true (Bool.and_eq_true_iff.mp h).1).2
    rw [List.length_erase_of_mem hm]; omega
  next => exact Nat.le_refl _

theorem step_no_drop (ok : Support → Bool) (s : Support) (j : Nat)
    (h : (step ok s j).length = s.length) :
    canDelete ok s j = false ∧ step ok s j = s := by
  cases hc : canDelete ok s j with
  | false => simp [step, hc]
  | true =>
    have hguard := of_decide_eq_true (Bool.and_eq_true_iff.mp hc).1
    simp only [step, hc, ite_true, List.length_erase_of_mem hguard.2] at h
    omega

def sweep (ok : Support → Bool) : List Nat → Support → Support
  | [], s => s
  | j :: js, s => sweep ok js (step ok s j)

theorem sweep_size (ok : Support → Bool) (js s : Support) :
    (sweep ok js s).length ≤ s.length := by
  induction js generalizing s with
  | nil => exact Nat.le_refl _
  | cons j js ih => exact Nat.le_trans (ih _) (step_size ok s j)

theorem unchanged_rejects (ok : Support → Bool) (js s : Support)
    (h : (sweep ok js s).length = s.length) :
    ∀ j ∈ js, canDelete ok s j = false := by
  induction js generalizing s with
  | nil => simp
  | cons j js ih =>
    have hle := sweep_size ok js (step ok s j)
    have hstep := step_size ok s j
    change (sweep ok js (step ok s j)).length = s.length at h
    obtain ⟨hno,hs⟩ := step_no_drop ok s j (by omega)
    rw [hs] at h
    intro i hi
    rcases List.mem_cons.mp hi with heq | ht
    · simpa [heq] using hno
    · exact ih s h i ht

theorem sweep_feasible (ok : Support → Bool) (js s : Support)
    (h : ok s = true ∧ 0 < s.length) :
    ok (sweep ok js s) = true ∧ 0 < (sweep ok js s).length := by
  induction js generalizing s with
  | nil => exact h
  | cons j js ih =>
    apply ih
    simp only [step]; split
    next hc =>
      obtain ⟨hg, ha⟩ := Bool.and_eq_true_iff.mp hc
      have hguard := of_decide_eq_true hg
      exact ⟨ha, by rw [List.length_erase_of_mem hguard.2]; omega⟩
    next => exact h

/-- Scan all remaining features; a strict size drop funds the next complete scan. -/
def settle (ok : Support → Bool) (s : Support) : Support :=
  let t := sweep ok s.reverse s
  if t.length < s.length then settle ok t else s
termination_by s.length
decreasing_by assumption

def Certified (ok : Support → Bool) (s : Support) : Prop :=
  ok s = true ∧ 0 < s.length ∧ (1 < s.length → ∀ j ∈ s, ok (s.erase j) = false)

theorem settle_certified (ok : Support → Bool) (s : Support)
    (h : ok s = true ∧ 0 < s.length) : Certified ok (settle ok s) := by
  rw [settle]
  split
  next hlt => exact settle_certified ok _ (sweep_feasible ok s.reverse s h)
  next hn =>
    have hsize := sweep_size ok s.reverse s
    have hr := unchanged_rejects ok s.reverse s (by omega)
    refine ⟨h.1, h.2, ?_⟩
    intro hs j hj
    simpa [canDelete, hs, hj] using hr j (List.mem_reverse.mpr hj)
termination_by s.length

/-- A fixed predicate accepting the complete triple and one singleton separates local
    irreducibility from minimum feasible cardinality. -/
theorem local_can_exceed_global :
    ∃ (ok : Support → Bool) (s t : Support),
      Certified ok s ∧ ok t = true ∧ 0 < t.length ∧ t.length < s.length ∧ t.Sublist s := by
  refine ⟨fun xs ↦ decide (xs.length = 3 ∨ xs = [1]), [0,1,2], [1], ?_⟩
  refine ⟨⟨by decide, by decide, ?_⟩, by decide, by decide, by decide, by decide⟩
  intro _ j hj
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
  rcases hj with rfl | rfl | rfl <;> decide

#print axioms scan_eq_min
#print axioms visits_le
#print axioms mask_injective
#print axioms partial_votes
#print axioms accepted_vote_radius
#print axioms settle_certified
#print axioms local_can_exceed_global

end MinPI
