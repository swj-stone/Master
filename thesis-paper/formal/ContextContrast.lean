import Std

/-!
The context margin is positive exactly when enough panels vote against a deletion.
These statements concern the directed two-feature contrast studied in the paper.
They do not assume the fitted predictor actually has such a margin.
-/

namespace MinPI.ContextContrast

/-- A feature's signature consists of the query indices that restore it. -/
def Below {I : Type} (s t : I → Bool) : Prop :=
  ∀ i, s i = true → t i = true

/-- A separating context exposes the increasing feature but omits the cancelling feature. -/
def Witness {I : Type} (s t : I → Bool) : Prop :=
  ∃ i, s i = true ∧ t i = false

theorem witness_iff_not_below {I : Type} (s t : I → Bool) :
    Witness s t ↔ ¬ Below s t := by
  constructor
  · rintro ⟨i, hs, ht⟩ h
    have := h i hs
    simp_all
  · intro h
    classical
    apply Classical.byContradiction
    intro no
    apply h
    intro i hs
    cases ht : t i with
    | false => exact False.elim (no ⟨i, hs, ht⟩)
    | true => rfl

/-- Detecting both orientations for every pair is equivalent to an antichain of signatures. -/
theorem covers_iff_antichain {F I : Type} (code : F → I → Bool) :
    (∀ u v, u ≠ v → Witness (code u) (code v)) ↔
      (∀ u v, u ≠ v → ¬ Below (code u) (code v)) := by
  simp only [witness_iff_not_below]

/-- Rational version of the perturbed detection margin, including equality at a passing boundary. -/
theorem directed_margin (base signal radius noise : Rat) (present absent : Bool)
    (hn : -radius ≤ noise ∧ noise ≤ radius) (hr : 0 ≤ radius)
    (hb : base + radius ≤ 0) (hs : 0 < base + signal - radius) :
    (0 < base + (if present then signal else 0) - (if absent then signal else 0) + noise)
      ↔ present = true ∧ absent = false := by
  cases present <;> cases absent <;> simp <;> grind

/-- The three ordered extensions have one informative bit, membership in their middle set. -/
def prefixSignature (inMiddle : Bool) : Fin 3 → Bool
  | 0 => false
  | 1 => inMiddle
  | 2 => true

theorem prefix_witness (u v : Bool) :
    Witness (prefixSignature u) (prefixSignature v) ↔ u = true ∧ v = false := by
  constructor
  · rintro ⟨⟨i, hi⟩, hu, hv⟩
    match i with
    | 0 => simp [prefixSignature] at hu
    | 1 => exact ⟨hu, hv⟩
    | 2 => simp [prefixSignature] at hv
    | i + 3 => omega
  · rintro ⟨rfl, rfl⟩
    exact ⟨1, rfl, rfl⟩

/-- Two empirical veto predicates can coincide on every queried mask and disagree at any
    omitted mask. The manuscript realizes them with zero-background regression predictors. -/
theorem omitted_mask {I Mask : Type} (query : I → Mask) (target : Mask)
    (miss : ∀ i, query i ≠ target) :
    ∃ alwaysPass isolated : Mask → Prop,
      (∀ i, alwaysPass (query i) ↔ isolated (query i)) ∧
      alwaysPass target ∧ ¬ isolated target := by
  refine ⟨fun _ ↦ True, fun e ↦ e ≠ target, ?_, trivial, fun h ↦ h rfl⟩
  intro i
  exact ⟨fun _ ↦ miss i, fun _ ↦ trivial⟩

#print axioms covers_iff_antichain
#print axioms directed_margin
#print axioms prefix_witness
#print axioms omitted_mask

end MinPI.ContextContrast
