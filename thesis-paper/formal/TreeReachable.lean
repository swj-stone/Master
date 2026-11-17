import Std

/-! Structural path requirements, independently of forest weights and training rows. -/
namespace MinPI.TreeReachable

variable {α : Type}

def Active (support : α → Prop) (path : List α) : Prop :=
  ∀ j ∈ path, support j

def Reach (paths : List (List α)) (support : α → Prop) (j : α) : Prop :=
  ∃ h ∈ paths, j ∈ h ∧ Active support h

theorem contractive {paths : List (List α)} {s : α → Prop} {j : α}
    (h : Reach paths s j) : s j := by
  obtain ⟨_, _, member, enabled⟩ := h
  exact enabled j member

theorem monotone {paths : List (List α)} {s r : α → Prop}
    (inc : ∀ j, s j → r j) : ∀ j, Reach paths s j → Reach paths r j := by
  intro j h
  obtain ⟨p, hp, member, enabled⟩ := h
  exact ⟨p, hp, member, fun k hk => inc k (enabled k hk)⟩

/-- Every recorded edge activates on exactly the same future masks after compaction. -/
theorem masked_active_iff {paths : List (List α)} {s t : α → Prop}
    {h : List α} (recorded : h ∈ paths) :
    Active (fun j => s j ∧ t j) h ↔ Active (fun j => Reach paths s j ∧ t j) h := by
  constructor
  · intro both j belongs
    exact ⟨⟨h, recorded, belongs, fun k hk => (both k hk).1⟩, (both j belongs).2⟩
  · intro both j belongs
    exact ⟨contractive (both j belongs).1, (both j belongs).2⟩

theorem idempotent {paths : List (List α)} {s : α → Prop} {j : α} :
    Reach paths (Reach paths s) j ↔ Reach paths s j := by
  constructor
  · exact contractive
  · rintro ⟨h, recorded, member, enabled⟩
    exact ⟨h, recorded, member, fun k hk => ⟨h, recorded, hk, enabled⟩⟩

/-- A support reproducing every originally active path contains the canonical support. -/
theorem least {paths : List (List α)} {s r : α → Prop}
    (preserves : ∀ h ∈ paths, Active s h → Active r h) :
    ∀ j, Reach paths s j → r j := by
  rintro j ⟨h, recorded, member, enabled⟩
  exact preserves h recorded enabled j member

end MinPI.TreeReachable
