import Std

namespace MinPI.ProposalCost

/-- Costs of a veto on a passing proposal and a missed failing proposal, respectively. -/
def loss (fp fn paid missed : Nat) := fp * paid + fn * missed

/-- Pairwise break-even of the separating minimum against the endpoint rule on the path sample. -/
theorem beats_endpoints (paid missed : Nat) :
    loss 194 5 paid missed < loss 174 16 paid missed ↔ 20 * paid < 11 * missed := by
  unfold loss
  omega

/-- On the observed path counts, one of two feasible alternatives strictly improves the default
    for every positive false-veto cost and nonnegative missed-loss cost. -/
theorem path_default_off_frontier (paid missed : Nat) (positive : 0 < paid) :
    loss 126 24 paid missed < loss 194 5 paid missed ∨
    loss 211 0 paid missed < loss 194 5 paid missed := by
  unfold loss
  omega

/-- On the drawn proposals the endpoints strictly improve cost at every finite cost ratio. -/
theorem drawn_endpoints_cheaper (paid missed : Nat) (positive : 0 < paid) :
    loss 169 2 paid missed < loss 178 2 paid missed := by
  unfold loss
  omega

#print axioms path_default_off_frontier
end MinPI.ProposalCost
