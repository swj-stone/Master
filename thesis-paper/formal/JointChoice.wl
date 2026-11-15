(* Equal savings d; w is the probability of choosing the first candidate. *)
ClearAll[d, w, regret, gap];
regret[w_, {a_, b_}] := d (Max[a, b] - (w a + (1 - w) b));
gap[{a_, b_}] := d (a - b);
dominant = {{0, 0}, {1, 0}, {1, 1}};
bothOrders = {{0, 1}, {1, 0}, {1, 1}};
worst[worlds_, q_] := Max @@ (regret[q, #] & /@ worlds);
checks = {
  Simplify[worst[dominant, 1] == 0, d > 0],
  Resolve[ForAll[{d, w}, Implies[d > 0 && 0 <= w <= 1, worst[bothOrders, w] >= d/2]], Reals],
  Simplify[worst[bothOrders, 1/2] == d/2, d > 0],
  Sort[DeleteDuplicates[gap /@ dominant /. d -> 2/7]] == {0, 2/7},
  Sort[DeleteDuplicates[gap /@ bothOrders /. d -> 2/7]] == {-2/7, 0, 2/7}
};
If[!And @@ checks, Abort[]];
checks
