(* Enumerate a squared-loss game from vector predictions, independently of its
   quadratic expansion. Rational entries allow exact comparisons, including ties. *)
ClearAll[gameChecks];
gameChecks[a_, r_] := Module[
  {p = Length[a], n, g, d, pair, util, marg, bounds, fix, exchange, checkWidth, subsets},
  n = Range[p]; g = a.Transpose[a]; d = 2 a.r - Diagonal[g]; subsets = Subsets[n];
  util[s_] := -Total[(r - Total[Prepend[a[[s]], 0 r]])^2];
  marg[i_, s_] := util[Append[s, i]] - util[s];
  bounds = And @@ Table[With[{others = DeleteCases[n, i]},
    Module[{gs = g[[i, others]], vals, low, high, mean, neg, pos},
      vals = marg[i, #] & /@ Subsets[others];
      pos = Select[others, g[[i, #]] > 0 &]; neg = Select[others, g[[i, #]] < 0 &];
      low = d[[i]] - 2 Total[Ramp[gs]]; high = d[[i]] + 2 Total[Ramp[-gs]];
      mean = d[[i]] - Total[gs];
      And[Min[vals] == low == marg[i, pos], Max[vals] == high == marg[i, neg],
        Mean[vals] == mean, low == mean - Total[Abs[gs]], high == mean + Total[Abs[gs]],
        And @@ Table[Min[marg[i, #] & /@ Subsets[DeleteCases[others, h]]] ==
             low + 2 Ramp[g[[i, h]]], {h, others}]]]], {i, n}];
  fix = And @@ Table[Module[{scores, opts},
    scores = util[#] - penalty Length[#] & /@ subsets;
    opts = Pick[subsets, scores, Max[scores]];
    And @@ Table[With[{v = marg[i, #] & /@ Subsets[DeleteCases[n, i]]},
      Implies[Min[v] > penalty, AllTrue[opts, MemberQ[#, i] &]] &&
      Implies[Max[v] < penalty, AllTrue[opts, FreeQ[#, i] &]]], {i, n}]],
    {penalty, {-2, -1, 0, 1, 2}}];
  pair[i_, j_, k_] := d[[i]] - d[[j]] - 2 Total[TakeLargest[
    (g[[i, #]] - g[[j, #]]) & /@ Complement[n, {i, j}], k - 1]];
  checkWidth[k_] := Module[{supp, opts, edge, exact, closed, counts, trans},
    supp = Subsets[n, {k}]; opts = Pick[supp, util /@ supp, Max[util /@ supp]];
    edge[i_, j_] := i != j && pair[i, j, k] > 0;
    exact = And @@ Flatten[Table[
      i == j || pair[i, j, k] == Min[
        (util[Append[#, i]] - util[Append[#, j]]) & /@ Subsets[Complement[n, {i, j}], {k - 1}]],
      {i, n}, {j, n}]];
    closed = And @@ Flatten[Table[Implies[edge[i, j] && MemberQ[s, j], MemberQ[s, i]],
      {i, n}, {j, n}, {s, opts}]];
    counts = And @@ Flatten[Table[
      Implies[Count[edge[#, j] & /@ n, True] >= k, FreeQ[s, j]] &&
      Implies[Count[edge[j, #] & /@ n, True] >= p - k, MemberQ[s, j]], {j, n}, {s, opts}]];
    trans = And @@ Flatten[Table[Implies[edge[i, j] && edge[j, h], edge[i, h]], {i,n},{j,n},{h,n}]];
    And[exact, closed, counts, trans]];
  And[bounds, fix, And @@ Table[checkWidth[k], {k, p-1}]]];
checks = {gameChecks[{{1, -2}, {1, 1}, {-1, 2}, {0, 0}, {2, -1}}/3, {2, 1}/3],
          gameChecks[{{1, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 1}}, {1, 2}],
          gameChecks[ConstantArray[{0, 0}, 4], {1, 0}]};
If[!TrueQ[And @@ checks], Abort[]];
checks
