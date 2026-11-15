(* Exact finite check of the signed-predictor example and the five-feature design. *)
ClearAll[rows, truth, prediction, metrics, reference, candidate, failed, covered];
rows = Tuples[{-1, 1}, 3];
truth = (#[[1]] #[[2]] &) /@ rows;
prediction[keep_, tail_] := (
    If[keep, #[[1]], 0] (
      If[MemberQ[tail, 1], #[[2]], 0] + If[MemberQ[tail, 2], #[[3]], 0])
  &) /@ rows;
metrics[p_] := {Mean[Abs[truth - p]], Max[0, 1 - Mean[(truth - p)^2]]};
reference = metrics[prediction[True, #]] & /@ {{}, {1}, {2}, {1, 2}};
candidate = metrics[prediction[False, {}]];
failed = (candidate[[1]] - #[[1]] > 3/100 Abs[#[[1]]] + 10^-12 ||
          #[[2]] - candidate[[2]] > 3/100 Abs[#[[2]]] + 10^-12 &) /@ reference;
features = Range[5];
nested = {{}, {1, 2, 3}, features};
signatures = Take[Subsets[Range[4], {2}], Length[features]];
separating = Table[Select[features, MemberQ[signatures[[#]], bit] &], {bit, 4}];
covered[family_] := Count[Permutations[features, {2}],
    {u_, v_} /; AnyTrue[family, Function[e, MemberQ[e, u] && !MemberQ[e, v]]]];
result = <|"ReferenceMAEandClippedR2" -> reference,
  "CandidateMAEandClippedR2" -> candidate, "Veto" -> failed,
  "DirectedPairsNested" -> covered[nested], "DirectedPairsSeparating" -> covered[separating]|>;
If[Values[result] =!= {{{1, 0}, {0, 1}, {1, 0}, {1, 0}}, {1, 0},
    {False, True, False, False}, 6, 20}, Abort[]];
result
