(* Enumerate the finite decision space. Each of the three contexts has
   four Boolean ballots. Counts describe the truth table, not their
   probability under fitted data or under the paired bootstrap. *)
ClearAll[rows, triples, strictContext, minRule, pooled, shortRule, visits];
rows = Tuples[{0, 1}, 12];
triples[r_] := Partition[r, 4];
strictContext[g_] := Total[g] >= 3;
minRule[r_] := Min[Total /@ triples[r]] >= 3;
pooled[r_] := Total[r] >= 7;
shortRule[r_] := AllTrue[triples[r], strictContext];
visits[r_] := Length[TakeWhile[Most[triples[r]], strictContext]] + 1;
report = <|
  "BallotArrays" -> Length[rows],
  "AcceptedByMin" -> Count[minRule /@ rows, True],
  "AcceptedByPooled" -> Count[pooled /@ rows, True],
  "ShortCircuitMismatches" -> Count[rows, r_ /; shortRule[r] =!= minRule[r]],
  "MinButNotPooled" -> Count[rows, r_ /; minRule[r] && !pooled[r]],
  "ContextVisits1to3" -> Table[Count[visits /@ rows, k], {k, 1, 3}]
|>;
If[Values[report] =!= {4096, 125, 1586, 0, 0, {2816, 880, 400}},
   Print[report]; Exit[1]];
Print[report];
