(* Append one tied bin above all existing bins in the AUC midrank formula. *)
ClearAll[triangle, oldPos, oldNeg, pos, neg, metrics];
triangle[x_] := x (x + 1)/2;
newRank = oldPos + oldNeg + (pos + neg + 1)/2;
rankStep = Expand[pos newRank - triangle[oldPos + pos] + triangle[oldPos]];
checkStep = Simplify[rankStep == pos (oldNeg + neg/2)];

(* Ordered score bins for the two fitted examples in ensemble_cells/examples.json.
   The first three evaluation rows are positive. Only score bin 6 predicts positive. *)
allowed = {{4, 5, 6}, {4, 5, 6}, {6}, {4, 5, 6}, {1, 2, 3}, {1}};
measuredBins = {6, 5, 6, 4, 2, 1};
metrics[bins_] := Module[{positive = Take[bins, 3], negative = Drop[bins, 3], wins},
  wins = Flatten[Outer[Function[{p, n}, (1 + Sign[p - n])/2], positive, negative]];
  {Mean[wins], (Count[positive, 6] + Count[negative, Except[6]])/6}
];
worlds = Tuples[allowed];
pooled = Select[worlds, Sort[#] == Sort[measuredBins] &];
observed = Union[metrics /@ pooled];
checks = {checkStep, Length[worlds] == 81, Length[Union[metrics /@ worlds]] == 11,
  Length[pooled] == 6, observed == {{13/18, 1/2}, {8/9, 5/6}, {1, 5/6}}};
If[!And @@ checks, Abort[]];
checks
