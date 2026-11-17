(* Exact rational utilities. Integrate the piecewise constant marginal ballot on
   intervals between consecutive endpoints. With identical base utilities this
   expression reduces to the smallest signed change. *)
ClearAll[cellIntegral, negativeUnionLength, intervalFormula, cases, checks];
cellIntegral[with_, without_] := Module[{cuts, cells},
  cuts = Union[{0, 1}, with, without];
  cells = Partition[cuts, 2, 1];
  Total[Function[c, With[{t = Mean[c]},
    (Last[c] - First[c]) Min[MapThread[Boole[#1 >= t] - Boole[#2 >= t] &,
      {with, without}]]]] /@ cells]];
negativeUnionLength[with_, without_] := Last[Fold[
  Function[{state, pair}, {
    Max[First[state], Last[pair]],
    Last[state] + Max[0, Last[pair] - Max[First[state], First[pair]]]}],
  {0, 0}, Sort[Select[Transpose[{with, without}], First[#] < Last[#] &]]]];
intervalFormula[with_, without_] :=
  Max[0, Min[with] - Max[without]] - negativeUnionLength[with, without];

cases = Tuples[Range[0, 3]/3, 3];
checks = {
  And @@ Flatten[Table[cellIntegral[w, b] == intervalFormula[w, b],
    {w, cases}, {b, cases}]],
  And @@ Flatten[Table[
    cellIntegral[w, ConstantArray[a, 3]] == Min[w] - a,
    {w, cases}, {a, Range[0, 3]/3}]],
  (* Positive swings can occur at disjoint quotas when the anchors differ. *)
  cellIntegral[{1/3, 1}, {0, 2/3}] == 0 &&
    Min[{1/3, 1} - {0, 2/3}] == 1/3
};
If[!TrueQ[And @@ checks], Abort[]];
checks
