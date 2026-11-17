(* The five-feature example and the triangular binomial system use exact rationals. *)
ClearAll[share, cases, means, minima, checks];
share[k_] := With[{chance = Binomial[3, k]/Binomial[4, k]},
  {5, 11 chance, 11 chance, 1, 1}/(7 + 22 chance)];
cases = Table[share[k], {k, {1, 2, 3}}];
means = Mean[cases];
minima = Min /@ Transpose[cases];
checks = {
  cases[[All, 1]] == {10/47, 5/18, 2/5},
  cases[[All, 2]] == {33/94, 11/36, 11/50},
  means[[1]] - means[[2]] == 589/126900,
  minima[[2]] - minima[[1]] == 17/2350,
  And @@ Table[Det[Table[Binomial[n-j,n-i],{i,n},{j,n}]] == 1, {n,1,24}]
};
If[!TrueQ[And @@ checks], Abort[]];
Print[{cases, checks}];
