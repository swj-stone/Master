(* Exact rational checks of the finite-population count and its replacement approximation. *)
ClearAll[peerList, finiteMass, finiteMarginal, enumerationMarginal, replacementMass,
  replacementMarginal, pivotalIncrement];

peerList[a_, b_, c_] := Join[ConstantArray[1, a], ConstantArray[-1, b], ConstantArray[0, c]];

finiteMass[a_, b_, c_, k_, margin_] :=
  Sum[If[i - j == margin && 0 <= k - i - j <= c,
    Binomial[a, i] Binomial[b, j] Binomial[c, k-i-j], 0],
    {i, 0, a}, {j, 0, b}]/Binomial[a+b+c, k];

pivotalIncrement[mass_, vote_] := Switch[vote,
  1, mass[-1] + mass[0], -1, -mass[0] - mass[1], 0, 0];

finiteMarginal[a_, b_, c_, k_, vote_] :=
  pivotalIncrement[Function[t, finiteMass[a, b, c, k, t]], vote];

enumerationMarginal[a_, b_, c_, k_, vote_] :=
  Mean[(Sign[Total[#] + vote] - Sign[Total[#]]) & /@ Subsets[peerList[a, b, c], {k}]];

(* Using polynomial coefficients avoids indeterminate powers 0^0 at boundary proportions. *)
replacementMass[a_, b_, c_, k_, margin_] :=
  Coefficient[Expand[((a x^2 + c x + b)/(a+b+c))^k], x, margin+k];

replacementMarginal[a_, b_, c_, k_, v_] :=
  pivotalIncrement[Function[t, replacementMass[a+Boole[v == 1], b+Boole[v == -1],
      c+Boole[v == 0], k, t]], v];

checks = Flatten[Table[
  With[{exact = finiteMarginal[a,b,c,k,v], n = a+b+c+1},
    {exact == enumerationMarginal[a,b,c,k,v],
     Abs[exact - replacementMarginal[a,b,c,k,v]] <= 1 - FactorialPower[n-1,k]/n^k}],
  {a,0,5}, {b,0,5-a}, {c,0,5-a-b}, {k,0,a+b+c}, {v,-1,1}], 4];
(* Check every margin, including zero counts outside the attainable range. *)
verifyMargins[] := Module[{histogram, trials = 0, errors = 0},
  Do[
    histogram = Counts[Total /@ Subsets[peerList[a,b,c], {k}]];
    Do[
      trials++;
      If[Lookup[histogram, margin, 0] !=
         Binomial[a+b+c, k] finiteMass[a,b,c,k,margin], errors++],
      {margin, -k, k}],
    {a,0,5}, {b,0,5-a}, {c,0,5-a-b}, {k,0,a+b+c}];
  {trials, errors}
];

cases = Partition[Flatten[checks],2];
{Length[cases], Count[cases[[All,1]],False], Count[cases[[All,2]],False]}
