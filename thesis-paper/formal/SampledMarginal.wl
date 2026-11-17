(* The scalar Schur equation, the old normal rows, and the validation loss identity. *)
ClearAll[d, qTb, qTc, t, r, g, e, coeff, lossShift, lambda];
coeff = (t - qTb)/(d - qTc);
lossShift = (e - r g)^2 - e^2;

(* An exact multioutput ridge test with a correlated candidate and a constant column. *)
train = {{1, 1, 3}, {0, 1, 2}, {1, 1, 1}, {2, 1, 4}};
test = {{-1, 1, 2}, {3, 1, 1}, {0, 1, 0}};
targets = {{0, 1}, {1, 0}, {0, 1}, {1, 0}};
ytest = {{1, 0}, {0, 1}, {1, 0}};
center[m_, mean_] := (# - mean) & /@ m;
x = center[train, Mean[train]]; z = center[test, Mean[train]];
y = center[targets, Mean[targets]]; u = center[ytest, Mean[targets]];
lambda = 1;
fit[s_] := Inverse[Transpose[x[[All, s]]].x[[All, s]] + lambda IdentityMatrix[Length[s]]].
             Transpose[x[[All,s]]].y;
mse[m_] := Total[Flatten[m]^2]/Length[m];
check[s_, i_] := Module[{xs = x[[All,s]], zs = z[[All,s]], xi = x[[All,i]],
                          zi = z[[All,i]], h, b, c, den, a, ext, resid, testResid},
  h = Transpose[xs].xs + lambda IdentityMatrix[Length[s]];
  b = fit[s]; c = LinearSolve[h, Transpose[xs].xi];
  den = xi.xi + lambda - (Transpose[xs].xi).c;
  a = (xi.(y - xs.b))/den;
  resid = zi - zs.c; testResid = u - zs.b;
  ext = Join[b - Outer[Times,c,a],{a}];
  {den >= lambda, ext == fit[Append[s,i]],
   zs.b + Outer[Times,resid,a] == z[[All,Append[s,i]]].ext,
   mse[u-z[[All,Append[s,i]]].ext]-mse[testResid] ==
      ((resid.resid) (a.a) - 2 resid.testResid.a)/Length[test]}];
checks = {Together[(d - qTc) coeff - t + qTb] === 0,
          Expand[lossShift - r^2 g^2 + 2 r g e] === 0,
          And @@ check[{1,2},3], And @@ check[{1,3},2]};
If[!TrueQ[And @@ checks], Abort[]]; checks
