(* Exact enumeration in the two worlds. Use an integer-valued model score. *)
ClearAll[weights, perf, mask, refit];
weights[y_, r_, p_] := If[y == r, p, 1 - p];
perf[f_, p_] := {
  Sum[weights[1,r,p] weights[-1,s,p] (1 + Sign[f[1,r]-f[-1,s]])/2,
      {r,{-1,1}}, {s,{-1,1}}],
  Sum[weights[y,r,p] Boole[If[f[y,r] >= 0, 1, -1] == y]/2, {y,{-1,1}}, {r,{-1,1}}]};
mask[d_,e_][y_,r_] := d y (2+e r);
refit[y_,r_] := r;
maskResults = Table[perf[mask[d,e],p], {p,{99/100,1/2}}, {d,0,1}, {e,0,1}];
surviving = Table[perf[refit,p],{p,{99/100,1/2}}];
If[maskResults[[1]] =!= maskResults[[2]] || surviving =!= {{99/100,99/100},{1/2,1/2}}, Abort[]];
{First[maskResults],surviving}
