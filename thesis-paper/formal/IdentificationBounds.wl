(* Independent real-algebra checks for the sharp two-score identified set. *)
ClearAll[p, q, u, a, m, z];
masses = {u, p - u, q - u, 1 - p - q + u};
feasible = And @@ Thread[masses >= 0];
bounds = Max[0, p + q - 1] <= u <= Min[p, q];
auc = (2 masses[[1]] masses[[4]] + masses[[1]] masses[[3]] +
        masses[[2]] masses[[4]])/(2 p (1 - p));
acc = masses[[1]] + masses[[4]];
checks = {
  Resolve[ForAll[{p,q,u}, Equivalent[feasible, bounds]], Reals],
  FullSimplify[auc == 1/2 + (u - p q)/(2 p (1-p)), 0 < p < 1],
  Simplify[acc == 1 - p - q + 2 u],
  FullSimplify[And[D[auc,u] > 0, D[acc,u] > 0], 0 < p < 1],
  FullSimplify[And[auc == 2 u, acc == 2 u] /. {p->1/2,q->1/2}],
  Resolve[ForAll[{a,m,z}, Implies[a>0 && m>0 && 0<=z<=1,
    Max[a z,m (1-z)] >= a m/(a+m)]], Reals],
  FullSimplify[Max[a z,m (1-z)] == a m/(a+m) /. z->m/(a+m), a>0 && m>0]
};
If[!And @@ checks, Abort[]];
checks
