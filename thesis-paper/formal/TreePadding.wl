(* Exact checks of padding and the collision bound for normalized coalition votes. *)
ClearAll[b, at, err];
b[p_, s_, d_] := Binomial[s - 1, d - 1]/Binomial[p - 1, d - 1];
at[p_, s_, d_] := If[1 <= s <= p, b[p,s,d], 0];
elevation = And @@ Flatten[Table[
  b[p+1,s,d] == ((p+1-s) at[p,s,d] + (s-1) at[p,s-1,d])/p,
  {p,1,18}, {d,1,p}, {s,1,p+1}]];
envelope = And @@ Flatten[Table[
  With[{u=(s-1)/(p-1), k=d-1},
    Abs[If[k==0,1,If[u==0,0,u^k]]-b[p,s,d]] <= Min[1,Binomial[k,2]/(p-1)]],
  {p,2,18}, {d,1,p}, {s,1,p}]];
mixture = And @@ Flatten[Table[
  Sum[Binomial[p-1,s-1] q^(s-1) (1-q)^(p-s) b[p,s,d],{s,p}] == q^(d-1),
  {p,1,14},{d,1,p},{q,{1/13,1/2,12/13}}]];
{elevation, envelope, mixture}
