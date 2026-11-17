(* Columns of A are grouped by the number of additional required features.
   Check the Pascal span, the sharp norm over the true affine fibre, equality
   for all-size votes, and the integer signed-polynomial test in exact arithmetic. *)
ClearAll[v, ballot, check];
v[p_,d_,s_] := Table[Product[s-i,{i,0,q-1}]/(q! Binomial[p-1,q]),{q,0,d-1}];
ballot[h_,x_] := If[h.h==0,0,(h.x)^2/(h.h)];
pascal = And@@Flatten[Table[With[{m=Table[v[p,d,s],{s,0,d-1}]},
      Det[m]==Product[1/Binomial[p-1,q],{q,0,d-1}]],
    {p,1,12},{d,1,p}]];
check[a_,p_,x_] := Module[{d=Length[First[a]],raw,dirs,k,g,proj,delta,ballots,
    mixture,fibre,equality,polynomial,z},
  raw=Table[v[p,d,s],{s,0,p-1}];
  dirs=(a.#)& /@ raw;
  k=Transpose[a].a;g=Transpose[a].x;proj=a.PseudoInverse[k].g;
  delta=x-proj;ballots=ballot[#,x]& /@ dirs;mixture=g.PseudoInverse[k].g;
  fibre=NullSpace[Transpose[a]];
  equality=proj.proj==0 || AnyTrue[dirs,#.#>0 && MatrixRank[{proj,#}]<=1&];
  And[MatrixRank[Take[dirs,d]]==MatrixRank[a], MatrixRank[dirs]==MatrixRank[a],
    Transpose[a].proj==g, mixture==proj.proj, proj.proj+delta.delta==x.x,
    And@@((proj.#==0)& /@ fibre), Max[ballots]<=mixture,
    Equivalent[Max[ballots]==mixture,equality],
    And@@Table[
      polynomial=Expand[(v[p,d,z].g)^2-t v[p,d,z].k.v[p,d,z]];
      PolynomialQ[polynomial,z] && Exponent[polynomial,z]<=2d-2 &&
        And@@Table[Equivalent[ballots[[s+1]]>t,(polynomial/.z->s)>0],{s,0,p-1}],
      {t,Union[{0,1/2,2},ballots]}]]];
strict=With[{a=IdentityMatrix[2],x={1,-1}},
  Max[Table[ballot[a.v[3,2,s],x],{s,0,2}]]==1 && x.x==2];
checks={pascal,strict,check[IdentityMatrix[2],3,{1,-1}],
  check[{{1,-2,0},{2,-4,0},{-1,2,0}},6,{1,1,-1}],
  check[ConstantArray[0,{3,2}],4,{1,2,-3}],
  check[{{1,0,1},{0,1,1},{0,0,0}},7,{1,-2,3}],
  check[{{2,-3},{0,1},{1,1}},2,{3/2,-5/3,1/4}]};
If[!TrueQ[And@@checks],Abort[]];checks
