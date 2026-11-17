(* Enumerate masks independently of the symbolic Gram coefficients. Group
   indices may repeat even if their required feature sets coincide. *)
ClearAll[check];
check[p_,sets_,parts_] := Module[{cols=Range[p],n=Length[First[parts]],masks,pred,
    losses,gram,base,quota=1/25,sizes=Range[0,p],allsets,degree,z},
  allsets=Subsets[cols]; masks[s_]:=Subsets[cols,{s}];
  pred[t_] := Total[Prepend[Pick[parts,(SubsetQ[t,#]& /@ sets),True],ConstantArray[0,n]]];
  losses[keep_,s_] := Mean[(With[{e=pred[#]-pred[Intersection[#,keep]]},e.e])& /@ masks[s]];
  gram[keep_,s_] := Module[{dis=Select[Range[Length[sets]],!SubsetQ[keep,sets[[#]]]&],u},
    Total[Flatten[Table[u=Length[Union[sets[[i]],sets[[j]]]];
      parts[[i]].parts[[j]] Product[s-t,{t,0,u-1}]/(u! Binomial[p,u]),{i,dis},{j,dis}]]]];
  base=gram[{},z];
  degree=Min[p,2 Max[Length /@ sets]];
  And@@Flatten[Table[Module[{poly=Expand[gram[k,z]],delta,min,f=w/Total[w]},
    delta=Table[quota (base/.z->s) - losses[k,s],{s,sizes}];min=Min[delta];
    PolynomialQ[poly,z] && Exponent[poly,z]<=degree &&
    And@@Table[(poly/.z->s)==losses[k,s],{s,sizes}] &&
    f.delta>=min && Min[IdentityMatrix[p+1].delta]==min],
    {k,allsets},{w,{Range[p+1],Reverse[Range[p+1]],ConstantArray[1,p+1]}}]]];
checks={check[1,{{1}},{{0,0}}],
  check[3,{{1},{1},{1,2},{2,3},{3}},{{1,2},{-1,0},{0,3},{-2,1},{4,0}}],
  check[4,{{1},{2},{2,4},{1,2,3}},{{1/2},{-1/2},{2},{1}}],
  (* A zero denominator can coexist with positive loss. Full prediction at
     size p cancels, while retaining only feature 1 gives error one. *)
  Total[{1,-1}]^2==0 && (Total[{1,-1}]-1)^2==1,
  (* Exchangeability does not promise fidelity for a specified individual mask. *)
  With[{f={1,1/10}},(1/10)^2/Total[f^2]<=1/25 && (1/10)^2/Total[f]^2<=1/25 &&
    (1/10)^2>(1/10)^2/25]};
If[!TrueQ[And@@checks],Abort[]];checks
