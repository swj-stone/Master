(* Directly score every available mask, independently of the Gram representation. *)
ClearAll[fixture];
fixture[p_, groups_, parts_, intercept_, target_] := Module[
  {sets=Subsets[Range[p]],m=Length[groups],zero=0 target,z=target-intercept,unions,w,
   weight,raw,excess,by,inc,both,normalized,agree,updates,mixture,tol=1/20},
  unions=DeleteDuplicates[Join[groups,Flatten[Table[Union[g,h],{g,groups},{h,groups}],1]]];
  w[u_]:=w[u]=2 z.Total[Prepend[Pick[parts,(#==u& /@ groups),True],zero]] -
    Sum[If[Union[groups[[i]],groups[[j]]]==u,parts[[i]].parts[[j]],0],{i,m},{j,m}];
  weight[s_,u_]:=Binomial[s-1,Length[u]-1]/Binomial[p-1,Length[u]-1];
  raw[t_]:=raw[t]=intercept+Total[Prepend[Pick[parts,(SubsetQ[t,#]& /@ groups),True],zero]];
  excess[k_,s_]:=excess[k,s]=Mean[Function[t,With[{a=target-raw[Intersection[k,t]],b=target-raw[t]},
     a.a-b.b]] /@ Subsets[Range[p],{s}]];
  by[k_,s_]:=Total[(If[SubsetQ[k,#],0,weight[s,#]w[#]]& /@ unions)];
  inc[k_,j_,s_]:=Total[(If[SubsetQ[k,#] && MemberQ[#,j],weight[s,#]w[#],0]& /@ unions)];
  both[k_,j_,i_,s_]:=Total[(If[SubsetQ[k,#] && MemberQ[#,j] && MemberQ[#,i],
    weight[s,#]w[#],0]& /@ unions)];
  normalized=And@@Flatten[Table[
    p Binomial[s,t]/(s Binomial[p,t])==Binomial[s-1,t-1]/Binomial[p-1,t-1],{s,p},{t,p}]];
  agree=And@@Flatten[Table[excess[k,0]==0 && by[k,s]==p excess[k,s]/s,{k,sets},{s,p}]];
  updates=And@@Flatten[Table[
    excess[DeleteCases[k,j],s]-excess[k,s]==(s/p)inc[k,j,s] &&
    And@@Table[inc[DeleteCases[k,j],i,s]==inc[k,i,s]-both[k,j,i,s],{i,DeleteCases[k,j]}],
    {k,sets},{j,k},{s,p}]];
  mixture=And@@Flatten[Table[Module[{es=Table[excess[k,s],{s,0,p}],ks=Range[0,p],q=law/Total[law]},
    Implies[Max[Table[by[k,s],{s,p}]]<=tol z.z, q.es<=tol(q.ks)z.z/p]],
    {k,sets},{law,Join[IdentityMatrix[p+1],{Range[p+1],Reverse[Range[p+1]]}]}]];
  And[normalized,agree,updates,mixture]];
checks={fixture[1,{{1}},{{1}},{0},{0}],
  fixture[3,{{1},{1},{2},{1,2},{2,3}},{{2,-1},{-2,3},{1,2},{-1,3},{-2,-1}}/3,
    {1/3,1},{2,-1}],
  fixture[2,{{1},{2},{1,2}},{{2,-2,1,-1},{-2,2,2,-2},{-2,2,0,0}}/20,
    ConstantArray[1/2,4],{1,0,0,1}],
  fixture[3,{{1},{2,3},{3}},{{0},{0},{0}},{1},{1}],
  Mean[{0,1}^2]-Mean[{0,1}]^2==1/4};
If[!TrueQ[And@@checks],Abort[]];checks
