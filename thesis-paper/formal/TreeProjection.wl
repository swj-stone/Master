(* Exact rational path-group contributions, including cancellation and inactive groups.
   Every subset and trial candidate is evaluated independently by the full grouped forest. *)
ClearAll[fixture];
fixture[groups_, parts_, p_] := Module[{n=Length[First[parts]], all, pred, full, effect,
    initial, sizes, anchor, sizeIdentities, directions, axis, cells, proj, tolSet,
    lower, densePick, screenedPick, trajectories, identities},
  all=Range[p];
  pred[s_]:=Total[Prepend[Pick[parts, (SubsetQ[s,#]& /@ groups),True], ConstantArray[0,n]]];
  full=pred[all];
  effect[s_,j_]:=pred[s]-pred[DeleteCases[s,j]];
  initial=effect[all,#]& /@ all;
  sizes=Union[{0,Floor[(p-1)/2],p-1}];
  anchor[j_,s_]:=Total[Prepend[MapThread[
    Function[{g,a},If[MemberQ[g,j],a Binomial[s,Length[g]-1]/Binomial[p-1,Length[g]-1],0 a]],
    {groups,parts}],ConstantArray[0,n]]];
  directions=Table[anchor[j,s],{j,all},{s,sizes}];
  sizeIdentities=And@@Flatten[Table[anchor[j,s]==Mean[
    (pred[Append[#,j]]-pred[#])& /@ Subsets[DeleteCases[all,j],{s}]],{j,all},{s,0,p-1}]];
  axis[a_,b_]:=If[a.a==0,0,(a.b)^2/(a.a)];
  cells[h_]:=GatherBy[Range[n], Function[i,{h[[i]]>0,h[[i]]^2>Mean[h^2]}]];
  proj[h_,d_]:=Total[Function[c, With[{a=h[[c]],b=d[[c]]},
    If[a.a==0,0,(a.b)^2/(a.a)]]] /@ cells[h]];
  lower[s_,j_,arm_]:=With[{r=full-pred[s],d=effect[s,j]},r.r+2r.d+
    If[arm==1,Max[(axis[#,d]& /@ directions[[j]])],proj[initial[[j]],d]]];
  identities=And@@Flatten[Table[
    With[{r=full-pred[s],d=effect[s,j],e=full-pred[DeleteCases[s,j]]},
      e.e==r.r+2r.d+d.d && lower[s,j,arm]<=e.e],{s,Subsets[all]},{j,s},{arm,2}]];
  tolSet=Union[{0,full.full/25,full.full},Table[With[{r=full-pred[s]},r.r],{s,Subsets[all]}]];
  densePick[s_,t_]:=Module[{sc=Select[
    Table[With[{e=full-pred[DeleteCases[s,j]]},{e.e,j}],{j,s}],First[#]<=t&]},
    If[sc=={},0,Last[First[Sort[sc]]]]];
  screenedPick[s_,t_,arm_]:=Module[{sc={t,p+1},won=False,bound,e,val},
    Do[bound=lower[s,j,arm];If[bound>First[sc],Break[]];
       e=full-pred[DeleteCases[s,j]];val=e.e;
       If[val<First[sc] || (val==First[sc]&&j<Last[sc]),sc={val,j};won=True],
       {j,SortBy[s,Function[q,{lower[s,q,arm],q}]]}];
    If[won,Last[sc],0]];
  trajectories=And@@Flatten[Table[densePick[s,t]==screenedPick[s,t,arm],
    {s,Subsets[all]},{t,tolSet},{arm,2}]];
  sizeIdentities&&identities&&trajectories];
groups={{1},{2},{1,2},{1,3},{2,3},{4},{1,2,4}};
parts={{2,1,0,0},{0,1,-1,1},{-2,3,1,-2},{1,0,0,0},{0,-1,1,2},{0,0,0,0},{2,-4,-1,0}}/3;
checks={fixture[groups,parts,4],fixture[{{1},{2},{1,2}},{{1,-1},{1,-1},{-1,1}},3],
  fixture[{{1}},{{0,0}},1],fixture[{{1,2}},{{1,-1}},2],
  (* Joint energy can veto an action even if each of the axes individually passes. *)
  With[{coef={1,1},budget=3/2},Max[coef^2]<=budget<coef.coef]};
If[!TrueQ[And@@checks],Abort[]];checks
