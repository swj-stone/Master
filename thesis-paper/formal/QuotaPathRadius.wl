(* Direct utility oracle checks both endpoints and every possible bijection on a radius.
   The minimum matching oracle enumerates all partial matchings on these tiny fixtures. *)
ClearAll[radiusCheck];
radiusCheck[parts_, y_, anchors_] := Module[
  {p=Length[parts], all, gram, d, value, delta, pairPrices, det},
  all=Range[p]; gram=parts.Transpose[parts]; d=2 parts.y-Diagonal[gram];
  value[s_]:=-Total[(y-Total[Prepend[parts[[s]], 0 y]])^2];
  delta[i_,j_,s_]:=value[Append[s,i]]-value[Append[s,j]];
  And @@ Flatten[Table[Module[{k=Length[a], out, m, prices, lo, neighbors, actual, gap, matching, checks},
    out=Complement[all,a]; m=Min[k,p-k];
    Table[
      prices=Table[Module[{za,zo,first,second},
        za=Table[gram[[i,h]]-gram[[j,h]], {h,DeleteCases[a,i]}];
        zo=Table[gram[[i,h]]-gram[[j,h]], {h,DeleteCases[out,j]}];
        first=delta[i,j,DeleteCases[a,i]];
        second=d[[i]]-d[[j]]-2(Total[TakeLargest[za,k-r]]+Total[TakeLargest[zo,r-1]]);
        first+second], {i,a},{j,out}];
      neighbors=Select[Subsets[all,{k}],Length[Complement[a,#]]==r&];
      actual=Min[(value[a]-value[#])& /@ neighbors];
      lo=Max[Total[TakeSmallest[Min /@ prices,r]],Total[TakeSmallest[Min /@ Transpose[prices],r]]];
      matching=Min[Flatten[Table[Table[Total[MapThread[prices[[#1,#2]]&, {rr,pp}]],
        {pp,Permutations[cc]}],{rr,Subsets[Range[k],{r}]},{cc,Subsets[Range[p-k],{r}]}]]];
      checks=And @@ Flatten[Table[Module[{removed=Complement[a,t], added=Complement[t,a]},
        Table[Total[MapThread[
          delta[#1,#2,DeleteCases[a,#1]]+delta[#1,#2,DeleteCases[t,#2]]&,
          {removed, perm}]] == 2(value[a]-value[t]), {perm,Permutations[added]}]], {t,neighbors}]];
      checks && lo<=matching<=2 actual, {r,1,m}]],{a,anchors}]]];
fixtures={
  {{{1,1},{-2,-1},{0,-2},{1,1},{1,-1}},{0,-2},Subsets[Range[5],{2}]},
  {{{-3,-3},{-1,3},{2,-1},{-3,3},{2,-2}},{-3,0},Subsets[Range[5],{2}]},
  {{{2,-1},{1,1},{-1,2},{0,-2},{3,1},{0,0}}/3,{2,-1}/3,{{1,3,6},{2,4,5}}},
  {ConstantArray[{0,0},6],{1,1},{{1,2,3}}}};
checks=radiusCheck@@#& /@ fixtures;
If[!TrueQ[And@@checks],Abort[]];checks
