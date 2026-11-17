(* Projection theorem on any segment, using a=||v||^2 and b=<F-q*,v>,
   v=q-q*. At a minimizer q*, all segments toward q in the compact convex
   relaxation have nonnegative first change in squared distance. *)
normalFromClosest = Resolve[ForAll[{a,b}, a>=0 &&
    ForAll[t, 0<=t<=1, t^2 a-2 t b >= 0], b<=0], Reals];
closestFromNormal = Resolve[ForAll[{a,b}, a>=0 && b<=0, a-2 b>=0],Reals];
(* In any dimension CS gives u.(F-q)<=||u|| ||F-q||. The reverse is attained
   by u=F-q. At q*, the normal property above gives u.q<=u.q* throughout the
   hull, u=F-q*, with separation margin ||u||^2-R||u||. Equality allows
   contact. This is the projection test for the hull; a fractional feasible
   point need not be an admissible binary support. *)
ClearAll[nodeCheck];
nodeCheck[parts_,rho_] := Module[{p=Length[parts], f=Total[parts], n, dirs, states},
  n=Range[p]; dirs=DeleteCases[Tuples[{-1,0,1},Length[f]],ConstantArray[0,Length[f]]];
  states=Tuples[{-1,0,1},p]; (* -1 excluded, 0 free, 1 forced. *)
  And@@Flatten[Table[Module[{fixed=Pick[n,state,1],free=Pick[n,state,0],maxWeight},
    If[Length[fixed]>k,True,
      Module[{allowed=Join[fixed,#]& /@ Subsets[free,{0,Min[k-Length[fixed],Length[free]]}]},
        And@@Table[Module[{w=parts.u, threshold=u.f-rho Norm[f] Norm[u], feasible, extreme},
          feasible=Select[allowed,Norm[f-Total[Prepend[parts[[#]],0f]]]<=rho Norm[f]&];
          extreme=Total[w[[fixed]]]+Total[TakeLargest[Map[Max[0,#]&,w[[free]]],
            Min[k-Length[fixed],Length[free]]]];
          extreme==Max[(Total[w[[#]]]& /@ allowed)] &&
            Implies[extreme<threshold,feasible=={}]], {u,dirs}]]]],
      {state,states},{k,0,p}]]];
(* An exact integrality gap. The full teacher f is the midpoint of a1 and a2.
   Their half-half convex combination has width budget one. *)
cols={{1,0},{0,1},{-1/2,-1/2}}; target=Total[cols];
integrality = target==(cols[[1]]+cols[[2]])/2 &&
    AllTrue[Subsets[Range[3],{0,1}],Norm[target-Total[Prepend[cols[[#]],0 target]]]>Norm[target]/2&];
checks={normalFromClosest,closestFromNormal,nodeCheck[cols,1/2],
  nodeCheck[{{1,-1},{-1,1},{0,0}},0],nodeCheck[{{1,0},{0,1},{1,1}},1],integrality};
If[!TrueQ[And@@checks],Abort[]];checks
