Clear[r];
answers = {
  Reduce[r >= 0 && 194 + 5 r < 174 + 16 r, r, Reals],
  Reduce[r >= 0 && 194 + 5 r <= 126 + 24 r && 194 + 5 r <= 211, r, Reals],
  Reduce[r >= 0 && 178 + 2 r <= 169 + 2 r, r, Reals]
};
If[answers =!= {r > 20/11, False, False}, Abort[]];
answers
