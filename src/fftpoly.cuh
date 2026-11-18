/*
 * fftpoly.cuh -- polynomial arithmetic mod n over FftMod, for the
 * polynomial (FFT continuation) stage 2 of libgmpecmfft.
 *
 * A Poly is a vector of slots, coefficient i in element i, every curve
 * its own polynomial; all curves' polynomials have the same length and
 * go through the same operations. Coefficients are in Montgomery form, so
 * products of polynomials come out in Montgomery form too. Reversal and
 * truncation are free: they only reorder or drop slot pointers.
 *
 * Copyright (c) 2026 Stefan Teleman and contributors.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.  See <http://www.gnu.org/licenses/>.
 */

#ifndef GMPECMFFT_FFTPOLY_CUH
#define GMPECMFFT_FFTPOLY_CUH

#include <vector>

#include "fftmod.cuh"

namespace gmpecmfft_detail {

typedef std::vector<slot_t> Poly;

class PolyOps {
public:
  // ONE holds R mod n (Montgomery 1) and ZERO 0, on every curve.
  PolyOps(FftMod &m, slot_t one, slot_t zero) : M(m), ONE(one), ZERO(zero) {}

  // Slots from a free list (FftMod never frees one before it is
  // destroyed, and trees make and drop many).
  slot_t take();
  void give(slot_t s);
  Poly take(unsigned len);
  void give(const Poly &p);

  // Every operation also comes in a batched form, a vector of
  // independent requests done together: their polynomial products go to
  // FftMod::polymul() as one call, so a tree level costs a few batched
  // transforms instead of a few per node.

  // out[t] = (A*B)[lo + t], t < nd (fresh slots).
  struct MulReq {
    const Poly *A, *B;
    unsigned lo, nd;
  };
  std::vector<Poly> mul(const std::vector<MulReq> &r);
  Poly mul(const Poly &A, const Poly &B, unsigned lo, unsigned nd);
  Poly mul_full(const Poly &A, const Poly &B) {
    return mul(A, B, 0, (unsigned)(A.size() + B.size() - 1));
  }

  // I = 1/rev(F) mod X^s, F monic of degree >= s - 1 given low to high
  // (F.size() = deg F + 1; rev(F)'s constant term is F's leading 1).
  // Newton iteration: I <- I + X^t * (I * (1 - rev(F) I)/X^t), t doubling.
  std::vector<Poly> inverse(const std::vector<const Poly *> &F,
                            const std::vector<unsigned> &s);
  Poly inverse(const Poly &F, unsigned s);

  // A mod F, F monic of degree s, IF = inverse(F, >= A.size() - s), A's
  // coefficients below 2n. Returns s coefficients (fresh slots) below 3n
  // (product inputs; not reduced further); A.size() <= s copies A, padded
  // with zeros.
  struct RemReq {
    const Poly *A, *F, *IF;
  };
  std::vector<Poly> rem(const std::vector<RemReq> &r);
  Poly rem(const Poly &A, const Poly &F, const Poly &IF);

  // (A*B) mod F, F monic of degree s, IF = inverse(F, >= A.size() +
  // B.size() - 1 - s). Cheaper than rem(mul_full(A, B), F, IF): A*B is
  // kept exact (FftMod::kprod) and only its top coefficients are reduced
  // on their own, for the quotient Q; the low s are reduced together with
  // F*Q's, as one difference of exact products (FftMod::kout). About 3s
  // REDCs instead of 4s, and |result| < 1.5n.
  Poly mulrem(const Poly &A, const Poly &B, const Poly &F, const Poly &IF);

  // A product tree over leaves: level 0 the leaves, each next level the
  // products of adjacent pairs (an odd one out is carried up as is), one
  // batch per level.
  // tree.back()[0] is the product of all leaves.
  typedef std::vector<std::vector<Poly>> Tree;
  Tree product_tree(const std::vector<Poly> &leaves, bool keep);
  void give(const Tree &t, bool leaves_too);

  // vals[j] = A(r_j) for the leaves X - r_j (they must be linear), by a
  // scaled remainder tree (Bernstein): a node P carries the first deg P
  // coefficients of (A mod P)/P as a series in 1/X; a child L of P = L R
  // gets its own from the middle product Y_P * R -- deg L outputs, so
  // deg L REDCs, and no inverses below the root. At a leaf X - r the one
  // coefficient is A(r). The root's is rev(A) * 1/rev(F), so IF is
  // inverse(root, >= deg root). A.size() <= deg root. One batch per
  // level.
  std::vector<slot_t> evaluate(const Poly &A, const Tree &tree,
                               const Poly &IF);

  FftMod &M;
  slot_t ONE, ZERO;

private:
  std::vector<slot_t> free_;
};

} // namespace gmpecmfft_detail

#endif
