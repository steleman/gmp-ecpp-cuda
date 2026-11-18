/*
 * ecmfft.cu -- ECM on the GPU with cuFFT arithmetic (FftMod), and the
 * libgmpecmfft C API, which hands n of up to 16352 bits to libgmpecpp's
 * CGBN kernels instead.
 *
 * The curves, and the work done on them, are those of the CGBN path in
 * ecm_cuda.cu: Montgomery curves in XZ coordinates, Suyama's
 * parametrisation with the same sigmas from the same seed, stage 1 by the
 * prime powers p^e <= B1, stage 2 with D = 210 over the same pair masks.
 * Identical curves must give identical hits on the two paths, and that is
 * how this one is validated (gmpecmfft_ecm_fft() runs it at any size).
 *
 * The differences are in how the work is organised, because here every
 * multiplication is a batch of FFTs across all curves rather than one
 * curve's arithmetic in one thread group:
 *
 *  - Stage 1 is one Montgomery ladder over k = prod p^e (about 1.44*B1
 *    bits) instead of a short ladder per prime power; [k]P is the same
 *    point either way. A ladder step is three rounds of 4, 4 and 3
 *    multiplications, and each bit's step is captured once as a CUDA
 *    graph, so the host launches one graph per bit.
 *  - Stage 2 normalizes the 24 baby points on the host (x_j = X_j/Z_j,
 *    one GMP inversion per curve), so a prime costs X_m - x_j*Z_m as on
 *    the CGBN path, with one accumulator per baby point so a giant step is
 *    a single batch, not a serial chain (the 24 are multiplied together at
 *    the end). Transforms of the x_j are kept for the whole of stage 2.
 *
 * Copyright (c) 2026 Stefan Teleman and contributors. The stage 2 masks
 * and curve setup follow ecm_cuda.cu.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.  See <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "fftmod.cuh"
#include "fftpoly.cuh"
#include "gmpecmfft.h"
#include "gmpecpp.h"

namespace gmpecmfft_detail {
namespace {

double now_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

bool stats_on() {
  const char *e = getenv("GMPECMFFT_STATS");
  return e && *e && *e != '0';
}

// Above this the FFT is not to be trusted; a correct transform at the
// chosen digit size measures around 0.01 or less.
const double MAX_ROUNDING_ERROR = 0.25;

struct Pt {
  slot_t X, Z;
};

// A vector of initialized mpz values, for FftMod::get/put.
struct MpzVec {
  std::vector<__mpz_struct> v;
  std::vector<mpz_ptr> p;
  explicit MpzVec(unsigned n) : v(n), p(n) {
    for (unsigned i = 0; i < n; i++) {
      mpz_init(&v[i]);
      p[i] = &v[i];
    }
  }
  ~MpzVec() {
    for (auto &x : v)
      mpz_clear(&x);
  }
  MpzVec(const MpzVec &) = delete;
  MpzVec &operator=(const MpzVec &) = delete;
  mpz_ptr operator[](unsigned i) { return p[i]; }
};

class Curves {
public:
  explicit Curves(FftMod &m) : M(m), A{NULL, NULL} {}

  bool alloc() {
    slot_t *s[] = {&p1, &q1, &p2, &q2, &t1, &t2, &aa, &bb,
                   &s_, &d_, &c,  &s2, &d2, &tc, &tt, &A24,
                   &ONE, &ZERO};
    for (slot_t *x : s)
      if (!(*x = M.slot()))
        return false;
    A = S(A24);
    return true;
  }
  Pt pt() { return Pt{M.slot(), M.slot()}; }

  // (R0, R1) <- (R0 + R1, 2 R1) if bit, else (2 R0, R0 + R1); R1 - R0 = P,
  // given as operands (PX, PZ) so that a fixed P can be a stored
  // transform.
  void ladder_step(int bit, Pt R0, Pt R1, Opnd PX, Opnd PZ) {
    const Pt ad = bit ? R0 : R1, db = bit ? R1 : R0;
    {
      slot_t d[] = {p1, q1, p2, q2}, a[] = {R0.X, R1.X, R0.X, R1.X},
             b[] = {R0.Z, R1.Z, R0.Z, R1.Z};
      int ng[] = {1, 0, 0, 1};
      M.add(4, d, a, b, ng);
    }
    const slot_t ea = bit ? q1 : p2, eb = bit ? q2 : p1; // X+Z, X-Z of db
    {
      // Four distinct operands, so four transforms.
      slot_t d[] = {t1, t2, aa, bb};
      Opnd a[] = {S(p1), S(p2), S(ea), S(eb)},
           b[] = {S(q1), S(q2), S(ea), S(eb)};
      M.mul(4, d, a, b);
    }
    {
      slot_t d[] = {s_, d_, c}, a[] = {t1, t1, aa}, b[] = {t2, t2, bb};
      int ng[] = {0, 1, 1};
      M.add(3, d, a, b, ng);
    }
    {
      slot_t d[] = {s2, d2, db.X, tc};
      Opnd a[] = {S(s_), S(d_), S(aa), A}, b[] = {S(s_), S(d_), S(bb), S(c)};
      M.mul(4, d, a, b);
    }
    {
      slot_t d[] = {tt}, a[] = {tc}, b[] = {bb};
      int ng[] = {0};
      M.add(1, d, a, b, ng);
    }
    {
      slot_t d[] = {ad.X, ad.Z, db.Z};
      Opnd a[] = {PZ, PX, S(c)}, b[] = {S(s2), S(d2), S(tt)};
      M.mul(3, d, a, b);
    }
  }

  // out = Pa + Pb, given Pd = Pa - Pb. out must not alias Pd.
  void xadd(Pt Pa, Pt Pb, Pt Pd, Pt out) {
    {
      slot_t d[] = {p1, q1, p2, q2}, a[] = {Pa.X, Pb.X, Pa.X, Pb.X},
             b[] = {Pa.Z, Pb.Z, Pa.Z, Pb.Z};
      int ng[] = {1, 0, 0, 1};
      M.add(4, d, a, b, ng);
    }
    {
      slot_t d[] = {t1, t2}, a[] = {p1, p2}, b[] = {q1, q2};
      M.mul(2, d, a, b);
    }
    {
      slot_t d[] = {s_, d_}, a[] = {t1, t1}, b[] = {t2, t2};
      int ng[] = {0, 1};
      M.add(2, d, a, b, ng);
    }
    {
      slot_t d[] = {s2, d2}, a[] = {s_, d_}, b[] = {s_, d_};
      M.mul(2, d, a, b);
    }
    {
      slot_t d[] = {out.X, out.Z}, a[] = {Pd.Z, Pd.X}, b[] = {s2, d2};
      M.mul(2, d, a, b);
    }
  }

  // out[i] = Pa[i] + Pb[i], given Pd[i] = Pa[i] - Pb[i], for i < n: the
  // same formulas as xadd(), every round batched across the n additions
  // (independent chains advanced together). tmp holds the temporaries,
  // ten slots per addition; n is done in chunks of tmp.size() / 10. No
  // out[i] may alias any Pa, Pb or Pd.
  void xadd_n(unsigned n, const Pt *Pa, const Pt *Pb, const Pt *Pd,
              const Pt *out, const std::vector<slot_t> &tmp) {
    const unsigned cn = (unsigned)(tmp.size() / 10);
    std::vector<slot_t> d, a, b;
    std::vector<int> ng;
    for (unsigned i0 = 0; i0 < n; i0 += cn) {
      const unsigned m = n - i0 < cn ? n - i0 : cn;
      auto T = [&](unsigned i, unsigned w) { return tmp[10 * i + w]; };
      // p1 = Xa - Za, q1 = Xb + Zb, p2 = Xa + Za, q2 = Xb - Zb
      d.clear(), a.clear(), b.clear(), ng.clear();
      for (unsigned i = 0; i < m; i++) {
        const Pt &A = Pa[i0 + i], &B = Pb[i0 + i];
        slot_t dd[] = {T(i, 0), T(i, 1), T(i, 2), T(i, 3)},
               aa[] = {A.X, B.X, A.X, B.X}, bb[] = {A.Z, B.Z, A.Z, B.Z};
        int g[] = {1, 0, 0, 1};
        d.insert(d.end(), dd, dd + 4);
        a.insert(a.end(), aa, aa + 4);
        b.insert(b.end(), bb, bb + 4);
        ng.insert(ng.end(), g, g + 4);
      }
      M.add(4 * m, d.data(), a.data(), b.data(), ng.data());
      // t1 = p1 q1, t2 = p2 q2
      d.clear(), a.clear(), b.clear();
      for (unsigned i = 0; i < m; i++) {
        slot_t dd[] = {T(i, 4), T(i, 5)}, aa[] = {T(i, 0), T(i, 2)},
               bb[] = {T(i, 1), T(i, 3)};
        d.insert(d.end(), dd, dd + 2);
        a.insert(a.end(), aa, aa + 2);
        b.insert(b.end(), bb, bb + 2);
      }
      M.mul(2 * m, d.data(), a.data(), b.data());
      // s = t1 + t2, d = t1 - t2
      d.clear(), a.clear(), b.clear(), ng.clear();
      for (unsigned i = 0; i < m; i++) {
        slot_t dd[] = {T(i, 6), T(i, 7)}, aa[] = {T(i, 4), T(i, 4)},
               bb[] = {T(i, 5), T(i, 5)};
        int g[] = {0, 1};
        d.insert(d.end(), dd, dd + 2);
        a.insert(a.end(), aa, aa + 2);
        b.insert(b.end(), bb, bb + 2);
        ng.insert(ng.end(), g, g + 2);
      }
      M.add(2 * m, d.data(), a.data(), b.data(), ng.data());
      // s2 = s^2, d2 = d^2
      d.clear(), a.clear(), b.clear();
      for (unsigned i = 0; i < m; i++) {
        slot_t dd[] = {T(i, 8), T(i, 9)}, aa[] = {T(i, 6), T(i, 7)};
        d.insert(d.end(), dd, dd + 2);
        a.insert(a.end(), aa, aa + 2);
        b.insert(b.end(), aa, aa + 2);
      }
      M.mul(2 * m, d.data(), a.data(), b.data());
      // out = (Zd s2, Xd d2)
      d.clear(), a.clear(), b.clear();
      for (unsigned i = 0; i < m; i++) {
        const Pt &P = Pd[i0 + i], &O = out[i0 + i];
        slot_t dd[] = {O.X, O.Z}, aa[] = {P.Z, P.X},
               bb[] = {T(i, 8), T(i, 9)};
        d.insert(d.end(), dd, dd + 2);
        a.insert(a.end(), aa, aa + 2);
        b.insert(b.end(), bb, bb + 2);
      }
      M.mul(2 * m, d.data(), a.data(), b.data());
    }
  }

  // out = 2P (out may alias P).
  void dbl(Pt P, Pt out) {
    {
      slot_t d[] = {p2, p1}, a[] = {P.X, P.X}, b[] = {P.Z, P.Z};
      int ng[] = {0, 1};
      M.add(2, d, a, b, ng);
    }
    {
      slot_t d[] = {aa, bb}, a[] = {p2, p1}, b[] = {p2, p1};
      M.mul(2, d, a, b);
    }
    {
      slot_t d[] = {c}, a[] = {aa}, b[] = {bb};
      int ng[] = {1};
      M.add(1, d, a, b, ng);
    }
    {
      slot_t d[] = {out.X, tc};
      Opnd a[] = {S(aa), A}, b[] = {S(bb), S(c)};
      M.mul(2, d, a, b);
    }
    {
      slot_t d[] = {tt}, a[] = {tc}, b[] = {bb};
      int ng[] = {0};
      M.add(1, d, a, b, ng);
    }
    {
      slot_t d[] = {out.Z}, a[] = {c}, b[] = {tt};
      M.mul(1, d, a, b);
    }
  }

  // out = [k]B, using R0, R1 as workspace (none may alias B).
  void ladder(Pt B, mpz_srcptr k, Pt R0, Pt R1, Pt out) {
    M.copy(R0.X, ONE);
    M.copy(R0.Z, ZERO);
    M.copy(R1.X, B.X);
    M.copy(R1.Z, B.Z);
    for (long i = (long)mpz_sizeinbase(k, 2) - 1; i >= 0; i--)
      ladder_step(mpz_tstbit(k, i), R0, R1, S(B.X), S(B.Z));
    M.copy(out.X, R0.X);
    M.copy(out.Z, R0.Z);
  }

  FftMod &M;
  slot_t p1, q1, p2, q2, t1, t2, aa, bb, s_, d_, c, s2, d2, tc, tt;
  slot_t A24, ONE, ZERO;
  Opnd A; // a24 as an operand: the slot, or its stored transform
};

// ---------------------------------------------------------------- //
// Stage 2 pair masks (as in ecm_cuda.cu)                             //
// ---------------------------------------------------------------- //

#define S2_D 210u
#define S2_H 105u
#define S2_NB 24u

inline bool s2_coprime(uint32_t j) {
  return (j & 1u) && (j % 3u) && (j % 5u) && (j % 7u);
}

// masks[m - m_lo] has bit k set when m*D +- j_k is a prime in (b1, b2].
void s2_masks(unsigned long b1, unsigned long b2, uint32_t *m_lo_out,
              std::vector<uint32_t> &masks) {
  int8_t kof[S2_H + 1];
  memset(kof, -1, sizeof kof);
  for (uint32_t j = 1, k = 0; j < S2_H; j += 2)
    if (s2_coprime(j))
      kof[j] = (int8_t)k++;

  const uint64_t lo = (uint64_t)b1 + 1, hi = b2;
  const uint32_t m_lo = (uint32_t)((lo + S2_H) / S2_D);
  const uint32_t m_hi = (uint32_t)((hi + S2_H) / S2_D);
  masks.assign(m_hi - m_lo + 1, 0);

  uint64_t r = 1;
  while ((r + 1) * (r + 1) <= hi)
    r++;
  std::vector<uint32_t> base;
  {
    std::vector<uint8_t> cmp(r + 1, 0);
    for (uint64_t i = 2; i <= r; i++)
      if (!cmp[i]) {
        base.push_back((uint32_t)i);
        for (uint64_t j = i * i; j <= r; j += i)
          cmp[j] = 1;
      }
  }
  const uint64_t SEG = 1u << 20;
  std::vector<uint8_t> comp(SEG);
  for (uint64_t s0 = lo; s0 <= hi; s0 += SEG) {
    const uint64_t s1 = s0 + SEG - 1 < hi ? s0 + SEG - 1 : hi;
    std::fill(comp.begin(), comp.end(), 0);
    for (uint32_t p : base) {
      uint64_t st = ((s0 + p - 1) / p) * p;
      if (st < (uint64_t)p * p)
        st = (uint64_t)p * p;
      for (uint64_t x = st; x <= s1; x += p)
        comp[x - s0] = 1;
    }
    for (uint64_t q = s0; q <= s1; q++) {
      if (comp[q - s0] || q < 11)
        continue;
      const uint32_t m = (uint32_t)((q + S2_H) / S2_D);
      const int64_t j = (int64_t)q - (int64_t)m * S2_D;
      const uint32_t a = (uint32_t)(j < 0 ? -j : j);
      masks[m - m_lo] |= 1u << kof[a];
    }
  }
  *m_lo_out = m_lo;
}

// ---------------------------------------------------------------- //
// The run                                                            //
// ---------------------------------------------------------------- //

struct Run {
  // The arithmetic; stage 2 may replace stage 1's with its own.
  std::unique_ptr<FftMod> Mp{new FftMod};
  FftMod &mod() { return *Mp; }
  mpz_t n;
  unsigned C = 0;
  std::vector<int> dead;  // curve setup already exposed gcd = initg
  MpzVec *initg = NULL;
  std::vector<int> s2dead; // stage 2 setup (a baby point's inversion)
  MpzVec *s2g = NULL;      // exposed gcd = s2g
  double maxe = 0.0;
  bool stats = false;
  bool quiet = false;  // scan() counts but does not print (curve groups)
  bool detail = true;  // per-phase statistics (only a first group's)
  unsigned hits = 0;  // the last scan()'s count
  int (*stop)(void *) = NULL;
  void *stop_arg = NULL;

  Run() { mpz_init(n); }
  ~Run() {
    delete initg;
    delete s2g;
    mpz_clear(n);
  }

  // After a slice of work: synchronize and check the arithmetic.
  gmpecmfft_status_t check() {
    if (!Mp->sync())
      return GMPECMFFT_ERR_CUDA;
    const double e = Mp->max_err();
    if (e > maxe)
      maxe = e;
    if (e > MAX_ROUNDING_ERROR)
      return GMPECMFFT_ERR_PRECISION;
    if (Mp->overflowed())
      return GMPECMFFT_ERR_INTERNAL;
    return GMPECMFFT_FACTOR_FOUND; // "no problem"
  }

  // gcd of every curve's residue with n; first proper divisor -> factor.
  bool scan(slot_t s, mpz_ptr factor, const char *stage, double secs) {
    MpzVec v(C);
    Mp->get(v.p, s);
    unsigned hits = 0;
    bool found = false;
    mpz_t g;
    mpz_init(g);
    for (unsigned i = 0; i < C && (stats || !found); i++) {
      if (dead[i])
        mpz_set(g, (*initg)[i]);
      else if (s2g && s2dead[i])
        mpz_set(g, (*s2g)[i]);
      else
        mpz_gcd(g, v[i], n);
      if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, n) < 0) {
        if (!found)
          mpz_set(factor, g);
        found = true;
        hits++;
      }
    }
    mpz_clear(g);
    this->hits = hits;
    if (stats && !quiet)
      fprintf(stderr,
              "gmpecmfft stats: %s: %u of %u curves found a factor "
              "(%.2f s)\n",
              stage, hits, C, secs);
    return found;
  }

  bool stopped() { return stop && stop(stop_arg); }
};

// Memory to reserve, in residue slots (a stored transform is ~4 slots):
// stage 1 uses 18 working slots, 3 points and 3 transforms; stage 2 adds
// 24 baby points, 24 accumulators, 24 products, 10 points, 2 slots and
// 26 transforms.
unsigned slots_needed(bool stage2) {
  return 18 + 6 + 3 * 4 + (stage2 ? S2_NB * 4 + 20 + 2 + 26 * 4 : 0);
}

// Slice sizing: aim for ~0.25 s between stop() polls.
unsigned next_slice(unsigned cnt, double dt) {
  double want = dt > 0 ? cnt * (0.25 / dt) : cnt * 4.0;
  if (want > cnt * 4.0)
    want = cnt * 4.0;
  return want < 1.0 ? 1u : (unsigned)want;
}

// x_j = X_j / Z_j, in Montgomery form, for the nb baby points of every
// curve. Batch inversion by a product tree on the GPU: the products of
// the Z_j up the tree (one batched round per level), one inversion per
// curve on the host at the root, and back down -- a child's inverse is
// its parent's times its sibling's product -- then x_j = X_j (1/Z_j).
// About 4 nb multiplications in 2 log2(nb) rounds; one slot to the host
// and one back. In Montgomery form the products stay products (P~ = P R),
// and 1/P is (1/P)~ = R^2 / P~ mod n.
//
// As on the CGBN path, a Z_j with no inverse exposes a factor: the root's
// gcd with n, if proper, becomes the curve's stage 2 result; if it is n,
// the Z_j are fetched and the first proper gcd(Z_j, n) is taken (rare).
// Such a curve gets x_j = 0. take/give supply and return temporaries.
void normalize_babies(Run &r, unsigned nb, const Pt *B, const slot_t *out,
                      const std::function<slot_t()> &take,
                      const std::function<void(slot_t)> &give) {
  FftMod &M = r.mod();
  const unsigned C = r.C;
  // Up: lv[0] the Z_j (not ours), each next level pairwise products, an
  // odd one carried up (the same slot).
  std::vector<std::vector<slot_t>> lv(1);
  for (unsigned j = 0; j < nb; j++)
    lv[0].push_back(B[j].Z);
  while (lv.back().size() > 1) {
    const std::vector<slot_t> &cur = lv.back();
    std::vector<slot_t> nx, a, b;
    for (size_t i = 0; i + 1 < cur.size(); i += 2) {
      nx.push_back(take());
      a.push_back(cur[i]);
      b.push_back(cur[i + 1]);
    }
    M.mul((unsigned)nx.size(), nx.data(), a.data(), b.data());
    if (cur.size() & 1)
      nx.push_back(cur.back());
    lv.push_back(nx);
  }

  // The root's inverse, per curve, on host threads.
  r.s2dead.assign(C, 0);
  r.s2g = new MpzVec(C);
  MpzVec P(C), I(C);
  M.get(P.p, lv.back()[0]);
  mpz_t R2;
  mpz_init(R2);
  mpz_mul(R2, M.R(), M.R());
  mpz_mod(R2, R2, r.n);
  std::vector<int> whole(C, 0); // gcd(P, n) = n: look at each Z_j
  auto work = [&](unsigned c0, unsigned step) {
    for (unsigned c = c0; c < C; c += step) {
      if (r.dead[c])
        continue; // I = 0
      if (mpz_invert(I[c], P[c], r.n)) {
        mpz_mul(I[c], I[c], R2);
        mpz_mod(I[c], I[c], r.n);
        continue;
      }
      mpz_set_ui(I[c], 0);
      mpz_gcd(P[c], P[c], r.n);
      if (mpz_cmp(P[c], r.n) != 0) {
        r.s2dead[c] = 1;
        mpz_set((*r.s2g)[c], P[c]);
      } else {
        whole[c] = 1;
      }
    }
  };
  unsigned nt = std::thread::hardware_concurrency();
  nt = nt < 1 ? 1 : (nt > C ? C : nt);
  {
    std::vector<std::thread> th;
    for (unsigned i = 1; i < nt; i++)
      th.emplace_back(work, i, nt);
    work(0, nt);
    for (auto &t : th)
      t.join();
  }
  mpz_clear(R2);
  if (std::find(whole.begin(), whole.end(), 1) != whole.end()) {
    MpzVec z(C);
    for (unsigned j = 0; j < nb; j++) {
      M.get(z.p, B[j].Z);
      for (unsigned c = 0; c < C; c++) {
        if (!whole[c] || r.s2dead[c])
          continue;
        mpz_gcd(z[c], z[c], r.n);
        if (mpz_cmp_ui(z[c], 1) > 0 && mpz_cmp(z[c], r.n) < 0) {
          r.s2dead[c] = 1;
          mpz_set((*r.s2g)[c], z[c]);
        }
      }
    }
  }

  // Down: inv[l][i] the inverse of lv[l][i].
  std::vector<slot_t> inv(1, take());
  M.put(inv[0], I.p);
  std::vector<slot_t> mine(inv); // to give back
  for (size_t l = lv.size() - 1; l > 0; l--) {
    const std::vector<slot_t> &kids = lv[l - 1];
    std::vector<slot_t> nx(kids.size()), d, a, b;
    for (size_t i = 0; i < inv.size(); i++) {
      const size_t x = 2 * i, y = 2 * i + 1;
      if (y >= kids.size()) { // carried: the same value
        nx[x] = inv[i];
        continue;
      }
      nx[x] = take();
      nx[y] = take();
      mine.push_back(nx[x]);
      mine.push_back(nx[y]);
      d.insert(d.end(), {nx[x], nx[y]});
      a.insert(a.end(), {inv[i], inv[i]});
      b.insert(b.end(), {kids[y], kids[x]});
    }
    M.mul((unsigned)d.size(), d.data(), a.data(), b.data());
    inv.swap(nx);
  }
  {
    std::vector<slot_t> d(out, out + nb), a(nb);
    for (unsigned j = 0; j < nb; j++)
      a[j] = B[j].X;
    M.mul(nb, d.data(), a.data(), inv.data());
  }
  for (slot_t x : mine)
    give(x);
  for (size_t l = 1; l < lv.size(); l++)
    for (size_t i = 0; i + 1 < lv[l - 1].size(); i += 2)
      give(lv[l][i / 2]);
}

gmpecmfft_status_t stage2(Run &r, Curves &W, Pt Q, unsigned long b1,
                          unsigned long b2, mpz_ptr factor, bool *found) {
  FftMod &M = r.mod();
  if (b2 > 0xffffffffUL)
    b2 = 0xffffffffUL;
  uint32_t m_lo = 0;
  std::vector<uint32_t> masks;
  s2_masks(b1, b2, &m_lo, masks);
  const uint32_t nm = (uint32_t)masks.size();

  const double t0 = now_seconds();
  Pt B[S2_NB], T[3], G[3], DQ = W.pt(), Q2 = W.pt(), R0 = W.pt(),
               R1 = W.pt();
  slot_t acc[S2_NB], w[S2_NB], qd[2] = {M.slot(), M.slot()};
  xslot_t xb[S2_NB], xqd[2] = {M.xslot(), M.xslot()};
  for (unsigned k = 0; k < S2_NB; k++) {
    B[k] = W.pt();
    acc[k] = M.slot();
    w[k] = M.slot();
    xb[k] = M.xslot();
  }
  for (int i = 0; i < 3; i++) {
    T[i] = W.pt();
    G[i] = W.pt();
  }
  if (!M.ok())
    return GMPECMFFT_ERR_NO_MEMORY;

  // Baby points [j]Q, j < 105 odd and coprime to 210.
  W.dbl(Q, Q2);
  unsigned k = 0, ti = 0;
  M.copy(B[k].X, Q.X);
  M.copy(B[k].Z, Q.Z);
  k++;
  Pt prev = Q, cur = T[ti++ % 3];
  W.xadd(Q2, Q, Q, cur); // [3]Q
  for (uint32_t j = 3; j < S2_H; j += 2) {
    if (s2_coprime(j)) {
      M.copy(B[k].X, cur.X);
      M.copy(B[k].Z, cur.Z);
      k++;
    }
    Pt nxt = T[ti++ % 3];
    W.xadd(cur, Q2, prev, nxt); // [j+2] = [j] + [2], difference [j-2]
    prev = cur;
    cur = nxt;
  }

  // [D]Q and the giant pair [m_lo D]Q, [(m_lo - 1) D]Q.
  mpz_t s;
  mpz_init_set_ui(s, S2_D);
  W.ladder(Q, s, R0, R1, DQ);
  mpz_set_ui(s, (unsigned long)m_lo * S2_D);
  W.ladder(Q, s, R0, R1, G[0]);
  mpz_set_ui(s, (unsigned long)(m_lo - 1) * S2_D);
  W.ladder(Q, s, R0, R1, G[1]);
  mpz_clear(s);
  {
    slot_t d[] = {qd[0], qd[1]}, a[] = {DQ.X, DQ.X}, b[] = {DQ.Z, DQ.Z};
    int ng[] = {0, 1};
    M.add(2, d, a, b, ng);
  }
  for (unsigned i = 0; i < S2_NB; i++)
    M.copy(acc[i], W.ONE);
  gmpecmfft_status_t st = r.check();
  if (st != GMPECMFFT_FACTOR_FOUND)
    return st;

  // Normalize the baby points, and keep the transforms of everything
  // that is used at every giant step: the x_j and DQ's X+Z, X-Z.
  normalize_babies(
      r, S2_NB, B, w, [&]() { return M.slot(); }, [](slot_t) {});
  M.xform(S2_NB, xb, w);
  M.xform(2, xqd, qd);
  st = r.check();
  if (st != GMPECMFFT_FACTOR_FOUND)
    return st;

  int gm = 0, gp = 1, gn = 2;
  uint32_t i = 0, slice = 16;
  std::vector<slot_t> d;
  std::vector<Opnd> a, b;
  std::vector<slot_t> sa, sb;
  std::vector<int> ng;
  while (i < nm) {
    if (r.stopped())
      return GMPECMFFT_STOPPED;
    const uint32_t cnt = slice < nm - i ? slice : nm - i;
    const double ts = now_seconds();
    for (uint32_t e = i + cnt; i < e; i++) {
      const Pt Gm = G[gm], Gp = G[gp], Gn = G[gn];
      const uint32_t mask = masks[i];
      // [m D]Q + [D]Q, difference [(m-1) D]Q: the X-Z, X+Z terms.
      {
        slot_t dd[] = {W.p1, W.p2}, aa[] = {Gm.X, Gm.X}, bb[] = {Gm.Z, Gm.Z};
        int nn[] = {1, 0};
        M.add(2, dd, aa, bb, nn);
      }
      // x_j * Z_m for every pair, and the point addition's products.
      // Z_m is transformed once for all of them; x_j and DQ's terms not
      // at all.
      d.clear(), a.clear(), b.clear();
      for (uint32_t mk = mask; mk; mk &= mk - 1) {
        const unsigned j = __builtin_ctz(mk);
        d.push_back(w[j]), a.push_back(S(Gm.Z)), b.push_back(X(xb[j]));
      }
      d.push_back(W.t1), a.push_back(S(W.p1)), b.push_back(X(xqd[0]));
      d.push_back(W.t2), a.push_back(S(W.p2)), b.push_back(X(xqd[1]));
      M.mul((unsigned)d.size(), d.data(), a.data(), b.data());

      // X_m - x_j Z_m
      d.clear(), sa.clear(), sb.clear(), ng.clear();
      for (uint32_t mk = mask; mk; mk &= mk - 1) {
        const unsigned j = __builtin_ctz(mk);
        d.push_back(w[j]), sa.push_back(Gm.X), sb.push_back(w[j]),
            ng.push_back(1);
      }
      d.push_back(W.s_), sa.push_back(W.t1), sb.push_back(W.t2),
          ng.push_back(0);
      d.push_back(W.d_), sa.push_back(W.t1), sb.push_back(W.t2),
          ng.push_back(1);
      M.add((unsigned)d.size(), d.data(), sa.data(), sb.data(), ng.data());

      d.clear(), sa.clear(), sb.clear();
      for (uint32_t mk = mask; mk; mk &= mk - 1) {
        const unsigned j = __builtin_ctz(mk);
        d.push_back(acc[j]), sa.push_back(acc[j]), sb.push_back(w[j]);
      }
      d.push_back(W.s2), sa.push_back(W.s_), sb.push_back(W.s_);
      d.push_back(W.d2), sa.push_back(W.d_), sb.push_back(W.d_);
      M.mul((unsigned)d.size(), d.data(), sa.data(), sb.data());
      {
        slot_t dd[] = {Gn.X, Gn.Z}, aa[] = {Gp.Z, Gp.X}, bb[] = {W.s2, W.d2};
        M.mul(2, dd, aa, bb);
      }
      const int old = gp;
      gp = gm, gm = gn, gn = old;
    }
    st = r.check();
    if (st != GMPECMFFT_FACTOR_FOUND)
      return st;
    slice = next_slice(cnt, now_seconds() - ts);
  }

  // acc = prod acc[j]
  for (unsigned cnt = S2_NB; cnt > 1;) {
    const unsigned half = cnt / 2;
    slot_t dd[S2_NB], aa[S2_NB], bb[S2_NB];
    for (unsigned j = 0; j < half; j++)
      dd[j] = acc[j], aa[j] = acc[2 * j], bb[j] = acc[2 * j + 1];
    M.mul(half, dd, aa, bb);
    if (cnt & 1)
      M.copy(acc[half], acc[cnt - 1]);
    cnt = half + (cnt & 1);
  }
  st = r.check();
  if (st != GMPECMFFT_FACTOR_FOUND)
    return st;
  *found = r.scan(acc[0], factor, "stage 2", now_seconds() - t0);
  return GMPECMFFT_FACTOR_FOUND;
}

// ---------------------------------------------------------------- //
// Polynomial stage 2 (FFT continuation)                              //
// ---------------------------------------------------------------- //
//
// The same product as the per-prime stage 2 -- prod (X_m Z_j - X_j Z_m)
// over giant points [mD]Q and baby points [j]Q -- but over every pair:
// all j < D/2 coprime to D, all m covering (B1, B2]. That covers every
// number in range coprime to D, primes included, at a cost that grows
// with sqrt(B2) instead of with the number of primes (Montgomery's FFT
// continuation, as in GMP-ECM):
//
//   F(X) = prod_j (X - x_j)            k = phi(D)/2 babies, normalized
//   G_b(X) = prod_{m in block b} (Z_m X - X_m)     k giants per block
//   H = prod_b G_b  mod F               H <- H G_b mod F, block by block
//   prod_j H(x_j) = prod_b prod_j G_b(x_j) = the product over all pairs
//
// F and every node of its product tree are kept for the last step, the
// evaluation of H at the x_j by a scaled remainder tree. Polynomial
// products sum up to k + 1 terms per coefficient, so stage 2 gets its
// own FftMod (smaller digits, larger R); Q and a24 move to it through
// the host.

// Stage 2's D, largest first; k = phi(D)/2.
const struct {
  unsigned D, k;
} S2_POLY[] = {{30030, 2880}, {2310, 240}, {210, 24}};

// Slots the polynomial stage 2 needs, roughly: F's product tree,
// k (log2 k + 2); the babies, x_j, F^-1, H; a block's leaves, its
// product tree (two levels at a time), H*G and the remainder's
// temporaries; and up to 10 * MAXK for batched additions (xadd_n).
unsigned poly_slots(unsigned k) {
  unsigned lg = 0;
  while ((1u << lg) < k)
    lg++;
  return k * (lg + 16) + 64 + 10 * MAXK;
}

// GMPECMFFT_S2_CHECK=1: replay the polynomial stage 2 of the first
// curves on the host with GMP -- the same ladders, chains and formulas, so
// the same projective coordinates -- and compare the product over all
// pairs with the GPU's, exactly. A debugging aid; costs ~k*giants
// multiplications per curve on one CPU thread.
struct HostPt {
  mpz_t X, Z;
  HostPt() { mpz_inits(X, Z, NULL); }
  ~HostPt() { mpz_clears(X, Z, NULL); }
  HostPt(const HostPt &o) {
    mpz_init_set(X, o.X);
    mpz_init_set(Z, o.Z);
  }
  HostPt &operator=(const HostPt &o) {
    mpz_set(X, o.X);
    mpz_set(Z, o.Z);
    return *this;
  }
};

void h_xadd(HostPt &o, const HostPt &a, const HostPt &b, const HostPt &d,
            mpz_srcptr n) {
  mpz_t t1, t2, u;
  mpz_inits(t1, t2, u, NULL);
  mpz_sub(t1, a.X, a.Z);
  mpz_add(u, b.X, b.Z);
  mpz_mul(t1, t1, u);
  mpz_add(t2, a.X, a.Z);
  mpz_sub(u, b.X, b.Z);
  mpz_mul(t2, t2, u);
  mpz_add(u, t1, t2);
  mpz_sub(t2, t1, t2);
  mpz_mul(u, u, u);
  mpz_mul(t2, t2, t2);
  mpz_mul(u, u, d.Z);
  mpz_mul(t2, t2, d.X);
  mpz_mod(o.X, u, n);
  mpz_mod(o.Z, t2, n);
  mpz_clears(t1, t2, u, NULL);
}

void h_dbl(HostPt &o, const HostPt &p, mpz_srcptr a24, mpz_srcptr n) {
  mpz_t aa, bb, c;
  mpz_inits(aa, bb, c, NULL);
  mpz_add(aa, p.X, p.Z);
  mpz_mul(aa, aa, aa);
  mpz_sub(bb, p.X, p.Z);
  mpz_mul(bb, bb, bb);
  mpz_sub(c, aa, bb);
  mpz_mul(o.X, aa, bb);
  mpz_mod(o.X, o.X, n);
  mpz_mul(aa, a24, c);
  mpz_add(aa, aa, bb);
  mpz_mul(o.Z, c, aa);
  mpz_mod(o.Z, o.Z, n);
  mpz_clears(aa, bb, c, NULL);
}

void h_ladder(HostPt &o, const HostPt &q, mpz_srcptr s, mpz_srcptr a24,
              mpz_srcptr n) {
  HostPt R0, R1, t, u;
  mpz_set_ui(R0.X, 1);
  mpz_set_ui(R0.Z, 0);
  R1 = q;
  for (long i = (long)mpz_sizeinbase(s, 2) - 1; i >= 0; i--) {
    if (mpz_tstbit(s, i)) {
      h_xadd(t, R0, R1, q, n);
      h_dbl(u, R1, a24, n);
      R0 = t;
      R1 = u;
    } else {
      h_xadd(t, R0, R1, q, n);
      h_dbl(u, R0, a24, n);
      R1 = t;
      R0 = u;
    }
  }
  o = R0;
}

// The product over all pairs for one curve, from Q and a24 (true values).
// The giants are computed as stage2_poly() computes them (G chains; see
// there), so that their projective coordinates are the GPU's; the babies
// are normalized, so how they are reached does not matter.
void h_stage2_product(mpz_ptr out, mpz_srcptr qx, mpz_srcptr qz,
                      mpz_srcptr a24, unsigned D, unsigned long long m_lo,
                      unsigned long long m_hi, unsigned G, mpz_srcptr n) {
  HostPt Q, Q2, prev, cur, nxt;
  mpz_set(Q.X, qx);
  mpz_set(Q.Z, qz);
  std::vector<HostPt> B;
  h_dbl(Q2, Q, a24, n);
  B.push_back(Q);
  prev = Q;
  h_xadd(cur, Q2, Q, Q, n);
  for (unsigned j = 3; j < D / 2; j += 2) {
    if (std::gcd(j, D) == 1)
      B.push_back(cur);
    h_xadd(nxt, cur, Q2, prev, n);
    prev = cur;
    cur = nxt;
  }
  std::vector<__mpz_struct> x(B.size());
  for (size_t j = 0; j < B.size(); j++) {
    mpz_init(&x[j]);
    if (mpz_invert(&x[j], B[j].Z, n)) {
      mpz_mul(&x[j], &x[j], B[j].X);
      mpz_mod(&x[j], &x[j], n);
    } else {
      mpz_set_ui(&x[j], 0);
    }
  }
  HostPt DQ;
  mpz_t s, t;
  mpz_inits(s, t, NULL);
  mpz_set_ui(s, D);
  h_ladder(DQ, Q, s, a24, n);
  mpz_set_ui(out, 1);
  auto use = [&](const HostPt &g, unsigned long long m) {
    if (m < m_lo || m > m_hi)
      return;
    for (size_t j = 0; j < x.size(); j++) {
      mpz_mul(t, g.Z, &x[j]);
      mpz_sub(t, t, g.X);
      mpz_mul(out, out, t);
      mpz_mod(out, out, n);
    }
  };
  const unsigned T = 2 * G;
  std::vector<HostPt> w(T + 1); // w[i] = [iD]Q
  w[1] = DQ;
  for (unsigned sd = 1; sd < T; sd *= 2) {
    for (unsigned j = 1; j < sd && sd + j <= T; j++)
      h_xadd(w[sd + j], w[sd], w[j], w[sd - j], n);
    if (2 * sd <= T)
      h_dbl(w[2 * sd], w[sd], a24, n);
  }
  for (unsigned i = 1; i <= T; i++)
    use(w[i], i);
  std::vector<HostPt> win(w.begin() + 1, w.end()), nw(G);
  const HostPt GD = w[G];
  for (unsigned long long lo = 1; lo + T - 1 < m_hi; lo += G) {
    for (unsigned i = 0; i < G; i++)
      h_xadd(nw[i], win[G + i], GD, win[i], n);
    for (unsigned i = 0; i < G; i++) {
      win[i] = win[G + i];
      win[G + i] = nw[i];
      use(nw[i], lo + T + i);
    }
  }
  for (auto &v : x)
    mpz_clear(&v);
  mpz_clears(s, t, NULL);
}

gmpecmfft_status_t stage2_poly(Run &r, unsigned D, unsigned k, Curves &W,
                               Pt Q, unsigned long b1, unsigned long b2,
                               mpz_ptr factor, bool *found,
                               MpzVec *hq[3]) {
  FftMod &M = r.mod();
  const double t0 = now_seconds();
  PolyOps ops(M, W.ONE, W.ZERO);
  auto take_pt = [&]() { return Pt{ops.take(), ops.take()}; };
  // Giant chains advanced together (a round of G additions is 2G
  // products in its widest step).
  const unsigned G = M.maxk();
  auto give_pt = [&](Pt p) {
    ops.give(p.X);
    ops.give(p.Z);
  };

  // Temporaries for batched differential additions (Curves::xadd_n),
  // which go in chunks of tmp.size() / 10: enough that a round's widest
  // step (2 products per addition) transforms ~1024 residues. Batching
  // pays where single additions are too small to fill the GPU (few
  // curves); with many curves, wider rounds were measured slower.
  unsigned xchunk = 512 / M.C();
  xchunk = xchunk < 1 ? 1 : (xchunk > M.maxk() ? M.maxk() : xchunk);
  std::vector<slot_t> tmp(10 * (size_t)xchunk);
  for (auto &t : tmp)
    t = ops.take();

  // Babies [j]Q, j < D/2 odd and coprime to D, normalized: x_j. Every odd
  // multiple up to D/2 by doubling ranges: from [1], [3], ..., [2s-1] and
  // [2s], all of [2s + j] = [2s] + [j] (difference [2s - j]) in one batch,
  // then [4s] = 2 [2s] -- about log2(D/4) rounds, not D/4 serial steps.
  std::vector<Pt> B;
  {
    const unsigned J = (D / 2) % 2 ? D / 2 - 2 : D / 2 - 1; // odd, < D/2
    std::vector<Pt> odd((J + 1) / 2);           // odd[i] = [2i + 1]Q
    odd[0] = Q;
    Pt E = take_pt(); // [2s]Q
    W.dbl(Q, E);
    for (unsigned sd = 1; 2 * sd + 1 <= J; sd *= 2) {
      std::vector<Pt> pa, pb, pd, po;
      for (unsigned j = 1; j < 2 * sd && 2 * sd + j <= J; j += 2) {
        pa.push_back(E);
        pb.push_back(odd[j / 2]);
        pd.push_back(odd[(2 * sd - j) / 2]);
        po.push_back(odd[(2 * sd + j) / 2] = take_pt());
      }
      W.xadd_n((unsigned)po.size(), pa.data(), pb.data(), pd.data(),
               po.data(), tmp);
      if (4 * sd + 1 <= J)
        W.dbl(E, E);
    }
    give_pt(E);
    for (unsigned j = 1; j <= J; j += 2)
      if (std::gcd(j, D) == 1)
        B.push_back(odd[j / 2]);
      else if (j > 1)
        give_pt(odd[j / 2]);
  }
  if (B.size() != k)
    return GMPECMFFT_ERR_INTERNAL;
  gmpecmfft_status_t st = r.check();
  if (st != GMPECMFFT_FACTOR_FOUND)
    return st;
  const double t_b = now_seconds();
  Poly xs = ops.take(k);
  normalize_babies(
      r, k, B.data(), xs.data(), [&]() { return ops.take(); },
      [&](slot_t x) { ops.give(x); });
  const double t_n = now_seconds();
  for (Pt p : B)
    if (p.X != Q.X)
      give_pt(p);

  // F = prod (X - x_j): leaves (-x_j, 1), the 1 shared.
  std::vector<Poly> leaves(k);
  {
    Poly c0 = ops.take(k);
    std::vector<slot_t> z(k, W.ZERO);
    std::vector<int> ng(k, 1);
    M.add(k, c0.data(), z.data(), xs.data(), ng.data());
    for (unsigned j = 0; j < k; j++)
      leaves[j] = Poly{c0[j], W.ONE};
  }
  ops.give(xs);
  PolyOps::Tree Ft = ops.product_tree(leaves, true);
  const Poly &F = Ft.back()[0];
  r.check();
  const double t_t = now_seconds();
  Poly IF = ops.inverse(F, k);
  st = r.check();
  if (st != GMPECMFFT_FACTOR_FOUND)
    return st;
  const double t_f = now_seconds();
  if (r.stats && r.detail)
    fprintf(stderr,
            "gmpecmfft stats: stage 2 setup: babies %.2f s, normalize %.2f "
            "s, F tree %.2f s, 1/F %.2f s\n",
            t_b - t0, t_n - t_b, t_t - t_n, t_f - t_t);

  // Giants [mD]Q for m in [m_lo, m_hi]: every q in (B1, B2] is m*D +- j
  // for one of them, j <= D/2. D <= B1/2 keeps m_lo - 1 >= 1, so the
  // differential additions never meet the point at infinity. A block
  // costs the same however few giants it has, so the last one is filled:
  // the stage covers somewhat past B2, for nothing.
  const unsigned long long m_lo = ((unsigned long long)b1 + 1 + D / 2) / D,
                           m_need = ((unsigned long long)b2 + D / 2) / D,
                           nblk = (m_need - m_lo + k) / k,
                           m_hi = m_lo + nblk * k - 1;
  // The giants come from G chains advanced together: [mD]Q for m = 1
  // .. 2G by doubling ranges (as the babies), then rounds of
  // [(m + G)D] = [mD] + [GD], difference [(m - G)D], for the G newest m
  // at once -- k/G rounds per block instead of k serial steps. The chains
  // start at m = 1, so about m_lo additions are spent below the range;
  // in exchange no difference is ever O and no other start points are
  // needed. The host replay (h_stage2_product) does the same.
  const unsigned T = 2 * G;
  Pt DQ = take_pt();
  {
    Pt R0 = take_pt(), R1 = take_pt();
    mpz_t s;
    mpz_init_set_ui(s, D);
    W.ladder(Q, s, R0, R1, DQ);
    mpz_clear(s);
    give_pt(R0);
    give_pt(R1);
  }
  std::vector<Pt> win(T); // [mD]Q, m = wlo .. wlo + T - 1
  win[0] = DQ;
  for (unsigned sd = 1; sd < T; sd *= 2) {
    std::vector<Pt> pa, pb, pd, po;
    for (unsigned j = 1; j < sd && sd + j <= T; j++) {
      pa.push_back(win[sd - 1]);
      pb.push_back(win[j - 1]);
      pd.push_back(win[sd - j - 1]);
      po.push_back(win[sd + j - 1] = take_pt());
    }
    W.xadd_n((unsigned)po.size(), pa.data(), pb.data(), pd.data(), po.data(),
             tmp);
    if (2 * sd <= T)
      W.dbl(win[sd - 1], win[2 * sd - 1] = take_pt());
  }
  Pt GD = take_pt();
  M.copy(GD.X, win[G - 1].X);
  M.copy(GD.Z, win[G - 1].Z);
  unsigned long long wlo = 1;
  // Leaves Z_m X - X_m = (-X_m, Z_m) for the giants in range, in
  // order of m, waiting for their block.
  std::vector<Poly> pend;
  auto emit = [&](const Pt *g, unsigned cnt, unsigned long long m0) {
    std::vector<slot_t> d, z, x;
    std::vector<int> ng;
    for (unsigned i = 0; i < cnt; i++) {
      const unsigned long long m = m0 + i;
      if (m < m_lo || m > m_hi)
        continue;
      Poly l = ops.take(2);
      d.push_back(l[0]);
      z.push_back(W.ZERO);
      x.push_back(g[i].X);
      ng.push_back(1);
      M.copy(l[1], g[i].Z);
      pend.push_back(l);
    }
    if (!d.empty())
      M.add((unsigned)d.size(), d.data(), z.data(), x.data(), ng.data());
  };
  emit(win.data(), T, 1);
  auto advance = [&]() { // one round: G new giants
    std::vector<Pt> pa(win.begin() + G, win.end()), pb(G, GD),
        pd(win.begin(), win.begin() + G), po(G);
    for (auto &p : po)
      p = take_pt();
    W.xadd_n(G, pa.data(), pb.data(), pd.data(), po.data(), tmp);
    for (unsigned i = 0; i < G; i++) {
      give_pt(win[i]);
      win[i] = win[G + i];
      win[G + i] = po[i];
    }
    wlo += G;
    emit(po.data(), G, wlo + G);
  };
  Poly H = ops.take(1);
  M.copy(H[0], W.ONE);
  unsigned blocks = 0;
  size_t used = 0; // leaves of pend already consumed
  for (unsigned long long m = m_lo; m <= m_hi; m += k) {
    if (r.stopped())
      return GMPECMFFT_STOPPED;
    const unsigned kb =
        (unsigned)(m_hi - m + 1 < k ? m_hi - m + 1 : (unsigned long long)k);
    while (pend.size() - used < kb)
      advance();
    std::vector<Poly> lg(pend.begin() + used, pend.begin() + used + kb);
    used += kb;
    if (used == pend.size()) {
      pend.clear();
      used = 0;
    }
    PolyOps::Tree Gt = ops.product_tree(lg, false);
    Poly Hn = ops.mulrem(H, Gt.back()[0], F, IF);
    ops.give(H);
    ops.give(Gt, true);
    H = Hn;
    blocks++;
    st = r.check();
    if (st != GMPECMFFT_FACTOR_FOUND)
      return st;
  }
  for (Pt p : win)
    give_pt(p);
  give_pt(GD);
  for (slot_t t : tmp)
    ops.give(t);
  const double t_g = now_seconds();

  // prod_j H(x_j)
  std::vector<slot_t> v = ops.evaluate(H, Ft, IF);
  ops.give(H);
  ops.give(IF);
  for (size_t cnt = v.size(); cnt > 1;) {
    const unsigned half = (unsigned)(cnt / 2);
    std::vector<slot_t> dd(half), aa(half), bb(half);
    for (unsigned j = 0; j < half; j++)
      dd[j] = v[j], aa[j] = v[2 * j], bb[j] = v[2 * j + 1];
    M.mul(half, dd.data(), aa.data(), bb.data());
    if (cnt & 1)
      M.copy(v[half], v[cnt - 1]);
    cnt = half + (cnt & 1);
  }
  st = r.check();
  if (st != GMPECMFFT_FACTOR_FOUND)
    return st;
  if (r.stats && r.detail)
    fprintf(stderr,
            "gmpecmfft stats: stage 2: polynomial, D %u, %u babies, %llu "
            "giants in %u blocks (B2 effectively %llu); F %.2f s, blocks "
            "%.2f s, evaluation %.2f s; %zu slots (estimate %u)\n",
            D, k, m_hi - m_lo + 1, blocks, (m_hi * D + D / 2), t_f - t0,
            t_g - t_f, now_seconds() - t_g, M.nslots(), poly_slots(k));
  const char *chk = getenv("GMPECMFFT_S2_CHECK");
  if (chk && *chk && *chk != '0' && hq) {
    MpzVec g(r.C);
    M.get(g.p, v[0]);
    mpz_t Ri, ref;
    mpz_inits(Ri, ref, NULL);
    mpz_invert(Ri, M.R(), r.n);
    for (unsigned c = 0; c < r.C && c < 2; c++) {
      if (r.dead[c] || r.s2dead[c])
        continue;
      h_stage2_product(ref, (*hq[0])[c], (*hq[1])[c], (*hq[2])[c], D, m_lo,
                       m_hi, G, r.n);
      mpz_mul(g[c], g[c], Ri);
      mpz_mod(g[c], g[c], r.n);
      fprintf(stderr, "gmpecmfft: stage 2 host check, curve %u: %s\n", c,
              mpz_cmp(g[c], ref) == 0 ? "identical" : "MISMATCH");
    }
    mpz_clears(Ri, ref, NULL);
  }
  *found = r.scan(v[0], factor, "stage 2", now_seconds() - t0);
  return GMPECMFFT_FACTOR_FOUND;
}

// Stage 2 after stage 1: polynomial when B1 and B2 make it worthwhile
// and it fits in memory, per prime otherwise. Q (stage 1's result) and
// a24 are in stage 1's FftMod, which the polynomial stage 2 replaces with
// its own; the values move through the host.
gmpecmfft_status_t stage2_any(Run &r, Curves &W1, Pt Q1, unsigned long b1,
                              unsigned long b2, mpz_ptr factor,
                              bool *found) {
  const char *env = getenv("GMPECMFFT_STAGE2");
  const bool want_prime = env && !strcmp(env, "prime"),
             want_poly = env && !strcmp(env, "poly");
  std::vector<unsigned> cand;
  if (!want_prime)
    for (unsigned i = 0; i < sizeof S2_POLY / sizeof *S2_POLY; i++) {
      const unsigned D = S2_POLY[i].D, k = S2_POLY[i].k;
      // D <= B1/2 (see stage2_poly), and at least half a block of giants
      // -- below that the per-prime stage 2 is cheaper -- unless forced.
      if (2ul * D <= b1 &&
          (want_poly || (b2 - b1 >= (unsigned long)D * k / 2 && D > 210)))
        cand.push_back(i);
    }
  if (cand.empty())
    return stage2(r, W1, Q1, b1, b2, factor, found);

  const unsigned C = r.C;
  MpzVec qx(C), qz(C), qa(C); // true values, not Montgomery form
  {
    FftMod &M1 = r.mod();
    M1.get(qx.p, Q1.X);
    M1.get(qz.p, Q1.Z);
    M1.get(qa.p, W1.A24);
    if (!M1.ok())
      return GMPECMFFT_ERR_CUDA;
    mpz_t Ri;
    mpz_init(Ri);
    mpz_invert(Ri, M1.R(), r.n);
    for (MpzVec *v : {&qx, &qz, &qa})
      for (unsigned c = 0; c < C; c++) {
        mpz_mul((*v)[c], (*v)[c], Ri);
        mpz_mod((*v)[c], (*v)[c], r.n);
      }
    mpz_clear(Ri);
  }
  // Stage 1's engine (and W1, Q1 with it) goes.
  r.Mp.reset(new FftMod);

  // The polynomial stage 2 needs memory per curve in proportion to k --
  // at D = 30030 (k = 2880) and 20000 bits, about a gigabyte -- so it
  // runs on groups of curves, one after the other, each on its own
  // engine: the largest D (most primes per giant) whose stage 2 fits for
  // some group size, the group as large as fits (halving from all C).
  // GMPECMFFT_S2_GROUP=N caps the group size (for testing: hit counts
  // must not depend on it).
  unsigned Cmax = C;
  if (const char *ge = getenv("GMPECMFFT_S2_GROUP")) {
    const unsigned long v = strtoul(ge, NULL, 10);
    if (v >= 1 && v < Cmax)
      Cmax = (unsigned)v;
  }
  int chosen = -1;
  unsigned Cg = Cmax;
  std::unique_ptr<FftMod> probe;
  for (unsigned i : cand) {
    for (Cg = Cmax;; Cg = (Cg + 1) / 2) {
      probe.reset(new FftMod);
      if (probe->init(r.n, Cg, poly_slots(S2_POLY[i].k), S2_POLY[i].k + 1)) {
        chosen = (int)i;
        break;
      }
      if (Cg == 1)
        break;
    }
    if (chosen >= 0)
      break;
  }
  probe.reset(); // re-created below, per group

  // Moves curves [c0, c0 + cnt) of the host values into a group's engine.
  auto setup = [&](Run &g, unsigned c0, unsigned cnt, Curves &W, Pt &Q) {
    FftMod &M = g.mod();
    if (!W.alloc())
      return GMPECMFFT_ERR_NO_MEMORY;
    Q = W.pt();
    if (!M.ok())
      return GMPECMFFT_ERR_NO_MEMORY;
    MpzVec x(cnt), z(cnt), a(cnt);
    for (unsigned c = 0; c < cnt; c++) {
      M.to_mont(x[c], qx[c0 + c]);
      M.to_mont(z[c], qz[c0 + c]);
      M.to_mont(a[c], qa[c0 + c]);
    }
    M.put(Q.X, x.p);
    M.put(Q.Z, z.p);
    M.put(W.A24, a.p);
    mpz_t one;
    mpz_init_set_ui(one, 1);
    M.to_mont(one, one);
    M.put_all(W.ONE, one);
    mpz_clear(one);
    if (g.stats && g.detail)
      fprintf(stderr,
              "gmpecmfft stats: stage 2 FFT: %u digits of %u bits, length "
              "%u, %u ops per batch\n",
              M.L(), M.b(), M.N(), M.maxk());
    return M.ok() ? GMPECMFFT_FACTOR_FOUND : GMPECMFFT_ERR_CUDA;
  };

  if (chosen < 0) {
    if (r.stats)
      fprintf(stderr, "gmpecmfft stats: polynomial stage 2 does not fit in "
                      "GPU memory; per-prime stage 2\n");
    if (!r.mod().init(r.n, C, slots_needed(true)))
      return GMPECMFFT_ERR_NO_MEMORY;
    Curves W(r.mod());
    Pt Q;
    gmpecmfft_status_t st = setup(r, 0, C, W, Q);
    if (st != GMPECMFFT_FACTOR_FOUND)
      return st;
    return stage2(r, W, Q, b1, b2, factor, found);
  }

  const unsigned D = S2_POLY[chosen].D, k = S2_POLY[chosen].k,
                 ngroups = (C + Cg - 1) / Cg;
  if (r.stats && ngroups > 1)
    fprintf(stderr,
            "gmpecmfft stats: polynomial stage 2 (D %u) fits %u curves at a "
            "time: %u groups\n",
            D, Cg, ngroups);
  const double t0 = now_seconds();
  unsigned hits = 0;
  *found = false;
  gmpecmfft_status_t res = GMPECMFFT_FACTOR_FOUND;
  for (unsigned c0 = 0; c0 < C; c0 += Cg) {
    const unsigned cnt = C - c0 < Cg ? C - c0 : Cg;
    // A Run of its own: the group's curves, renumbered from 0.
    Run g;
    mpz_set(g.n, r.n);
    g.C = cnt;
    g.stats = r.stats;
    g.quiet = true;
    g.detail = c0 == 0;
    g.stop = r.stop;
    g.stop_arg = r.stop_arg;
    g.dead.assign(r.dead.begin() + c0, r.dead.begin() + c0 + cnt);
    if (r.initg) {
      g.initg = new MpzVec(cnt);
      for (unsigned c = 0; c < cnt; c++)
        mpz_set((*g.initg)[c], (*r.initg)[c0 + c]);
    }
    if (!g.mod().init(r.n, cnt, poly_slots(k), k + 1)) {
      res = GMPECMFFT_ERR_NO_MEMORY;
      break;
    }
    Curves W(g.mod());
    Pt Q;
    gmpecmfft_status_t st = setup(g, c0, cnt, W, Q);
    if (st == GMPECMFFT_FACTOR_FOUND) {
      MpzVec gx(cnt), gz(cnt), ga(cnt);
      for (unsigned c = 0; c < cnt; c++) {
        mpz_set(gx[c], qx[c0 + c]);
        mpz_set(gz[c], qz[c0 + c]);
        mpz_set(ga[c], qa[c0 + c]);
      }
      MpzVec *hq[3] = {&gx, &gz, &ga};
      mpz_t f;
      mpz_init(f);
      bool gf = false;
      st = stage2_poly(g, D, k, W, Q, b1, b2, f, &gf, hq);
      if (g.maxe > r.maxe)
        r.maxe = g.maxe;
      if (st == GMPECMFFT_FACTOR_FOUND) {
        hits += g.hits;
        if (gf && !*found) {
          mpz_set(factor, f);
          *found = true;
        }
      }
      mpz_clear(f);
    }
    if (st != GMPECMFFT_FACTOR_FOUND) {
      res = st;
      break;
    }
    if (*found && !r.stats) // stats mode counts every group's hits
      break;
  }
  if (r.stats && res == GMPECMFFT_FACTOR_FOUND)
    fprintf(stderr,
            "gmpecmfft stats: stage 2: %u of %u curves found a factor "
            "(%.2f s)\n",
            hits, C, now_seconds() - t0);
  return res;
}

gmpecmfft_status_t run_fft(mpz_srcptr n, unsigned long b1, unsigned long b2,
                           unsigned curves, unsigned long seed,
                           int (*stop)(void *), void *stop_arg,
                           mpz_ptr factor) {
  Run r;
  mpz_set(r.n, n);
  r.stats = stats_on();
  r.stop = stop;
  r.stop_arg = stop_arg;
  const unsigned nslots = slots_needed(b2 > b1);
  if (!r.mod().init(n, curves, nslots)) {
    if (r.stats && r.mod().error())
      fprintf(stderr, "gmpecmfft: %s\n", r.mod().error());
    const char *e = r.mod().error();
    return e && strstr(e, "memory") ? GMPECMFFT_ERR_NO_MEMORY
                                    : GMPECMFFT_ERR_CUDA;
  }
  FftMod &M = r.mod();
  r.C = M.C();
  if (r.stats)
    fprintf(stderr,
            "gmpecmfft stats: n %u bits; FFT: %u digits of %u bits, "
            "length %u; %u curves, %u ops per batch\n",
            M.bits(), M.L(), M.b(), M.N(), M.C(), M.maxk());

  Curves W(M);
  if (!W.alloc())
    return GMPECMFFT_ERR_NO_MEMORY;
  Pt P = W.pt(), R0 = W.pt(), R1 = W.pt();
  if (!M.ok())
    return GMPECMFFT_ERR_NO_MEMORY;

  // Curve setup on the host (Suyama, as k_init in ecm_cuda.cu):
  // u = s^2 - 5, v = 4s, X = u^3, Z = v^3,
  // a24 = (v - u)^3 (3u + v) / (16 u^3 v).
  const unsigned C = r.C;
  r.dead.assign(C, 0);
  r.initg = new MpzVec(C);
  {
    MpzVec X(C), Z(C), A(C);
    gmp_randstate_t st;
    gmp_randinit_default(st);
    gmp_randseed_ui(st, seed);
    mpz_t sg, u, v, t, w, num, den;
    mpz_inits(sg, u, v, t, w, num, den, NULL);
    for (unsigned i = 0; i < C; i++) {
      mpz_set_ui(sg, 6 + (unsigned long)gmp_urandomm_ui(st, 1ul << 30));
      mpz_mul(u, sg, sg);
      mpz_sub_ui(u, u, 5);
      mpz_mod(u, u, n);
      mpz_mul_ui(v, sg, 4);
      mpz_mod(v, v, n);
      mpz_powm_ui(X[i], u, 3, n);
      mpz_powm_ui(Z[i], v, 3, n);
      mpz_sub(t, v, u);
      mpz_powm_ui(num, t, 3, n);
      mpz_mul_ui(t, u, 3);
      mpz_add(t, t, v);
      mpz_mul(num, num, t);
      mpz_mod(num, num, n);
      mpz_mul(den, X[i], v);
      mpz_mul_ui(den, den, 16);
      mpz_mod(den, den, n);
      if (!mpz_invert(w, den, n)) {
        mpz_gcd((*r.initg)[i], den, n);
        r.dead[i] = 1;
        mpz_set_ui(X[i], 0);
        mpz_set_ui(Z[i], 0);
        mpz_set_ui(A[i], 0);
        continue;
      }
      mpz_mul(A[i], num, w);
      mpz_mod(A[i], A[i], n);
      M.to_mont(X[i], X[i]);
      M.to_mont(Z[i], Z[i]);
      M.to_mont(A[i], A[i]);
    }
    mpz_clears(sg, u, v, t, w, num, den, NULL);
    gmp_randclear(st);
    M.put(P.X, X.p);
    M.put(P.Z, Z.p);
    M.put(W.A24, A.p);
  }
  {
    mpz_t one;
    mpz_init_set_ui(one, 1);
    M.to_mont(one, one);
    M.put_all(W.ONE, one); // ZERO is zero from allocation
    mpz_clear(one);
  }

  // k = prod p^e <= B1.
  mpz_t k;
  mpz_init_set_ui(k, 1);
  {
    mpz_t p;
    mpz_init_set_ui(p, 2);
    while (mpz_cmp_ui(p, b1) <= 0) {
      const unsigned long pi = mpz_get_ui(p);
      unsigned long pe = pi;
      while (pe <= b1 / pi)
        pe *= pi;
      mpz_mul_ui(k, k, pe);
      mpz_nextprime(p, p);
    }
    mpz_clear(p);
  }

  // Stage 1: ladder from (O, P). The top bit is done directly, which
  // also creates every FFT plan and loads cuFFT's kernels, so that the
  // two per-bit graphs can then be captured.
  const double t1 = now_seconds();
  M.copy(R0.X, W.ONE);
  M.copy(R0.Z, W.ZERO);
  M.copy(R1.X, P.X);
  M.copy(R1.Z, P.Z);
  // P and a24 are fixed for the whole of stage 1 (a24 for stage 2 too):
  // transform them once.
  xslot_t xP[3] = {M.xslot(), M.xslot(), M.xslot()};
  if (!M.ok()) {
    mpz_clear(k);
    return GMPECMFFT_ERR_NO_MEMORY;
  }
  {
    slot_t src[] = {P.X, P.Z, W.A24};
    M.xform(3, xP, src);
  }
  const Opnd oPX = X(xP[0]), oPZ = X(xP[1]);
  W.A = X(xP[2]);
  long bit = (long)mpz_sizeinbase(k, 2) - 1;
  M.prepare(4);
  W.ladder_step(mpz_tstbit(k, bit--), R0, R1, oPX, oPZ);
  gmpecmfft_status_t st = r.check();
  if (st != GMPECMFFT_FACTOR_FOUND) {
    mpz_clear(k);
    return st;
  }

  cudaGraphExec_t ge[2] = {NULL, NULL};
  bool graphs = true;
  for (int b = 0; b < 2 && graphs; b++) {
    cudaGraph_t g = NULL;
    if (cudaStreamBeginCapture(M.stream(), cudaStreamCaptureModeRelaxed) !=
        cudaSuccess) {
      graphs = false;
      break;
    }
    W.ladder_step(b, R0, R1, oPX, oPZ);
    if (cudaStreamEndCapture(M.stream(), &g) != cudaSuccess || !g ||
        cudaGraphInstantiate(&ge[b], g, 0) != cudaSuccess)
      graphs = false;
    if (g)
      cudaGraphDestroy(g);
  }
  if (!graphs) {
    cudaGetLastError(); // fall back to direct launches
    for (int b = 0; b < 2; b++)
      if (ge[b])
        cudaGraphExecDestroy(ge[b]);
    ge[0] = ge[1] = NULL;
  }
  if (r.stats && !graphs)
    fprintf(stderr, "gmpecmfft stats: CUDA graph capture unavailable, "
                    "launching directly\n");

  bool found = false;
  unsigned slice = 8;
  while (bit >= 0) {
    if (r.stopped()) {
      st = GMPECMFFT_STOPPED;
      break;
    }
    const unsigned cnt = (unsigned)(bit + 1 < (long)slice ? bit + 1 : slice);
    const double ts = now_seconds();
    for (unsigned j = 0; j < cnt; j++, bit--) {
      const int b = mpz_tstbit(k, bit);
      if (graphs)
        cudaGraphLaunch(ge[b], M.stream());
      else
        W.ladder_step(b, R0, R1, oPX, oPZ);
    }
    st = r.check();
    if (st != GMPECMFFT_FACTOR_FOUND)
      break;
    slice = next_slice(cnt, now_seconds() - ts);
  }
  for (int b = 0; b < 2; b++)
    if (ge[b])
      cudaGraphExecDestroy(ge[b]);
  mpz_clear(k);

  // A failed curve setup counts as a stage 1 find even when stopped.
  if (st == GMPECMFFT_FACTOR_FOUND || st == GMPECMFFT_STOPPED)
    found = r.scan(R0.Z, factor, "stage 1", now_seconds() - t1);
  if (st != GMPECMFFT_FACTOR_FOUND)
    return found ? GMPECMFFT_FACTOR_FOUND : st;

  // Stats mode runs stage 2 even after a stage 1 find, to count its hits
  // (cumulative, as on the CGBN path).
  if (b2 > b1 && (!found || r.stats)) {
    mpz_t f2;
    mpz_init(f2);
    bool found2 = false;
    st = stage2_any(r, W, R0, b1, b2, f2, &found2); // W, R0 may go
    if (!found && found2) {
      mpz_set(factor, f2);
      found = true;
    }
    mpz_clear(f2);
    if (st != GMPECMFFT_FACTOR_FOUND && !found)
      return st;
  }
  if (r.stats)
    fprintf(stderr, "gmpecmfft stats: worst FFT rounding error %.3g\n",
            r.maxe);
  return found ? GMPECMFFT_FACTOR_FOUND : GMPECMFFT_NO_FACTOR;
}

// ---------------------------------------------------------------- //
// Input handling shared by the entry points                          //
// ---------------------------------------------------------------- //

bool device_ok() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Parses n; returns INVALID_INPUT unless n is odd and >= 5.
gmpecmfft_status_t parse(mpz_ptr v, const char *n) {
  if (!n || mpz_set_str(v, n, 10) != 0 || mpz_cmp_ui(v, 5) < 0 ||
      mpz_even_p(v))
    return GMPECMFFT_ERR_INVALID_INPUT;
  return GMPECMFFT_FACTOR_FOUND;
}

char *dup_mpz(mpz_srcptr x) {
  const size_t len = mpz_sizeinbase(x, 10) + 2;
  char *s = (char *)malloc(len);
  if (s)
    mpz_get_str(s, 10, x);
  return s;
}

struct StopWrap {
  int (*f)(void *);
  void *a;
  int fired;
};

int stop_wrap(void *p) {
  StopWrap *w = (StopWrap *)p;
  if (w->f && w->f(w->a))
    w->fired = 1;
  return w->fired;
}

gmpecmfft_status_t ecm(const char *ns, unsigned long b1, unsigned long b2,
                       unsigned curves, unsigned long seed,
                       int (*stop)(void *), void *stop_arg, char **factor,
                       bool force_fft) {
  if (factor)
    *factor = NULL;
  if (b1 < 1000 || b1 > 0xffffffffUL)
    return GMPECMFFT_ERR_INVALID_INPUT;
  mpz_t n, f;
  mpz_init(n);
  gmpecmfft_status_t st = parse(n, ns);
  if (st != GMPECMFFT_FACTOR_FOUND) {
    mpz_clear(n);
    return st;
  }
  if (!device_ok()) {
    mpz_clear(n);
    return GMPECMFFT_ERR_CUDA;
  }
  if (!seed)
    seed = 1;
  mpz_init(f);
  if (!force_fft && mpz_sizeinbase(n, 2) <= GMPECMFFT_CGBN_MAX_BITS) {
    StopWrap w = {stop, stop_arg, 0};
    char *fs = NULL;
    gmpecpp_status_t s = gmpecpp_cuda_ecm(ns, b1, b2, curves, seed,
                                          stop ? stop_wrap : NULL, &w, &fs);
    if (s == GMPECPP_COMPOSITE) {
      st = GMPECMFFT_FACTOR_FOUND;
      if (fs)
        mpz_set_str(f, fs, 10);
    } else if (s == GMPECPP_PROBABLY_PRIME) {
      st = w.fired ? GMPECMFFT_STOPPED : GMPECMFFT_NO_FACTOR;
    } else if (s == GMPECPP_ERR_INVALID_INPUT) {
      st = GMPECMFFT_ERR_INVALID_INPUT;
    } else {
      st = GMPECMFFT_ERR_CUDA;
    }
    gmpecpp_string_free(fs);
  } else {
    st = run_fft(n, b1, b2, curves, seed, stop, stop_arg, f);
  }
  if (st == GMPECMFFT_FACTOR_FOUND && factor)
    *factor = dup_mpz(f);
  mpz_clear(f);
  mpz_clear(n);
  return st;
}

} // namespace
} // namespace gmpecmfft_detail

// ---------------------------------------------------------------- //
// C API                                                              //
// ---------------------------------------------------------------- //

using namespace gmpecmfft_detail;

extern "C" {

gmpecmfft_status_t gmpecmfft_ecm(const char *n, unsigned long b1,
                                 unsigned long b2, unsigned int curves,
                                 unsigned long seed, int (*stop)(void *),
                                 void *stop_arg, char **factor) {
  return ecm(n, b1, b2, curves, seed, stop, stop_arg, factor, false);
}

gmpecmfft_status_t gmpecmfft_ecm_fft(const char *n, unsigned long b1,
                                     unsigned long b2, unsigned int curves,
                                     unsigned long seed, int (*stop)(void *),
                                     void *stop_arg, char **factor) {
  return ecm(n, b1, b2, curves, seed, stop, stop_arg, factor, true);
}

unsigned int gmpecmfft_ecm_curves(const char *ns, int fft) {
  mpz_t n;
  mpz_init(n);
  unsigned c = 0;
  if (parse(n, ns) == GMPECMFFT_FACTOR_FOUND && device_ok()) {
    if (!fft && mpz_sizeinbase(n, 2) <= GMPECMFFT_CGBN_MAX_BITS) {
      c = gmpecpp_cuda_ecm_curves(ns);
    } else {
      FftMod M;
      if (M.init(n, 0, slots_needed(true)))
        c = M.C();
    }
  }
  mpz_clear(n);
  return c;
}

int gmpecmfft_selftest(const char *ns, unsigned int rounds,
                       unsigned int curves, unsigned long seed,
                       double *max_err) {
  if (max_err)
    *max_err = 0.0;
  mpz_t n;
  mpz_init(n);
  if (parse(n, ns) != GMPECMFFT_FACTOR_FOUND || !device_ok()) {
    mpz_clear(n);
    return -1;
  }
  FftMod M;
  if (!M.init(n, curves ? curves : 64, 8)) {
    mpz_clear(n);
    return -1;
  }
  const unsigned C = M.C();
  slot_t sa = M.slot(), sb = M.slot(), sc = M.slot(), sd = M.slot(),
         se = M.slot();
  if (!M.ok()) {
    mpz_clear(n);
    return -1;
  }
  MpzVec a(C), b(C), c(C), d(C), e(C);
  mpz_t two, four, Rinv, t, u;
  mpz_inits(two, four, Rinv, t, u, NULL);
  mpz_mul_ui(two, n, 2);
  mpz_mul_ui(four, n, 4);
  mpz_invert(Rinv, M.R(), n);
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, seed ? seed : 1);
  int bad = 0;
  double worst = 0.0;

  // Is x == a*b/R (mod n), with |x| < n (residues are signed)?
  auto mul_ok = [&](mpz_ptr x, mpz_ptr p, mpz_ptr q) {
    mpz_mul(t, p, q);
    mpz_mul(t, t, Rinv);
    mpz_sub(u, x, t);
    return mpz_cmpabs(x, n) < 0 && mpz_divisible_p(u, n);
  };
  // Uniform in (-lim, lim).
  auto sym = [&](mpz_ptr x, mpz_srcptr lim) {
    mpz_mul_2exp(t, lim, 1);
    mpz_sub_ui(t, t, 1);
    mpz_urandomm(x, rs, t);
    mpz_sub(x, x, lim);
    mpz_add_ui(x, x, 1);
  };

  for (unsigned r = 0; r < rounds && M.ok(); r++) {
    // Multiplication inputs anywhere in (-4n, 4n), with the extremes on
    // the first curves.
    for (unsigned i = 0; i < C; i++) {
      sym(a[i], four);
      sym(b[i], four);
    }
    mpz_sub_ui(a[0], four, 1);
    mpz_sub_ui(b[0], four, 1);
    if (C > 1) {
      mpz_set_ui(a[1], 0);
      mpz_sub_ui(b[1], four, 1);
    }
    if (C > 2) {
      mpz_sub_ui(a[2], four, 1);
      mpz_neg(a[2], a[2]);
      mpz_sub_ui(b[2], four, 1);
    }
    if (C > 3) {
      mpz_sub_ui(a[3], four, 1);
      mpz_neg(a[3], a[3]);
      mpz_set(b[3], a[3]);
    }
    M.put(sa, a.p);
    M.put(sb, b.p);
    {
      slot_t dd[] = {sc, sd, se}, aa[] = {sa, sa, sb}, bb[] = {sb, sa, sb};
      M.mul(3, dd, aa, bb);
    }
    M.get(c.p, sc);
    M.get(d.p, sd);
    M.get(e.p, se);
    for (unsigned i = 0; i < C; i++)
      bad += !mul_ok(c[i], a[i], b[i]) + !mul_ok(d[i], a[i], a[i]) +
             !mul_ok(e[i], b[i], b[i]);

    // Addition and subtraction inputs in (-2n, 2n): exact a + b, a - b.
    for (unsigned i = 0; i < C; i++) {
      sym(a[i], two);
      sym(b[i], two);
    }
    mpz_sub_ui(a[0], two, 1);
    mpz_set_ui(b[0], 0);
    if (C > 1) {
      mpz_set_ui(a[1], 0);
      mpz_sub_ui(b[1], two, 1);
    }
    if (C > 2) {
      mpz_sub_ui(a[2], two, 1);
      mpz_neg(b[2], a[2]);
    }
    M.put(sa, a.p);
    M.put(sb, b.p);
    {
      slot_t dd[] = {sc, sd}, aa[] = {sa, sa}, bb[] = {sb, sb};
      int ng[] = {0, 1};
      M.add(2, dd, aa, bb, ng);
    }
    M.get(c.p, sc);
    M.get(d.p, sd);
    for (unsigned i = 0; i < C; i++) {
      mpz_add(t, a[i], b[i]);
      bad += mpz_cmp(c[i], t) != 0;
      mpz_sub(t, a[i], b[i]);
      bad += mpz_cmp(d[i], t) != 0;
    }
    const double err = M.max_err();
    if (err > worst)
      worst = err;
    if (M.overflowed())
      bad++;
  }
  if (!M.ok())
    bad = -1;

  // Polynomial products (stage 2's arithmetic), in their own FftMod:
  // lengths 37 x 29, a squaring, and an output window running past the
  // end of the product (those coefficients must come out 0 mod n); then
  // trees, inverses and remainders. Up to 100000 bits: beyond that each
  // of the hundreds of products is a transform of tens of millions of
  // points per curve, and the polynomial code does not depend on n's
  // size apart from the digit arithmetic checked above.
  if (bad >= 0 && rounds && mpz_sizeinbase(n, 2) <= 100000) {
    const unsigned NA = 37, NB = 29;
    FftMod P;
    if (!P.init(n, curves ? curves : 16, 3 * (NA + NB) + 70, NA)) {
      bad = -1;
    } else {
      const unsigned Cp = P.C();
      std::vector<slot_t> A(NA), B(NB), O(70);
      std::vector<MpzVec *> av(NA), bv(NB);
      mpz_t Rinvp, acc;
      mpz_inits(Rinvp, acc, NULL);
      mpz_invert(Rinvp, P.R(), n);
      for (unsigned i = 0; i < NA; i++) {
        A[i] = P.slot();
        av[i] = new MpzVec(Cp);
        for (unsigned c = 0; c < Cp; c++)
          mpz_urandomm((*av[i])[c], rs, four);
        P.put(A[i], av[i]->p);
      }
      for (unsigned i = 0; i < NB; i++) {
        B[i] = P.slot();
        bv[i] = new MpzVec(Cp);
        for (unsigned c = 0; c < Cp; c++)
          mpz_urandomm((*bv[i])[c], rs, four);
        P.put(B[i], bv[i]->p);
      }
      for (auto &o : O)
        o = P.slot();
      MpzVec ov(Cp);
      // (A*B)[t] for t in [lo, lo + nd), and A*A.
      struct Case {
        bool sq;
        unsigned lo, nd;
      } cases[] = {{false, 0, NA + NB - 1}, {false, 20, 50}, {true, 0, 70}};
      for (const Case &cs : cases) {
        const unsigned nb2 = cs.sq ? NA : NB;
        const std::vector<MpzVec *> &bb = cs.sq ? av : bv;
        P.polymul(NA, A.data(), nb2, cs.sq ? A.data() : B.data(), cs.lo,
                  cs.nd, O.data());
        for (unsigned t = 0; t < cs.nd; t++) {
          P.get(ov.p, O[t]);
          const unsigned k = cs.lo + t;
          for (unsigned c = 0; c < Cp; c++) {
            mpz_set_ui(acc, 0);
            for (unsigned i = 0; i < NA; i++)
              if (k >= i && k - i < nb2)
                mpz_addmul(acc, (*av[i])[c], (*bb[k - i])[c]);
            mpz_mul(acc, acc, Rinvp);
            mpz_sub(acc, ov[c], acc);
            bad += !(mpz_cmpabs(ov[c], two) < 0 &&
                     mpz_divisible_p(acc, n));
          }
        }
      }
      // Trees, Newton inverse, remainders: build F = prod (X - r_j) over
      // 33 random r_j (not a power of two: a carried node), evaluate a
      // random polynomial A at the r_j through the scaled remainder
      // tree, and a 65-coefficient one after reducing it mod F; both
      // must match A(r_j) computed directly.
      {
        const unsigned S = 33;
        mpz_t one, Rv, v, x;
        mpz_inits(one, Rv, v, x, NULL);
        mpz_set_ui(one, 1);
        P.to_mont(one, one);
        slot_t ONE = P.slot(), ZERO = P.slot();
        P.put_all(ONE, one); // ZERO is zero from allocation
        PolyOps ops(P, ONE, ZERO);
        std::vector<MpzVec *> rv(S);
        std::vector<Poly> leaves(S);
        for (unsigned j = 0; j < S; j++) {
          rv[j] = new MpzVec(Cp);
          for (unsigned c = 0; c < Cp; c++)
            mpz_urandomm((*rv[j])[c], rs, two);
          leaves[j] = ops.take(2);
          P.put(leaves[j][1], rv[j]->p); // borrow as staging for r_j
          slot_t z[] = {ZERO};
          int ng[] = {1};
          slot_t d0[] = {leaves[j][0]}, b0[] = {leaves[j][1]};
          P.add(1, d0, z, b0, ng);        // c0 = -r_j
          P.copy(leaves[j][1], ONE);      // c1 = 1
        }
        PolyOps::Tree tree = ops.product_tree(leaves, true);
        for (unsigned len : {S, 2 * S - 1}) {
          Poly A = ops.take(len);
          std::vector<MpzVec *> avv(len);
          for (unsigned i = 0; i < len; i++) {
            avv[i] = new MpzVec(Cp);
            for (unsigned c = 0; c < Cp; c++)
              mpz_urandomm((*avv[i])[c], rs, two);
            P.put(A[i], avv[i]->p);
          }
          Poly Ar;
          if (len > S) {
            Poly IF = ops.inverse(tree.back()[0], len - S);
            Ar = ops.rem(A, tree.back()[0], IF);
            ops.give(IF);
          } else {
            Ar = A;
          }
          Poly IFr = ops.inverse(tree.back()[0], S);
          std::vector<slot_t> vals = ops.evaluate(Ar, tree, IFr);
          ops.give(IFr);
          for (unsigned j = 0; j < S; j++) {
            P.get(ov.p, vals[j]);
            for (unsigned c = 0; c < Cp; c++) {
              // True values: a_i = A~ / R, r = r~ / R; expect v~ = A(r) R.
              mpz_mul(x, (*rv[j])[c], Rinvp);
              mpz_mod(x, x, n);
              mpz_set_ui(v, 0);
              for (unsigned i = len; i-- > 0;) {
                mpz_mul(v, v, x);
                mpz_addmul(v, (*avv[i])[c], Rinvp);
                mpz_mod(v, v, n);
              }
              mpz_mul(v, v, P.R());
              mpz_sub(v, ov[c], v);
              bad += !(mpz_cmpabs(ov[c], two) < 0 &&
                       mpz_divisible_p(v, n));
            }
            ops.give(vals[j]);
          }
          if (len > S)
            ops.give(A);
          for (auto p : avv)
            delete p;
        }
        // mulrem(A, B, F) must agree mod n with rem(A * B, F), and be
        // below 2n: A of S coefficients, B of S + 1 (as H and G).
        {
          const Poly &F = tree.back()[0];
          Poly A = ops.take(S), B = ops.take(S + 1);
          MpzVec t(Cp);
          for (Poly *p : {&A, &B})
            for (slot_t c : *p) {
              for (unsigned i = 0; i < Cp; i++)
                mpz_urandomm(t[i], rs, two);
              P.put(c, t.p);
            }
          Poly IF = ops.inverse(F, S + 1);
          Poly AB = ops.mul_full(A, B);
          Poly R1 = ops.rem(AB, F, IF), R2 = ops.mulrem(A, B, F, IF);
          MpzVec o2(Cp);
          for (unsigned i = 0; i < S; i++) {
            P.get(ov.p, R1[i]);
            P.get(o2.p, R2[i]);
            for (unsigned c = 0; c < Cp; c++) {
              mpz_sub(v, ov[c], o2[c]);
              bad += !(mpz_cmpabs(o2[c], two) < 0 &&
                       mpz_divisible_p(v, n));
            }
          }
          for (const Poly *p : {&A, &B, &IF, &AB, &R1, &R2})
            ops.give(*p);
        }
        ops.give(tree, true);
        for (auto p : rv)
          delete p;
        mpz_clears(one, Rv, v, x, NULL);
      }
      const double perr = P.max_err();
      if (perr > worst)
        worst = perr;
      if (P.overflowed() || !P.ok())
        bad = bad >= 0 ? bad + 1 : bad;
      for (auto p : av)
        delete p;
      for (auto p : bv)
        delete p;
      mpz_clears(Rinvp, acc, NULL);
    }
  }
  if (max_err)
    *max_err = worst;
  gmp_randclear(rs);
  mpz_clears(two, four, Rinv, t, u, NULL);
  mpz_clear(n);
  return bad;
}

const char *gmpecmfft_status_string(gmpecmfft_status_t s) {
  switch (s) {
  case GMPECMFFT_FACTOR_FOUND:
    return "factor found";
  case GMPECMFFT_NO_FACTOR:
    return "no factor found";
  case GMPECMFFT_STOPPED:
    return "stopped";
  case GMPECMFFT_ERR_INVALID_INPUT:
    return "invalid input (n must be odd and >= 5, B1 in [1000, 2^32))";
  case GMPECMFFT_ERR_CUDA:
    return "CUDA or cuFFT error, or no usable device";
  case GMPECMFFT_ERR_PRECISION:
    return "FFT rounding error too large";
  case GMPECMFFT_ERR_NO_MEMORY:
    return "not enough GPU memory";
  case GMPECMFFT_ERR_INTERNAL:
    return "internal error (exactness check failed)";
  }
  return "unknown status";
}

void gmpecmfft_string_free(char *s) { free(s); }

} // extern "C"
