/*
 * fftmod.cuh -- batched modular arithmetic on the GPU by cuFFT
 * multiplication, for libgmpecmfft.
 *
 * A residue mod n is an integer x (of either sign) as L balanced digits
 * d_i in [-2^(b-1), 2^(b-1)), x = sum d_i 2^(bi) (int32, little end
 * first). Balanced digits make the worst convolution coefficient a
 * quarter of the unsigned digits' -- one more bit per digit for the same
 * FFT precision. A "slot"
 * holds one residue for each of C curves, contiguously: curve c's digits
 * start at slot + c*L. Every operation is applied to whole slots, so all
 * C curves move in lockstep -- which ECM allows, because every curve
 * walks the same scalar.
 *
 * Multiplication is Montgomery's REDC with R = 2^(b*L), the three integer
 * products done as double precision FFT convolutions:
 *
 *   T = a*b                          (2L digits, length N = 2L)
 *   m = (T mod R) * n'   mod R       (n' = -1/n mod R, length N)
 *   U = T_lo + T_hi + m*n mod (R-1)  (= (T + m*n)/R, cyclic, length L)
 *
 * The last is the half-length REDC (k_redc has the argument): since
 * m*n = -T_lo mod R, (T + m*n)/R = T_lo + T_hi + m*n mod R - 1, and
 * |U| far below R/2 makes the balanced residue U itself. So L is even
 * and 7-smooth.
 *
 * Bounds are on absolute values. m is balanced, |m| <= R/2, and
 * R > 32 poly_len n, so inputs with |a|, |b| < 4n give |U| < 16n^2/R +
 * n/2 < n (|U| < 1.5n for kout()'s differences). Sums and differences
 * a +- b of values with |.| < 2n are below 4n: legal multiplication
 * inputs, with no comparison against n and no offset. Nothing is ever
 * reduced to [0, n); that is fine for ECM, which only ever takes
 * gcd(Z, n).
 *
 * b is chosen so that a convolution coefficient is below 2^44 in the
 * worst case, and the rounding error of every coefficient is measured;
 * max_err() reports the worst seen, and the caller treats anything above
 * 0.25 as a precision failure rather than trusting the result.
 *
 * Copyright (c) 2026 Stefan Teleman and contributors.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.  See <http://www.gnu.org/licenses/>.
 */

#ifndef GMPECMFFT_FFTMOD_CUH
#define GMPECMFFT_FFTMOD_CUH

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>
#include <gmp.h>

namespace gmpecmfft_detail {

// Largest number of slot operations in one batched round. Rounds with
// more operations are split into chunks of at most maxk.
enum { MAXK = 64 };

typedef int32_t *slot_t;
// A slot's FFT, kept for an operand used many times (C rows of NC).
typedef cufftDoubleComplex *xslot_t;

// A multiplication operand: a slot (transformed on the fly; a slot used
// by several ops of one mul() call is transformed once), or a stored
// transform (not transformed at all).
struct Opnd {
  slot_t s;
  xslot_t x;
};
inline Opnd S(slot_t s) { return Opnd{s, NULL}; }
inline Opnd X(xslot_t x) { return Opnd{NULL, x}; }

// One polynomial product for FftMod::polymul(): out[t] = coefficient
// lo + t of A*B, t < nd. A has na coefficients, B nb.
struct PolyJob {
  unsigned na;
  const slot_t *A;
  unsigned nb;
  const slot_t *B;
  unsigned lo, nd;
  const slot_t *out;
};

class FftMod {
public:
  FftMod();
  ~FftMod();

  // Sets up arithmetic mod n (odd, >= 5) for `curves` curves (0: pick a
  // default from n's size), reserving room for `nslots` slots. Returns
  // false on failure; error() then says why.
  //
  // poly_len > 0 also allows polymul() with operands of up to poly_len
  // coefficients: the digits are made smaller, so that a product
  // coefficient's sum of poly_len terms keeps the same FFT precision
  // margin, and R gains log2(poly_len) bits, so that REDC of such a sum
  // still lands below 2n.
  bool init(mpz_srcptr n, unsigned curves, unsigned nslots,
            unsigned poly_len = 0);
  const char *error() const { return err_; }

  // Geometry.
  unsigned bits() const { return bits_; }  // of n
  unsigned b() const { return b_; }        // bits per digit
  unsigned L() const { return L_; }        // digits per residue
  unsigned N() const { return N_; }        // real FFT length
  unsigned C() const { return C_; }        // curves
  unsigned maxk() const { return maxk_; }  // ops per batched round
  mpz_srcptr R() const { return R_; }      // 2^(b*L)
  mpz_srcptr modulus() const { return n_; }

  // One slot (C residues). Owned by this object, freed by its destructor.
  slot_t slot();
  // Slots and transform slots allocated so far.
  size_t nslots() const { return slots_.size(); }

  // dst[i] = a[i] * b[i] / R mod n (in [0, 2n)), for i < k, on every
  // curve. dst may alias its own a/b, never another op's input.
  void mul(unsigned k, const slot_t *dst, const Opnd *a, const Opnd *b);
  void mul(unsigned k, const slot_t *dst, const slot_t *a, const slot_t *b);
  // A transform slot, and dst[i] = FFT(src[i]) for i < k.
  xslot_t xslot();
  void xform(unsigned k, const xslot_t *dst, const slot_t *src);

  // Polynomials are arrays of slots, coefficient i in slot i, every curve
  // its own polynomial. out[t] = coefficient lo + t of A*B, for t < nd,
  // each reduced like mul() (Montgomery form in, Montgomery form out: the
  // product of two polynomials in Montgomery form, in Montgomery form).
  // Coefficients past the end of the product come out zero. Kronecker
  // substitution: each curve's coefficients spaced 2L digits apart in one
  // long digit vector, one FFT per operand, one inverse. Needs
  // init(..., poly_len) with na, nb <= poly_len (only min(na, nb) terms
  // meet in a coefficient, so that is what must be <= poly_len). A == B
  // with na == nb is a squaring. out must not alias A or B.
  void polymul(unsigned na, const slot_t *A, unsigned nb, const slot_t *B,
               unsigned lo, unsigned nd, const slot_t *out);
  // Several independent products at once: every job's curves are rows
  // of one batched transform (as many jobs per batch as the Kronecker
  // buffers reserved by init() hold, padded to the longest job), and
  // their output coefficients share the REDC rounds. No job's out may
  // alias any job's A or B.
  void polymul(const std::vector<PolyJob> &jobs);

  // Kronecker products kept for later reduction (PolyOps::mulrem).
  // kprod() puts the exact product A*B of every curve in region `reg`
  // (0 or 1) of the product buffer, at Kronecker length NK (0: the least
  // that holds it; at most what init()'s poly_len allows). kout()
  // reduces coefficients lo .. lo + nd - 1 of region r0's product into
  // out -- minus region r1's, if r1 >= 0, before reducing (|difference|
  // < 32 poly_len n^2, which R > 32 poly_len n leaves room for). Either
  // way |out| < 2n, with one REDC per coefficient. A difference needs both
  // products at the same NK. polymul() overwrites region 0 (and region 1
  // if one of its products exceeds init()'s budget); kout() then fails
  // rather than read a stale product.
  unsigned kron_len(unsigned na, unsigned nb) const;
  void kprod(unsigned reg, unsigned na, const slot_t *A, unsigned nb,
             const slot_t *B, unsigned NK = 0);
  void kout(unsigned r0, int r1, unsigned lo, unsigned nd,
            const slot_t *out);
  // dst[i] = a[i] + b[i], or a[i] - b[i] when neg[i]; |inputs| < 2n.
  void add(unsigned k, const slot_t *dst, const slot_t *a, const slot_t *b,
           const int *neg);
  // dst = src on every curve.
  void copy(slot_t dst, slot_t src);

  // Host transfer. put() takes C values (|x| < 4n, any sign); get()
  // returns C, exactly as held (any sign: reduce mod n as needed).
  void put(slot_t s, const std::vector<mpz_ptr> &v);
  void put_all(slot_t s, mpz_srcptr x);  // the same value on every curve
  void get(std::vector<mpz_ptr> &v, slot_t s);

  // x*R mod n.
  void to_mont(mpz_ptr out, mpz_srcptr x) const;

  // Worst rounding error since the last call (synchronizes), and whether
  // an exactness check failed (an internal error, never expected).
  double max_err();
  bool overflowed();

  cudaStream_t stream() const { return s_; }
  bool ok() const { return cuda_ok_; }
  bool sync();
  // Pre-create every FFT plan a round of up to k ops can need, so that a
  // CUDA graph capture never has to create one.
  bool prepare(unsigned k);

private:
  struct Plan {
    cufftHandle fwd, inv;
  };
  Plan *plan(unsigned batch) { return plan_n(N_, batch); }
  Plan *plan_n(unsigned n, unsigned batch);
  void tail(unsigned k, const slot_t *dst);
  bool kron_buffers(size_t needA, size_t needB);
  void polymul_group(const PolyJob *jobs, unsigned m, unsigned NK);
  // Products of jobs[j].A, .B (outputs ignored) into rows j*C .. j*C +
  // C - 1 of XA, at length NK.
  bool kron_transform(const PolyJob *jobs, unsigned m, unsigned NK,
                      cufftDoubleComplex *XA);
  // REDC of coefficient coef[i] of the product in rows row0[i] + c of
  // r0 (minus r1's, if r1), into out[i], for i < cnt.
  void kron_out(const double *r0, const double *r1, unsigned NK,
                unsigned cnt, const unsigned *row0, const unsigned *coef,
                const slot_t *out);
  void chunk_mul(unsigned k, const slot_t *dst, const Opnd *a,
                 const Opnd *b);
  void fail(const char *why);

  mpz_t n_, R_, np_;
  unsigned bits_, b_, L_, N_, NC_, C_, maxk_;
  unsigned LC_, hd_; // L/2 + 1; digits a REDC output may occupy
  cudaStream_t s_;
  bool cuda_ok_;
  const char *err_;

  int32_t *d_n_;                    // n, L digits
  cufftDoubleComplex *d_Fn_, *d_Fnp_; // n at length L, n' at length N
  cufftDoubleComplex *d_A_;          // 2*maxk*C operand transforms
  cufftDoubleComplex *d_B_;          // maxk*C products
  int32_t *d_T_, *d_m_;             // maxk*C*2L, maxk*C*L
  unsigned *d_err_, *d_ovf_;
  void *d_work_;
  size_t work_size_;
  std::map<unsigned long long, Plan> plans_; // by (length, batch)
  cufftDoubleComplex *d_KA_, *d_KB_;          // Kronecker operands
  size_t kronA_cap_, kronB_cap_;               // complex values
  // Complex values per curve init() allows one product (KB, cuFFT's
  // work area, a region of KA); KA has two regions.
  size_t kron_budget_;
  unsigned kreg_NK_[2]; // kprod()'s length per region
  const int32_t **d_ptrs_;                     // coefficient tables
  size_t ptrs_cap_;
  uint2 *d_jobs_; // per job: offset into the coefficient table, count
  size_t jobs_cap_;
  unsigned poly_len_;
  std::vector<void *> slots_;
  std::vector<int32_t> h_buf_;
};

// Digit conversions (host).
bool mpz_to_digits(int32_t *d, unsigned L, unsigned b, mpz_srcptr x);
void digits_to_mpz(mpz_ptr x, const int32_t *d, unsigned L, unsigned b);

} // namespace gmpecmfft_detail

#endif
