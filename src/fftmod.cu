/*
 * fftmod.cu -- batched Montgomery arithmetic by cuFFT convolution.
 * See fftmod.cuh for the representation and the bounds it relies on.
 *
 * Copyright (c) 2026 Stefan Teleman and contributors.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.  See <http://www.gnu.org/licenses/>.
 */

#include "fftmod.cuh"

#include <math.h>
#include <string.h>

#include <map>
#include <mutex>
#include <tuple>

namespace gmpecmfft_detail {

namespace {

struct PtrK {
  const int32_t *p[2 * MAXK]; // up to two distinct operands per op
};
struct PtrC {
  const cufftDoubleComplex *p[MAXK];
};
struct PtrKW {
  int32_t *p[MAXK];
};

// ---------------------------------------------------------------- //
// Kernels                                                            //
// ---------------------------------------------------------------- //

// Block (op i, curve c) writes `cnt` digits of src.p[i] + c*cstride as
// doubles, zero padded to N, into a real FFT buffer of row stride rstride.
__global__ void k_pack(double *out, PtrK src, unsigned cstride, unsigned cnt,
                       unsigned C, unsigned rstride, unsigned N) {
  const unsigned blk = blockIdx.x, i = blk / C, c = blk % C;
  const int32_t *s = src.p[i] + (size_t)c * cstride;
  double *o = out + (size_t)blk * rstride;
  for (unsigned t = threadIdx.x; t < N; t += blockDim.x)
    o[t] = t < cnt ? (double)s[t] : 0.0;
}

__device__ __forceinline__ cufftDoubleComplex cmul(cufftDoubleComplex a,
                                                   cufftDoubleComplex b) {
  cufftDoubleComplex r;
  r.x = a.x * b.x - a.y * b.y;
  r.y = a.x * b.y + a.y * b.x;
  return r;
}

// P[i] = a[i] * b[i], pointwise, per op row of C*NC values.
__global__ void k_pmul(cufftDoubleComplex *P, PtrC a, PtrC b, unsigned k,
                       size_t row) {
  const size_t total = (size_t)k * row;
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < total;
       x += (size_t)gridDim.x * blockDim.x) {
    const unsigned i = (unsigned)(x / row);
    const size_t off = x - (size_t)i * row;
    P[x] = cmul(a.p[i][off], b.p[i][off]);
  }
}

// A[x] *= F[x mod NC]: multiply every row by one fixed transform.
__global__ void k_pconst(cufftDoubleComplex *A, const cufftDoubleComplex *F,
                         unsigned NC, size_t total) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < total;
       x += (size_t)gridDim.x * blockDim.x)
    A[x] = cmul(A[x], F[x % NC]);
}

// Exact carry normalization of M digits, block-wide. val(i) gives digit
// i's integer value (any sign, |v| < 2^62); the normalized digits,
// balanced -- each in [-2^(b-1), 2^(b-1)) -- are written to o. Returns
// (in thread blockDim-1) the carry out of digit M-1 (any sign).
//
// Each thread ripples its own chunk of digits, then the chunks' carries
// are passed up and rippled again until none is left. A carry entering a
// chunk is absorbed within a digit or two unless the chunk's digits sit
// at the edge of the range, so this settles in one or two rounds.
template <class F>
__device__ int64_t block_normalize(F val, int32_t *o, unsigned M,
                                   unsigned b, int64_t *sh) {
  const unsigned T = blockDim.x, t = threadIdx.x;
  const unsigned K = (M + T - 1) / T;
  const unsigned lo = min(M, t * K), hi = min(M, lo + K);
  const int64_t mask = ((int64_t)1 << b) - 1, half = (int64_t)1 << (b - 1);

  // v = d + c 2^b, d = ((v + half) mod 2^b) - half in [-half, half).
  int64_t c = 0;
  for (unsigned i = lo; i < hi; i++) {
    const int64_t v = val(i) + c + half;
    o[i] = (int32_t)((v & mask) - half);
    c = v >> b; // arithmetic: floor division, also for negative v
  }
  sh[t] = c;
  __syncthreads();

  int64_t top = 0;
  for (;;) {
    int64_t cin = t ? sh[t - 1] : 0;
    if (t == T - 1)
      top += sh[t];
    __syncthreads();
    for (unsigned i = lo; i < hi && cin != 0; i++) {
      const int64_t v = (int64_t)o[i] + cin + half;
      o[i] = (int32_t)((v & mask) - half);
      cin = v >> b;
    }
    sh[t] = cin;
    const int more = __syncthreads_or(cin != 0 && t != T - 1);
    if (!more) {
      if (t == T - 1)
        top += cin;
      break;
    }
  }
  return top;
}

__device__ __forceinline__ void warp_max_err(float e, unsigned *err) {
  for (int s = 16; s > 0; s >>= 1)
    e = fmaxf(e, __shfl_xor_sync(0xffffffffu, e, s));
  if ((threadIdx.x & 31) == 0 && e > 0.0f)
    atomicMax(err, __float_as_uint(e)); // e >= 0: bits order like values
}

struct RoundVal {
  const double *r;
  double scale;
  const int32_t *add; // NULL or digits to add
  float *e;
  int64_t c0;         // added to digit 0
  void *unused;
  __device__ int64_t operator()(unsigned i) const {
    const double x = r[i] * scale, xr = rint(x);
    *e = fmaxf(*e, (float)fabs(x - xr));
    int64_t v = (int64_t)xr;
    if (add)
      v += add[i];
    if (i == 0)
      v += c0;
    return v;
  }
};

// After an inverse transform: round, carry, store.
//   MODE 0: T = a*b, M = 2L digits into o.
//   MODE 1: m = low L digits of T_low * n' into o (carry out discarded:
//           that is the reduction mod R).
template <int MODE>
__global__ void k_carry(const double *r, unsigned rstride, double scale,
                        unsigned M, int32_t *o, unsigned ostride,
                        unsigned b, unsigned *err, unsigned *ovf) {
  __shared__ int64_t sh[1024];
  const unsigned blk = blockIdx.x;
  int32_t *oo = o + (size_t)blk * ostride;
  float e = 0.0f;
  RoundVal f = {r + (size_t)blk * rstride, scale, NULL, &e, 0, NULL};
  const int64_t top = block_normalize(f, oo, M, b, sh);
  warp_max_err(e, err);
  if (MODE == 0 && threadIdx.x == blockDim.x - 1 && top != 0)
    atomicOr(ovf, 1u);
}

// REDC: dst = (T + m*n) / R, from m*n mod (R - 1) -- a cyclic
// convolution of length L, half the length of the full product.
//
// With T = T_hi R + T_lo and m*n = h R + l, l = -T_lo mod R, so
// T_lo + l = c R and the result is U = T_hi + h + c. Mod R - 1, where
// R = 1: m*n = h + l and l = c - T_lo, so h = m*n - c + T_lo and
// U = T_lo + T_hi + m*n (mod R - 1) -- the carry c cancels. With
// balanced digits a normalized L-digit value lies within about R/2 of
// zero, and |U| < 2n is far inside that, so the residue is U itself. So:
// round the cyclic convolution, add T's two halves, and normalize with
// the carry out of the top digit wrapped around to digit 0 (B^L = 1)
// until none is left.
//
// The full-length REDC's check -- the low half of T + m*n comes out zero
// -- has no counterpart here; instead every digit from hd up (2n <
// 2^(b hd - 1)) must be zero, which a wrong m would almost always
// violate.
struct CycVal {
  const double *r;
  double scale;
  const int32_t *T;
  unsigned L;
  float *e;
  __device__ int64_t operator()(unsigned j) const {
    const double x = r[j] * scale, xr = rint(x);
    *e = fmaxf(*e, (float)fabs(x - xr));
    return (int64_t)xr + T[j] + T[L + j];
  }
};
struct WrapVal { // digits already in o, plus a carry into digit 0
  const int32_t *o;
  int64_t c0;
  __device__ int64_t operator()(unsigned j) const {
    return (int64_t)o[j] + (j == 0 ? c0 : 0);
  }
};

__global__ void k_redc(const double *r, unsigned rstride, double scale,
                       const int32_t *T, PtrKW dst, unsigned C, unsigned L,
                       unsigned b, unsigned hd, unsigned *err,
                       unsigned *ovf) {
  __shared__ int64_t sh[1024];
  __shared__ int64_t wrap;
  const unsigned blk = blockIdx.x, i = blk / C, c = blk % C;
  int32_t *o = dst.p[i] + (size_t)c * L;
  float e = 0.0f;
  CycVal f = {r + (size_t)blk * rstride, scale, T + (size_t)blk * 2 * L, L,
              &e};
  int64_t top = block_normalize(f, o, L, b, sh);
  warp_max_err(e, err);
  for (;;) { // end-around carry: at most a few rounds
    if (threadIdx.x == blockDim.x - 1)
      wrap = top;
    __syncthreads();
    const int64_t w = wrap;
    __syncthreads();
    if (w == 0)
      break;
    WrapVal g = {o, w};
    top = block_normalize(g, o, L, b, sh);
  }
  bool high = false;
  for (unsigned j = hd + threadIdx.x; j < L; j += blockDim.x)
    high |= o[j] != 0;
  if (__syncthreads_or(high) && threadIdx.x == 0)
    atomicOr(ovf, 2u);
}

struct KVal {
  const double *r, *r2; // r2: a product to subtract (or NULL)
  size_t off;
  unsigned NK;
  double scale;
  float *e;
  __device__ int64_t get(const double *q, unsigned j) const {
    if (off + j >= NK)
      return 0;
    const double x = q[off + j] * scale, xr = rint(x);
    *e = fmaxf(*e, (float)fabs(x - xr));
    return (int64_t)xr;
  }
  __device__ int64_t operator()(unsigned j) const {
    return r2 ? get(r, j) - get(r2, j) : get(r, j);
  }
};

// Kronecker packing: row r = j*C + c is job j's polynomial on curve c,
// coefficient i's L digits at i*S of one digit vector of NK doubles.
// job[j] = (offset of its first coefficient in coef, coefficient count).
__global__ void k_kpack(double *out, const int32_t *const *coef,
                        const uint2 *job, unsigned rows, unsigned C,
                        unsigned L, unsigned S, size_t rstride, unsigned NK) {
  for (unsigned row = blockIdx.y; row < rows; row += gridDim.y) {
    const unsigned j = row / C, c = row - j * C;
    const uint2 jb = job[j];
    double *o = out + (size_t)row * rstride;
    for (unsigned p = blockIdx.x * blockDim.x + threadIdx.x; p < NK;
         p += gridDim.x * blockDim.x) {
      const unsigned i = p / S, d = p - i * S;
      o[p] = (i < jb.y && d < L) ? (double)coef[jb.x + i][(size_t)c * L + d]
                                 : 0.0;
    }
  }
}

__global__ void k_kmul(cufftDoubleComplex *A, const cufftDoubleComplex *B,
                       size_t total) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < total;
       x += (size_t)gridDim.x * blockDim.x)
    A[x] = cmul(A[x], B[x]);
}

// Output row i of a round: coefficient coef[i] of the product in rows
// row0[i] + c of the inverse transform.
struct KRows {
  unsigned row0[MAXK], coef[MAXK];
};

// Product coefficient coef[i] of curve c: the 2L digits at coef[i]*S of
// row row0[i] + c (minus r2's, if r2), rounded and normalized into T row
// i*C + c. A coefficient sums up to poly_len terms; 2L digits still hold
// it (a carry out of them is an exactness failure: ovf).
__global__ void k_kcarry(const double *r, const double *r2, size_t rstride,
                         unsigned S,
                         KRows rows, unsigned NK, double scale, unsigned C,
                         unsigned L, unsigned b, int32_t *T, unsigned *err,
                         unsigned *ovf) {
  __shared__ int64_t sh[1024];
  const unsigned blk = blockIdx.x, i = blk / C, c = blk % C;
  const size_t off = (size_t)rows.coef[i] * S,
               ro = (size_t)(rows.row0[i] + c) * rstride;
  float e = 0.0f;
  KVal f = {r + ro, r2 ? r2 + ro : NULL, off, NK, scale, &e};
  const int64_t top =
      block_normalize(f, T + (size_t)blk * 2 * L, 2 * L, b, sh);
  warp_max_err(e, err);
  if (threadIdx.x == blockDim.x - 1 && top != 0)
    atomicOr(ovf, 1u);
}

struct AddVal {
  const int32_t *a, *b;
  bool neg;
  __device__ int64_t operator()(unsigned i) const {
    return neg ? (int64_t)a[i] - b[i] : (int64_t)a[i] + b[i];
  }
};

__global__ void k_add(PtrKW dst, PtrK a, PtrK b, unsigned long long neg,
                      unsigned C, unsigned L, unsigned bb, unsigned *ovf) {
  __shared__ int64_t sh[1024];
  const unsigned blk = blockIdx.x, i = blk / C, c = blk % C;
  const size_t off = (size_t)c * L;
  AddVal f = {a.p[i] + off, b.p[i] + off, ((neg >> i) & 1ull) != 0};
  const int64_t top = block_normalize(f, dst.p[i] + off, L, bb, sh);
  if (threadIdx.x == blockDim.x - 1 && top != 0)
    atomicOr(ovf, 4u);
}

// Threads per block for normalizing M digits: ~8 digits each, at least
// one warp, at most 1024.
unsigned norm_threads(unsigned M) {
  unsigned t = (M / 8 + 31) / 32 * 32;
  return t < 32 ? 32 : (t > 1024 ? 1024 : t);
}

unsigned grid_for(size_t total) {
  size_t g = (total + 255) / 256;
  return (unsigned)(g > 65535 * 8 ? 65535 * 8 : (g ? g : 1));
}

bool smooth7(unsigned x) {
  for (unsigned p : {2u, 3u, 5u, 7u})
    while (x % p == 0)
      x /= p;
  return x == 1;
}

// cuFFT's time (ms) for one in-place R2C + C2R pair of length len over
// `batch` rows of buf; negative on failure.
float fft_pair_ms(unsigned len, unsigned batch, double *buf) {
  int n = (int)len, rin = (int)(2 * (len / 2 + 1)), cin = (int)(len / 2 + 1);
  cufftHandle f, i;
  if (cufftPlanMany(&f, 1, &n, &rin, 1, rin, &cin, 1, cin, CUFFT_D2Z,
                    (int)batch) != CUFFT_SUCCESS)
    return -1.0f;
  if (cufftPlanMany(&i, 1, &n, &cin, 1, cin, &rin, 1, rin, CUFFT_Z2D,
                    (int)batch) != CUFFT_SUCCESS) {
    cufftDestroy(f);
    return -1.0f;
  }
  cufftDoubleComplex *z = (cufftDoubleComplex *)buf;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  bool ok = cufftExecD2Z(f, buf, z) == CUFFT_SUCCESS &&
            cufftExecZ2D(i, z, buf) == CUFFT_SUCCESS; // warm up
  const int reps = 5;
  cudaEventRecord(e0);
  for (int r = 0; r < reps && ok; r++)
    ok = cufftExecD2Z(f, buf, z) == CUFFT_SUCCESS &&
         cufftExecZ2D(i, z, buf) == CUFFT_SUCCESS;
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = -1.0f;
  if (ok && cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess)
    ms /= reps;
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cufftDestroy(f);
  cufftDestroy(i);
  return ok ? ms : -1.0f;
}

// The digit count to use: among the even 7-smooth L in [Lmin, Lmax], the
// one whose transforms run fastest on this device, as a multiplication
// uses them -- about 2.25 R2C/C2R pairs of length 2L (the product and
// REDC's first step) and one of length L (its second). cuFFT's speed
// depends on the factorization far more than on the length: at 20000
// bits and 64 curves, L = 1280 (2^8 5) ran stage 1 15% faster than the
// shorter L = 1200 (2^4 3 5^2), and L = 1344 slower than both. Measured
// once per (Lmin, Lmax, batch) and process.
unsigned tune_L(unsigned Lmin, unsigned Lmax, unsigned batch) {
  static std::mutex mu;
  static std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> memo;
  std::lock_guard<std::mutex> lock(mu);
  const auto key = std::make_tuple(Lmin, Lmax, batch);
  auto it = memo.find(key);
  if (it != memo.end())
    return it->second;
  std::vector<unsigned> cand;
  for (unsigned L = Lmin + (Lmin & 1); L <= Lmax; L += 2)
    if (smooth7(L))
      cand.push_back(L);
  unsigned best = cand.empty() ? 0 : cand[0];
  if (cand.size() > 1) {
    double *buf = NULL;
    const size_t need = (size_t)batch * 2 * (cand.back() + 1);
    if (cudaMalloc(&buf, sizeof(double) * need) == cudaSuccess) {
      cudaMemset(buf, 0, sizeof(double) * need);
      // Two passes, each candidate's best: the first may run while the
      // GPU's clocks are still rising.
      std::vector<float> cost(cand.size(), -1.0f);
      for (int pass = 0; pass < 2; pass++)
        for (size_t x = 0; x < cand.size(); x++) {
          const float t2 = fft_pair_ms(2 * cand[x], batch, buf),
                      t1 = fft_pair_ms(cand[x], batch, buf);
          if (t2 < 0.0f || t1 < 0.0f)
            continue;
          const float c = 2.25f * t2 + t1;
          if (cost[x] < 0.0f || c < cost[x])
            cost[x] = c;
        }
      float bc = -1.0f;
      for (size_t x = 0; x < cand.size(); x++)
        if (cost[x] >= 0.0f && (bc < 0.0f || cost[x] < bc)) {
          bc = cost[x];
          best = cand[x];
        }
      cudaFree(buf);
    }
  }
  memo[key] = best;
  return best;
}

} // namespace

// ---------------------------------------------------------------- //
// Digit conversion                                                   //
// ---------------------------------------------------------------- //

// Balanced digits, d in [-2^(b-1), 2^(b-1)), of x (any sign). Returns
// false if x does not fit in L of them (|x| >= about 2^(bL)/2); the
// digits are then x mod 2^(bL), which is what n' needs.
bool mpz_to_digits(int32_t *d, unsigned L, unsigned b, mpz_srcptr x) {
  const size_t nl = mpz_size(x); // limbs of |x|
  const int64_t mask = ((int64_t)1 << b) - 1, half = (int64_t)1 << (b - 1);
  int64_t c = 0;
  for (unsigned i = 0; i < L; i++) {
    const uint64_t bit = (uint64_t)i * b;
    const size_t li = bit / 64;
    const unsigned off = bit % 64;
    uint64_t v = li < nl ? (uint64_t)mpz_getlimbn(x, li) >> off : 0;
    if (off + b > 64 && li + 1 < nl)
      v |= (uint64_t)mpz_getlimbn(x, li + 1) << (64 - off);
    const int64_t u = (int64_t)(v & (uint64_t)mask) + c + half;
    d[i] = (int32_t)((u & mask) - half);
    c = u >> b;
  }
  const bool fits = c == 0 && (uint64_t)L * b >= mpz_sizeinbase(x, 2);
  if (mpz_sgn(x) < 0) { // negate, renormalizing: -(-half) is out of range
    c = 0;
    for (unsigned i = 0; i < L; i++) {
      const int64_t u = -(int64_t)d[i] + c + half;
      d[i] = (int32_t)((u & mask) - half);
      c = u >> b;
    }
  }
  return fits;
}

// The integer whose balanced digits are d.
void digits_to_mpz(mpz_ptr x, const int32_t *d, unsigned L, unsigned b) {
  // Positive and negative digits apart, each a plain base-2^b number.
  const size_t nl = ((size_t)L * b + 63) / 64 + 1;
  std::vector<uint64_t> lp(nl, 0), ln(nl, 0);
  for (unsigned i = 0; i < L; i++) {
    const uint64_t bit = (uint64_t)i * b;
    const size_t li = bit / 64;
    const unsigned off = bit % 64;
    std::vector<uint64_t> &limb = d[i] < 0 ? ln : lp;
    const uint64_t v = (uint64_t)(d[i] < 0 ? -(int64_t)d[i] : d[i]);
    limb[li] |= v << off;
    if (off + b > 64)
      limb[li + 1] |= v >> (64 - off);
  }
  mpz_t y;
  mpz_init(y);
  mpz_import(x, nl, -1, sizeof(uint64_t), 0, 0, lp.data());
  mpz_import(y, nl, -1, sizeof(uint64_t), 0, 0, ln.data());
  mpz_sub(x, x, y);
  mpz_clear(y);
}

// ---------------------------------------------------------------- //
// FftMod                                                             //
// ---------------------------------------------------------------- //

FftMod::FftMod()
    : bits_(0), b_(0), L_(0), N_(0), NC_(0), C_(0), maxk_(0), LC_(0), hd_(0),
      s_(NULL),
      cuda_ok_(true), err_(NULL), d_n_(NULL), d_Fn_(NULL),
      d_Fnp_(NULL), d_A_(NULL), d_B_(NULL), d_T_(NULL), d_m_(NULL),
      d_err_(NULL), d_ovf_(NULL), d_work_(NULL), work_size_(0),
      d_KA_(NULL), d_KB_(NULL), kronA_cap_(0), kronB_cap_(0),
      kron_budget_(0), kreg_NK_{0, 0},
      d_ptrs_(NULL), ptrs_cap_(0), d_jobs_(NULL), jobs_cap_(0),
      poly_len_(0) {
  mpz_inits(n_, R_, np_, NULL);
}

FftMod::~FftMod() {
  if (s_)
    cudaStreamSynchronize(s_);
  for (auto &kv : plans_) {
    cufftDestroy(kv.second.fwd);
    cufftDestroy(kv.second.inv);
  }
  for (void *p : slots_)
    cudaFree(p);
  void *bufs[] = {d_n_,   d_Fn_, d_Fnp_, d_A_,   d_B_,
                  d_T_,   d_m_,    d_err_, d_ovf_, d_work_, d_KA_,
                  d_KB_,  (void *)d_ptrs_, d_jobs_};
  for (void *p : bufs)
    if (p)
      cudaFree(p);
  if (s_)
    cudaStreamDestroy(s_);
  mpz_clears(n_, R_, np_, NULL);
}

void FftMod::fail(const char *why) {
  if (!err_)
    err_ = why;
  cuda_ok_ = false;
}

bool FftMod::init(mpz_srcptr n, unsigned curves, unsigned nslots,
                  unsigned poly_len) {
  mpz_set(n_, n);
  bits_ = (unsigned)mpz_sizeinbase(n, 2);
  poly_len_ = poly_len;
  unsigned lgk = 0;
  while ((1u << lgk) < poly_len)
    lgk++;

  // Digit size: the largest b whose worst case convolution coefficient
  // stays below 2^44 -- with balanced digits, |d| <= 2^(b-1), that is
  // L (2^(b-1))^2 = 2^(2b - 2 + log2 L), and poly_len times that for
  // polymul(). That leaves nine bits between the coefficients and the
  // 53-bit mantissa for the FFT's rounding error, which max_err()
  // measures rather than assumes. (Unsigned digits needed 2b + log2 L
  // <= 44: balancing is one bit more per digit.)
  for (unsigned b = 20; b >= 6; b--) {
    // R = 2^(bL) > 32n, or > 32 * poly_len * n: 16 for products of
    // inputs below 4n, twice that for kout()'s differences of them.
    // L even and 7-smooth: REDC's second product is a cyclic
    // convolution of length exactly L (see k_redc), and N = 2L.
    unsigned L = (bits_ + 5 + lgk + b - 1) / b;
    L += L & 1;
    while (!smooth7(L))
      L += 2;
    unsigned lg = 0;
    while ((1u << lg) < L)
      lg++;
    if (2 * b + lg + lgk <= 46) {
      b_ = b;
      L_ = L;
      break;
    }
  }
  if (!b_) {
    fail("number too large for double precision FFT arithmetic");
    return false;
  }
  {
    // A longer L can be faster (tune_L); it only makes R larger, but
    // must not raise log2 L past what the digit size allows.
    const unsigned lgmax = 46 - 2 * b_ - lgk;
    unsigned Lmax = L_ + L_ / 4;
    if (lgmax < 31 && Lmax > (1u << lgmax))
      Lmax = 1u << lgmax;
    size_t cg = curves;
    if (!cg) {
      cg = ((size_t)1 << 19) / (2 * L_);
      cg = cg < 8 ? 8 : (cg > 2048 ? 2048 : cg);
    }
    const unsigned t = tune_L(L_, Lmax, (unsigned)(4 * cg));
    if (t)
      L_ = t;
  }
  N_ = 2 * L_; // 7-smooth, as L is
  NC_ = N_ / 2 + 1;
  LC_ = L_ / 2 + 1;
  hd_ = (bits_ + 2 + b_ - 1) / b_; // |REDC output| < 2n < 2^(b hd - 1)

  mpz_set_ui(R_, 1);
  mpz_mul_2exp(R_, R_, (mp_bitcnt_t)b_ * L_);
  if (!mpz_invert(np_, n_, R_)) {
    fail("modulus is even");
    return false;
  }
  mpz_sub(np_, R_, np_); // -1/n mod R

  if (cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking) != cudaSuccess) {
    fail("cudaStreamCreate failed");
    return false;
  }

  // Curves and batch width from free memory. Per (op, curve) a round
  // needs up to two operand transforms (A), the product transform (B),
  // the 2L-digit T, the L-digit m, and about one more transform of cuFFT
  // work area.
  size_t freeb = 0, totalb = 0;
  if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) {
    fail("cudaMemGetInfo failed");
    return false;
  }
  const size_t per_op = 4 * sizeof(cufftDoubleComplex) * (size_t)NC_ +
                        12 * (size_t)L_;
  const size_t per_slot = 4 * (size_t)L_;
  const size_t budget = (size_t)(freeb * 0.85);
  unsigned C = curves;
  if (!C) {
    // Enough curves that a round's transforms fill the GPU; measured at
    // 20000 bits, throughput per curve stops improving near 256 curves
    // and falls beyond ~1000.
    size_t c = ((size_t)1 << 19) / N_;
    C = (unsigned)(c < 8 ? 8 : (c > 2048 ? 2048 : c));
  }
  // polymul()'s two Kronecker operands for the longest product, and as
  // much again for cuFFT's work area, per curve.
  size_t per_curve_kron = 0;
  if (poly_len_) {
    unsigned nk = 2 * poly_len_ * 2 * L_;
    while (!smooth7(nk))
      nk += 2;
    // KA two regions (kprod), KB and cuFFT's work area one each.
    kron_budget_ = (size_t)nk / 2 + 1;
    per_curve_kron = 4 * sizeof(cufftDoubleComplex) * kron_budget_;
  }
  for (;;) {
    const size_t fixed =
        (size_t)nslots * C * per_slot + (size_t)C * per_curve_kron;
    size_t k = fixed < budget ? (budget - fixed) / ((size_t)C * per_op) : 0;
    if (k > MAXK)
      k = MAXK;
    if (k >= 4) {
      C_ = C;
      maxk_ = (unsigned)k;
      break;
    }
    if (curves || C <= 1) {
      fail("not enough GPU memory for this many curves at this size");
      return false;
    }
    C = C / 2;
  }

  const size_t nA = (size_t)maxk_ * C_ * NC_;
  bool ok = true;
  ok = ok && cudaMalloc(&d_n_, 4 * (size_t)L_) == cudaSuccess;
  ok = ok && cudaMalloc(&d_Fn_, sizeof(cufftDoubleComplex) * LC_) ==
                 cudaSuccess;
  ok = ok && cudaMalloc(&d_Fnp_, sizeof(cufftDoubleComplex) * NC_) ==
                 cudaSuccess;
  ok = ok && cudaMalloc(&d_A_, 2 * sizeof(cufftDoubleComplex) * nA) ==
                 cudaSuccess;
  ok = ok && cudaMalloc(&d_B_, sizeof(cufftDoubleComplex) * nA) == cudaSuccess;
  ok = ok && cudaMalloc(&d_T_, 8 * (size_t)L_ * maxk_ * C_) == cudaSuccess;
  ok = ok && cudaMalloc(&d_m_, 4 * (size_t)L_ * maxk_ * C_) == cudaSuccess;
  ok = ok && cudaMalloc(&d_err_, sizeof(unsigned)) == cudaSuccess;
  ok = ok && cudaMalloc(&d_ovf_, sizeof(unsigned)) == cudaSuccess;
  if (!ok) {
    fail("cudaMalloc failed");
    return false;
  }
  cudaMemsetAsync(d_err_, 0, sizeof(unsigned), s_);
  cudaMemsetAsync(d_ovf_, 0, sizeof(unsigned), s_);

  // n, and the transforms of n and n' (computed once, shared by every
  // curve).
  std::vector<int32_t> h(L_);
  mpz_t t;
  mpz_init(t);
  mpz_to_digits(h.data(), L_, b_, n_);
  cudaMemcpy(d_n_, h.data(), 4 * (size_t)L_, cudaMemcpyHostToDevice);
  // n' at length N (for m = T n' mod R, a linear product), n at length L
  // (for m n mod (R - 1), a cyclic one).
  for (int which = 0; which < 2; which++) {
    const unsigned len = which ? N_ : L_, lc = len / 2 + 1;
    Plan *p1 = plan_n(len, 1);
    if (!p1) {
      mpz_clear(t);
      return false;
    }
    if (which) {
      mpz_to_digits(h.data(), L_, b_, np_);
      cudaMemcpy(d_m_, h.data(), 4 * (size_t)L_, cudaMemcpyHostToDevice);
    }
    PtrK src;
    src.p[0] = which ? d_m_ : d_n_;
    k_pack<<<1, 256, 0, s_>>>((double *)d_A_, src, L_, L_, 1, 2 * lc, len);
    if (cufftExecD2Z(p1->fwd, (cufftDoubleReal *)d_A_, d_A_) !=
        CUFFT_SUCCESS) {
      fail("cufftExecD2Z failed");
      break;
    }
    cudaMemcpyAsync(which ? d_Fnp_ : d_Fn_, d_A_,
                    sizeof(cufftDoubleComplex) * lc,
                    cudaMemcpyDeviceToDevice, s_);
  }
  mpz_clear(t);
  return sync();
}

FftMod::Plan *FftMod::plan_n(unsigned len, unsigned batch) {
  const unsigned long long key = ((unsigned long long)len << 32) | batch;
  auto it = plans_.find(key);
  if (it != plans_.end())
    return &it->second;
  Plan p;
  int n = (int)len, rin = (int)(2 * (len / 2 + 1)), cin = (int)(len / 2 + 1);
  size_t wf = 0, wi = 0;
  if (cufftCreate(&p.fwd) != CUFFT_SUCCESS ||
      cufftCreate(&p.inv) != CUFFT_SUCCESS) {
    fail("cufftCreate failed");
    return NULL;
  }
  cufftSetAutoAllocation(p.fwd, 0);
  cufftSetAutoAllocation(p.inv, 0);
  // In place: the real input is padded to 2*NC doubles per row so that
  // the NC complex outputs fit over it.
  if (cufftMakePlanMany(p.fwd, 1, &n, &rin, 1, rin, &cin, 1, cin, CUFFT_D2Z,
                        (int)batch, &wf) != CUFFT_SUCCESS ||
      cufftMakePlanMany(p.inv, 1, &n, &cin, 1, cin, &rin, 1, rin, CUFFT_Z2D,
                        (int)batch, &wi) != CUFFT_SUCCESS) {
    fail("cufftMakePlanMany failed");
    return NULL;
  }
  cufftSetStream(p.fwd, s_);
  cufftSetStream(p.inv, s_);
  plans_[key] = p;
  const size_t need = wf > wi ? wf : wi;
  if (need > work_size_) {
    cudaStreamSynchronize(s_);
    if (d_work_)
      cudaFree(d_work_);
    d_work_ = NULL;
    if (cudaMalloc(&d_work_, need) != cudaSuccess) {
      fail("cudaMalloc (cuFFT work area) failed");
      return NULL;
    }
    work_size_ = need;
    for (auto &kv : plans_) {
      cufftSetWorkArea(kv.second.fwd, d_work_);
      cufftSetWorkArea(kv.second.inv, d_work_);
    }
  } else {
    cufftSetWorkArea(p.fwd, d_work_);
    cufftSetWorkArea(p.inv, d_work_);
  }
  return &plans_[key];
}

bool FftMod::prepare(unsigned k) {
  for (unsigned i = 1; i <= 2 * k && i <= 2 * maxk_; i++)
    if (!plan(i * C_) || (i <= maxk_ && !plan_n(L_, i * C_)))
      return false;
  return cuda_ok_;
}

slot_t FftMod::slot() {
  void *p = NULL;
  if (cudaMalloc(&p, 4 * (size_t)L_ * C_) != cudaSuccess) {
    fail("cudaMalloc (slot) failed");
    return NULL;
  }
  cudaMemsetAsync(p, 0, 4 * (size_t)L_ * C_, s_);
  slots_.push_back(p);
  return (slot_t)p;
}

xslot_t FftMod::xslot() {
  void *p = NULL;
  if (cudaMalloc(&p, sizeof(cufftDoubleComplex) * C_ * NC_) != cudaSuccess) {
    fail("cudaMalloc (transform slot) failed");
    return NULL;
  }
  slots_.push_back(p);
  return (xslot_t)p;
}

void FftMod::xform(unsigned k, const xslot_t *dst, const slot_t *src) {
  const size_t row = (size_t)C_ * NC_;
  for (unsigned i0 = 0; i0 < k && cuda_ok_; i0 += 2 * maxk_) {
    const unsigned kc = k - i0 < 2 * maxk_ ? k - i0 : 2 * maxk_;
    PtrK ps;
    for (unsigned i = 0; i < kc; i++)
      ps.p[i] = src[i0 + i];
    Plan *p = plan(kc * C_);
    if (!p)
      return;
    k_pack<<<kc * C_, 256, 0, s_>>>((double *)d_A_, ps, L_, L_, C_, 2 * NC_,
                                    N_);
    if (cufftExecD2Z(p->fwd, (double *)d_A_, d_A_) != CUFFT_SUCCESS)
      fail("cufftExecD2Z failed");
    for (unsigned i = 0; i < kc; i++)
      cudaMemcpyAsync(dst[i0 + i], d_A_ + i * row,
                      sizeof(cufftDoubleComplex) * row,
                      cudaMemcpyDeviceToDevice, s_);
  }
}

void FftMod::chunk_mul(unsigned k, const slot_t *dst, const Opnd *a,
                       const Opnd *b) {
  const size_t row = (size_t)C_ * NC_;
  // Distinct slot operands get one transform each, in A.
  PtrK pu;
  PtrC xa, xb;
  unsigned nu = 0;
  auto map = [&](const Opnd &o) -> const cufftDoubleComplex * {
    if (o.x)
      return o.x;
    for (unsigned u = 0; u < nu; u++)
      if (pu.p[u] == o.s)
        return d_A_ + u * row;
    pu.p[nu] = o.s;
    return d_A_ + (nu++) * row;
  };
  for (unsigned i = 0; i < k; i++) {
    xa.p[i] = map(a[i]);
    xb.p[i] = map(b[i]);
  }
  Plan *pk = plan(k * C_), *pn = nu ? plan(nu * C_) : NULL;
  if (!pk || (nu && !pn))
    return;
  double *Ar = (double *)d_A_, *Br = (double *)d_B_;
  const unsigned rs = 2 * NC_, nrm2 = norm_threads(2 * L_);
  const double scale = 1.0 / N_;
  bool ok = true;

  // T = a*b
  if (nu) {
    k_pack<<<nu * C_, 256, 0, s_>>>(Ar, pu, L_, L_, C_, rs, N_);
    ok = cufftExecD2Z(pn->fwd, Ar, d_A_) == CUFFT_SUCCESS;
  }
  k_pmul<<<grid_for(k * row), 256, 0, s_>>>(d_B_, xa, xb, k, row);
  ok = ok && cufftExecZ2D(pk->inv, d_B_, Br) == CUFFT_SUCCESS;
  k_carry<0><<<k * C_, nrm2, 0, s_>>>(Br, rs, scale, 2 * L_, d_T_, 2 * L_,
                                      b_, d_err_, d_ovf_);
  if (!ok)
    fail("cuFFT execution failed");
  tail(k, dst);
}

// dst[i] = REDC(T_i) for the k*C products T in d_T_ (row i*C + c).
void FftMod::tail(unsigned k, const slot_t *dst) {
  const size_t row = (size_t)C_ * NC_;
  PtrK pt, pm;
  PtrKW pd;
  for (unsigned i = 0; i < k; i++) {
    pd.p[i] = dst[i];
    pt.p[i] = d_T_ + (size_t)i * C_ * 2 * L_;
    pm.p[i] = d_m_ + (size_t)i * C_ * L_;
  }
  Plan *pk = plan(k * C_);
  if (!pk)
    return;
  double *Br = (double *)d_B_;
  const unsigned rs = 2 * NC_, nrm1 = norm_threads(L_);
  const double scale = 1.0 / N_;
  bool ok = true;
  // m = (T mod R) * n' mod R
  k_pack<<<k * C_, 256, 0, s_>>>(Br, pt, 2 * L_, L_, C_, rs, N_);
  ok = ok && cufftExecD2Z(pk->fwd, Br, d_B_) == CUFFT_SUCCESS;
  k_pconst<<<grid_for(k * row), 256, 0, s_>>>(d_B_, d_Fnp_, NC_, k * row);
  ok = ok && cufftExecZ2D(pk->inv, d_B_, Br) == CUFFT_SUCCESS;
  k_carry<1><<<k * C_, nrm1, 0, s_>>>(Br, rs, scale, L_, d_m_, L_, b_,
                                      d_err_, d_ovf_);
  // dst = (T + m*n) / R, from m*n mod (R - 1): length L, cyclic.
  Plan *pl = plan_n(L_, k * C_);
  if (!pl)
    return;
  const unsigned rsl = 2 * LC_;
  const size_t rowl = (size_t)C_ * LC_;
  k_pack<<<k * C_, 256, 0, s_>>>(Br, pm, L_, L_, C_, rsl, L_);
  ok = ok && cufftExecD2Z(pl->fwd, Br, d_B_) == CUFFT_SUCCESS;
  k_pconst<<<grid_for(k * rowl), 256, 0, s_>>>(d_B_, d_Fn_, LC_, k * rowl);
  ok = ok && cufftExecZ2D(pl->inv, d_B_, Br) == CUFFT_SUCCESS;
  k_redc<<<k * C_, nrm1, 0, s_>>>(Br, rsl, 1.0 / L_, d_T_, pd, C_, L_, b_,
                                  hd_, d_err_, d_ovf_);
  if (!ok)
    fail("cuFFT execution failed");
}

bool FftMod::kron_buffers(size_t needA, size_t needB) {
  for (int w = 0; w < 2; w++) {
    cufftDoubleComplex *&p = w ? d_KB_ : d_KA_;
    size_t &cap = w ? kronB_cap_ : kronA_cap_;
    const size_t need = w ? needB : needA;
    if (need <= cap)
      continue;
    sync();
    if (!w)
      kreg_NK_[0] = kreg_NK_[1] = 0; // kprod() products are lost
    if (p)
      cudaFree(p);
    p = NULL;
    cap = 0;
    if (cudaMalloc(&p, sizeof(cufftDoubleComplex) * need) != cudaSuccess) {
      fail("cudaMalloc (polynomial product buffers) failed: not enough "
           "memory");
      return false;
    }
    cap = need;
  }
  return true;
}

namespace {
// Kronecker length for a product of na- and nb-coefficient polynomials,
// coefficients S digits apart.
unsigned kron_len(unsigned na, unsigned nb, unsigned S) {
  unsigned NK = (na + nb - 1) * S;
  NK += NK & 1;
  while (!smooth7(NK))
    NK += 2;
  return NK;
}
} // namespace

void FftMod::polymul(unsigned na, const slot_t *A, unsigned nb,
                     const slot_t *B, unsigned lo, unsigned nd,
                     const slot_t *out) {
  polymul(std::vector<PolyJob>{PolyJob{na, A, nb, B, lo, nd, out}});
}

void FftMod::polymul(const std::vector<PolyJob> &jobs) {
  if (!cuda_ok_)
    return;
  kreg_NK_[0] = 0; // region 0 is about to be overwritten
  const unsigned S = 2 * L_;
  // Jobs with nothing to compute go now; an empty operand's product is
  // zero.
  std::vector<PolyJob> todo;
  for (const PolyJob &j : jobs) {
    if (!j.nd)
      continue;
    if (!j.na || !j.nb) {
      for (unsigned t = 0; t < j.nd; t++)
        cudaMemsetAsync(j.out[t], 0, 4 * (size_t)L_ * C_, s_);
      continue;
    }
    // A product coefficient sums min(na, nb) terms; init() sized the
    // digits and R for poly_len of them.
    if ((j.na < j.nb ? j.na : j.nb) > poly_len_) {
      fail("polymul: operands longer than init()'s poly_len");
      return;
    }
    todo.push_back(j);
  }
  // Groups of consecutive jobs, each one batched transform of the
  // group's longest length, as many as init()'s Kronecker budget holds
  // (a single job may exceed it; kron_buffers() then grows).
  for (size_t g0 = 0; g0 < todo.size() && cuda_ok_;) {
    unsigned NK = gmpecmfft_detail::kron_len(todo[g0].na, todo[g0].nb, S);
    size_t g1 = g0 + 1;
    for (; g1 < todo.size(); g1++) {
      unsigned nk = gmpecmfft_detail::kron_len(todo[g1].na, todo[g1].nb, S);
      if (nk < NK)
        nk = NK;
      if ((g1 - g0 + 1) * ((size_t)nk / 2 + 1) > kron_budget_)
        break;
      NK = nk;
    }
    polymul_group(todo.data() + g0, (unsigned)(g1 - g0), NK);
    g0 = g1;
  }
}

unsigned FftMod::kron_len(unsigned na, unsigned nb) const {
  return gmpecmfft_detail::kron_len(na, nb, 2 * L_);
}

bool FftMod::kron_transform(const PolyJob *jobs, unsigned m, unsigned NK,
                            cufftDoubleComplex *XA) {
  const unsigned S = 2 * L_, NKC = NK / 2 + 1;
  const size_t rstride = 2 * (size_t)NKC;
  const unsigned rows = m * C_;
  bool square = true;
  for (unsigned j = 0; j < m; j++)
    square = square && jobs[j].A == jobs[j].B && jobs[j].na == jobs[j].nb;
  Plan *pk = plan_n(NK, rows);
  if (!pk)
    return false;

  // Coefficient pointer and job tables (stream ordered, so reusing the
  // buffers after an earlier call's kernels is safe; the copies are from
  // pageable memory, so the host vectors may go when they return).
  std::vector<const int32_t *> tab;
  std::vector<uint2> jt(2 * (size_t)m);
  for (unsigned j = 0; j < m; j++) {
    jt[j] = make_uint2((unsigned)tab.size(), jobs[j].na);
    tab.insert(tab.end(), jobs[j].A, jobs[j].A + jobs[j].na);
  }
  if (!square)
    for (unsigned j = 0; j < m; j++) {
      jt[m + j] = make_uint2((unsigned)tab.size(), jobs[j].nb);
      tab.insert(tab.end(), jobs[j].B, jobs[j].B + jobs[j].nb);
    }
  if (tab.size() > ptrs_cap_ || jt.size() > jobs_cap_) {
    sync();
    if (tab.size() > ptrs_cap_) {
      if (d_ptrs_)
        cudaFree((void *)d_ptrs_);
      d_ptrs_ = NULL;
      ptrs_cap_ = 0;
      if (cudaMalloc((void **)&d_ptrs_, sizeof(int32_t *) * tab.size()) !=
          cudaSuccess) {
        fail("cudaMalloc (pointer table) failed");
        return false;
      }
      ptrs_cap_ = tab.size();
    }
    if (jt.size() > jobs_cap_) {
      if (d_jobs_)
        cudaFree(d_jobs_);
      d_jobs_ = NULL;
      jobs_cap_ = 0;
      if (cudaMalloc((void **)&d_jobs_, sizeof(uint2) * jt.size()) !=
          cudaSuccess) {
        fail("cudaMalloc (job table) failed");
        return false;
      }
      jobs_cap_ = jt.size();
    }
  }
  cudaMemcpyAsync((void *)d_ptrs_, tab.data(),
                  sizeof(int32_t *) * tab.size(), cudaMemcpyHostToDevice, s_);
  cudaMemcpyAsync(d_jobs_, jt.data(), sizeof(uint2) * jt.size(),
                  cudaMemcpyHostToDevice, s_);

  const unsigned gx = (NK + 255) / 256 < 4096 ? (NK + 255) / 256 : 4096;
  const dim3 g(gx, rows < 65535 ? rows : 65535);
  double *KA = (double *)XA, *KB = (double *)d_KB_;
  k_kpack<<<g, 256, 0, s_>>>(KA, d_ptrs_, d_jobs_, rows, C_, L_, S, rstride,
                             NK);
  bool ok = cufftExecD2Z(pk->fwd, KA, XA) == CUFFT_SUCCESS;
  if (!square) {
    k_kpack<<<g, 256, 0, s_>>>(KB, d_ptrs_, d_jobs_ + m, rows, C_, L_, S,
                               rstride, NK);
    ok = ok && cufftExecD2Z(pk->fwd, KB, d_KB_) == CUFFT_SUCCESS;
  }
  const size_t total = (size_t)rows * NKC;
  k_kmul<<<grid_for(total), 256, 0, s_>>>(XA, square ? XA : d_KB_, total);
  ok = ok && cufftExecZ2D(pk->inv, XA, KA) == CUFFT_SUCCESS;
  if (!ok) {
    fail("cuFFT execution failed (polynomial product)");
    return false;
  }
  return true;
}

void FftMod::kron_out(const double *r0, const double *r1, unsigned NK,
                      unsigned cnt, const unsigned *row0,
                      const unsigned *coef, const slot_t *out) {
  const unsigned S = 2 * L_, NKC = NK / 2 + 1, nrm2 = norm_threads(2 * L_);
  const size_t rstride = 2 * (size_t)NKC;
  for (unsigned i0 = 0; i0 < cnt && cuda_ok_; i0 += maxk_) {
    const unsigned k = cnt - i0 < maxk_ ? cnt - i0 : maxk_;
    KRows kr;
    for (unsigned i = 0; i < k; i++) {
      kr.row0[i] = row0[i0 + i];
      kr.coef[i] = coef[i0 + i];
    }
    k_kcarry<<<k * C_, nrm2, 0, s_>>>(r0, r1, rstride, S, kr, NK,
                                      1.0 / NK, C_, L_, b_, d_T_, d_err_,
                                      d_ovf_);
    tail(k, out + i0);
  }
}

void FftMod::polymul_group(const PolyJob *jobs, unsigned m, unsigned NK) {
  const unsigned NKC = NK / 2 + 1;
  const size_t need = (size_t)m * C_ * NKC;
  if (need > (size_t)C_ * kron_budget_)
    kreg_NK_[1] = 0; // a single oversized job reaches into region 1
  if (!kron_buffers(need, need) || !kron_transform(jobs, m, NK, d_KA_))
    return;
  // Every job's output coefficients, maxk to a REDC round.
  std::vector<unsigned> row0, coef;
  std::vector<slot_t> out;
  for (unsigned j = 0; j < m; j++)
    for (unsigned t = 0; t < jobs[j].nd; t++) {
      row0.push_back(j * C_);
      coef.push_back(jobs[j].lo + t);
      out.push_back(jobs[j].out[t]);
    }
  kron_out((const double *)d_KA_, NULL, NK, (unsigned)out.size(),
           row0.data(), coef.data(), out.data());
}

void FftMod::kprod(unsigned reg, unsigned na, const slot_t *A, unsigned nb,
                   const slot_t *B, unsigned NK) {
  if (!cuda_ok_)
    return;
  if (!NK)
    NK = kron_len(na, nb);
  if (reg > 1 || !na || !nb || (na < nb ? na : nb) > poly_len_ ||
      NK < kron_len(na, nb) || (size_t)NK / 2 + 1 > kron_budget_) {
    fail("kprod: bad region or operands");
    return;
  }
  const size_t unit = (size_t)C_ * kron_budget_;
  if (!kron_buffers(2 * unit, (size_t)C_ * (NK / 2 + 1)))
    return;
  PolyJob j{na, A, nb, B, 0, 0, NULL};
  if (kron_transform(&j, 1, NK, d_KA_ + reg * unit))
    kreg_NK_[reg] = NK;
}

void FftMod::kout(unsigned r0, int r1, unsigned lo, unsigned nd,
                  const slot_t *out) {
  if (!cuda_ok_ || !nd)
    return;
  if (r0 > 1 || r1 > 1 || !kreg_NK_[r0] ||
      (r1 >= 0 && kreg_NK_[r1] != kreg_NK_[r0])) {
    fail("kout: no product in that region, or lengths differ");
    return;
  }
  const size_t unit = (size_t)C_ * kron_budget_;
  const double *b0 = (const double *)(d_KA_ + r0 * unit),
               *b1 = r1 >= 0 ? (const double *)(d_KA_ + r1 * unit) : NULL;
  std::vector<unsigned> row0(nd, 0), coef(nd);
  for (unsigned t = 0; t < nd; t++)
    coef[t] = lo + t;
  kron_out(b0, b1, kreg_NK_[r0], nd, row0.data(), coef.data(), out);
}

void FftMod::mul(unsigned k, const slot_t *dst, const Opnd *a,
                 const Opnd *b) {
  for (unsigned i = 0; i < k && cuda_ok_; i += maxk_)
    chunk_mul(k - i < maxk_ ? k - i : maxk_, dst + i, a + i, b + i);
}

void FftMod::mul(unsigned k, const slot_t *dst, const slot_t *a,
                 const slot_t *b) {
  Opnd oa[MAXK], ob[MAXK];
  for (unsigned i0 = 0; i0 < k; i0 += MAXK) {
    const unsigned kc = k - i0 < MAXK ? k - i0 : MAXK;
    for (unsigned i = 0; i < kc; i++) {
      oa[i] = S(a[i0 + i]);
      ob[i] = S(b[i0 + i]);
    }
    mul(kc, dst + i0, oa, ob);
  }
}

void FftMod::add(unsigned k, const slot_t *dst, const slot_t *a,
                 const slot_t *b, const int *neg) {
  for (unsigned i0 = 0; i0 < k; i0 += MAXK) {
    const unsigned kc = k - i0 < MAXK ? k - i0 : MAXK;
    PtrKW pd;
    PtrK pa, pb;
    unsigned long long ng = 0;
    for (unsigned i = 0; i < kc; i++) {
      pd.p[i] = dst[i0 + i];
      pa.p[i] = a[i0 + i];
      pb.p[i] = b[i0 + i];
      if (neg[i0 + i])
        ng |= 1ull << i;
    }
    k_add<<<kc * C_, norm_threads(L_), 0, s_>>>(pd, pa, pb, ng, C_,
                                                L_, b_, d_ovf_);
  }
}

void FftMod::copy(slot_t dst, slot_t src) {
  cudaMemcpyAsync(dst, src, 4 * (size_t)L_ * C_, cudaMemcpyDeviceToDevice,
                  s_);
}

void FftMod::put(slot_t s, const std::vector<mpz_ptr> &v) {
  h_buf_.resize((size_t)L_ * C_);
  for (unsigned c = 0; c < C_; c++)
    mpz_to_digits(h_buf_.data() + (size_t)c * L_, L_, b_, v[c]);
  cudaMemcpyAsync(s, h_buf_.data(), 4 * (size_t)L_ * C_,
                  cudaMemcpyHostToDevice, s_);
  sync();
}

void FftMod::put_all(slot_t s, mpz_srcptr x) {
  h_buf_.resize((size_t)L_ * C_);
  mpz_to_digits(h_buf_.data(), L_, b_, x);
  for (unsigned c = 1; c < C_; c++)
    memcpy(h_buf_.data() + (size_t)c * L_, h_buf_.data(), 4 * (size_t)L_);
  cudaMemcpyAsync(s, h_buf_.data(), 4 * (size_t)L_ * C_,
                  cudaMemcpyHostToDevice, s_);
  sync();
}

void FftMod::get(std::vector<mpz_ptr> &v, slot_t s) {
  h_buf_.resize((size_t)L_ * C_);
  cudaMemcpyAsync(h_buf_.data(), s, 4 * (size_t)L_ * C_,
                  cudaMemcpyDeviceToHost, s_);
  if (!sync())
    return;
  for (unsigned c = 0; c < C_; c++)
    digits_to_mpz(v[c], h_buf_.data() + (size_t)c * L_, L_, b_);
}

void FftMod::to_mont(mpz_ptr out, mpz_srcptr x) const {
  mpz_mul_2exp(out, x, (mp_bitcnt_t)b_ * L_);
  mpz_mod(out, out, n_);
}

bool FftMod::sync() {
  if (cudaStreamSynchronize(s_) != cudaSuccess)
    fail("CUDA error");
  return cuda_ok_;
}

double FftMod::max_err() {
  unsigned u = 0;
  cudaMemcpyAsync(&u, d_err_, sizeof u, cudaMemcpyDeviceToHost, s_);
  cudaMemsetAsync(d_err_, 0, sizeof(unsigned), s_);
  sync();
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}

bool FftMod::overflowed() {
  unsigned u = 0;
  cudaMemcpyAsync(&u, d_ovf_, sizeof u, cudaMemcpyDeviceToHost, s_);
  sync();
  return u != 0;
}

} // namespace gmpecmfft_detail
