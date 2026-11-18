/*
 * ecm_cuda.cu -- CUDA/CGBN backend for the ECM factoring stage.
 *
 * Lenstra's elliptic curve method is the one part of this prover that is
 * embarrassingly parallel: every curve is an independent trial, and only
 * one of them has to succeed.  That maps directly onto CGBN, which runs
 * many fixed-width big integer instances at once, each spread across TPI
 * cooperating threads.
 *
 * Stage 1 only.  Curves are Montgomery form B*y^2 = x^3 + A*x^2 + x with
 * Suyama's parametrisation, worked in XZ coordinates so the inner loop
 * needs no modular inversion -- the whole point, since an inversion per
 * step would dominate.  All arithmetic is in Montgomery representation.
 *
 * Copyright (c) 2005 Paul Zimmermann and contributors (the CPU ECM this
 * mirrors); GPL-3.0-or-later, as the rest of the program.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gmp.h>

#include <cgbn/cgbn.h>

namespace {

// ---------------------------------------------------------------- //
// Device helpers                                                     //
// ---------------------------------------------------------------- //

// CGBN's add/sub are plain multiprecision ops returning carry/borrow;
// these wrap them into arithmetic modulo n.
template <class env_t>
__device__ __forceinline__ void
mod_add(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a,
        const typename env_t::cgbn_t &b, const typename env_t::cgbn_t &n) {
  int32_t carry = cgbn_add(env, r, a, b);
  if (carry || cgbn_compare(env, r, n) >= 0)
    cgbn_sub(env, r, r, n);
}

template <class env_t>
__device__ __forceinline__ void
mod_sub(env_t env, typename env_t::cgbn_t &r, const typename env_t::cgbn_t &a,
        const typename env_t::cgbn_t &b, const typename env_t::cgbn_t &n) {
  if (cgbn_sub(env, r, a, b))
    cgbn_add(env, r, r, n);
}

/* Montgomery curve doubling in XZ coordinates.
 *   a = X+Z, b = X-Z, c = a^2 - b^2
 *   X2 = a^2 * b^2
 *   Z2 = c * (b^2 + a24*c)
 * Output must not alias the input. */
template <class env_t>
__device__ __forceinline__ void
xz_double(env_t env, typename env_t::cgbn_t &X2, typename env_t::cgbn_t &Z2,
          const typename env_t::cgbn_t &X, const typename env_t::cgbn_t &Z,
          const typename env_t::cgbn_t &a24, const typename env_t::cgbn_t &n,
          uint32_t np0) {
  typename env_t::cgbn_t a, b, aa, bb, c, t;
  mod_add(env, a, X, Z, n);
  mod_sub(env, b, X, Z, n);
  cgbn_mont_sqr(env, aa, a, n, np0);
  cgbn_mont_sqr(env, bb, b, n, np0);
  mod_sub(env, c, aa, bb, n);
  cgbn_mont_mul(env, X2, aa, bb, n, np0);
  cgbn_mont_mul(env, t, a24, c, n, np0);
  mod_add(env, t, t, bb, n);
  cgbn_mont_mul(env, Z2, c, t, n, np0);
}

/* Differential addition: given P1, P2 and their difference D = P1 - P2,
 *   X3 = Zd * ((X1-Z1)(X2+Z2) + (X1+Z1)(X2-Z2))^2
 *   Z3 = Xd * ((X1-Z1)(X2+Z2) - (X1+Z1)(X2-Z2))^2
 * Output must not alias the inputs. */
template <class env_t>
__device__ __forceinline__ void
xz_add(env_t env, typename env_t::cgbn_t &X3, typename env_t::cgbn_t &Z3,
       const typename env_t::cgbn_t &X1, const typename env_t::cgbn_t &Z1,
       const typename env_t::cgbn_t &X2, const typename env_t::cgbn_t &Z2,
       const typename env_t::cgbn_t &Xd, const typename env_t::cgbn_t &Zd,
       const typename env_t::cgbn_t &n, uint32_t np0) {
  typename env_t::cgbn_t p, q, t1, t2;
  mod_sub(env, p, X1, Z1, n);
  mod_add(env, q, X2, Z2, n);
  cgbn_mont_mul(env, t1, p, q, n, np0);
  mod_add(env, p, X1, Z1, n);
  mod_sub(env, q, X2, Z2, n);
  cgbn_mont_mul(env, t2, p, q, n, np0);
  mod_add(env, p, t1, t2, n);
  mod_sub(env, q, t1, t2, n);
  cgbn_mont_sqr(env, p, p, n, np0);
  cgbn_mont_sqr(env, q, q, n, np0);
  cgbn_mont_mul(env, X3, Zd, p, n, np0);
  cgbn_mont_mul(env, Z3, Xd, q, n, np0);
}

/* Montgomery ladder: (X,Z) <- [k](X,Z), k a small scalar.
 * Maintains R1 - R0 = P so the differential addition always has P as its
 * difference. */
template <class env_t>
__device__ void xz_mul_ui(env_t env, typename env_t::cgbn_t &X,
                          typename env_t::cgbn_t &Z,
                          const typename env_t::cgbn_t &a24,
                          const typename env_t::cgbn_t &n, uint32_t np0,
                          uint32_t k) {
  if (k == 0) {
    cgbn_set_ui32(env, X, 0);
    cgbn_set_ui32(env, Z, 0);
    return;
  }
  if (k == 1)
    return;

  typename env_t::cgbn_t Xp, Zp, X0, Z0, X1, Z1, tX, tZ;
  cgbn_set(env, Xp, X);
  cgbn_set(env, Zp, Z);
  cgbn_set(env, X0, X);
  cgbn_set(env, Z0, Z);
  xz_double(env, X1, Z1, X, Z, a24, n, np0);

  int top = 31;
  while (top > 0 && ((k >> top) & 1u) == 0)
    top--;

  for (int i = top - 1; i >= 0; i--) {
    if ((k >> i) & 1u) {
      xz_add(env, tX, tZ, X0, Z0, X1, Z1, Xp, Zp, n, np0);
      cgbn_set(env, X0, tX);
      cgbn_set(env, Z0, tZ);
      xz_double(env, tX, tZ, X1, Z1, a24, n, np0);
      cgbn_set(env, X1, tX);
      cgbn_set(env, Z1, tZ);
    } else {
      xz_add(env, tX, tZ, X0, Z0, X1, Z1, Xp, Zp, n, np0);
      cgbn_set(env, X1, tX);
      cgbn_set(env, Z1, tZ);
      xz_double(env, tX, tZ, X0, Z0, a24, n, np0);
      cgbn_set(env, X0, tX);
      cgbn_set(env, Z0, tZ);
    }
  }
  cgbn_set(env, X, X0);
  cgbn_set(env, Z, Z0);
}

// ---------------------------------------------------------------- //
// Kernel                                                             //
// ---------------------------------------------------------------- //

template <uint32_t TPI, uint32_t BITS>
__global__ void ecm_stage1_kernel(cgbn_error_report_t *report,
                                  cgbn_mem_t<BITS> *n_in,
                                  const uint32_t *sigmas, uint32_t ncurves,
                                  const uint32_t *powers, uint32_t npowers,
                                  cgbn_mem_t<BITS> *out) {
  typedef cgbn_context_t<TPI> context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (idx >= ncurves)
    return;

  context_t ctx(cgbn_report_monitor, report, idx);
  env_t env(ctx);

  bn_t n, one, sig, u, v, t, w, X, Z, a24, num, den, inv;
  cgbn_load(env, n, n_in);

  cgbn_set_ui32(env, one, 1);
  uint32_t np0 = cgbn_bn2mont(env, one, one, n); // one is now R mod n

  // sigma, in Montgomery form
  cgbn_set_ui32(env, sig, sigmas[idx]);
  cgbn_bn2mont(env, sig, sig, n);

  // u = sigma^2 - 5,  v = 4*sigma
  cgbn_mont_sqr(env, u, sig, n, np0);
  cgbn_set_ui32(env, t, 5);
  cgbn_bn2mont(env, t, t, n);
  mod_sub(env, u, u, t, n);
  mod_add(env, v, sig, sig, n);
  mod_add(env, v, v, v, n);

  // X = u^3, Z = v^3
  cgbn_mont_sqr(env, t, u, n, np0);
  cgbn_mont_mul(env, X, t, u, n, np0);
  cgbn_mont_sqr(env, w, v, n, np0);
  cgbn_mont_mul(env, Z, w, v, n, np0);

  // num = (v-u)^3 * (3u+v)
  mod_sub(env, t, v, u, n);
  cgbn_mont_sqr(env, w, t, n, np0);
  cgbn_mont_mul(env, num, w, t, n, np0);
  mod_add(env, t, u, u, n);
  mod_add(env, t, t, u, n);
  mod_add(env, t, t, v, n);
  cgbn_mont_mul(env, num, num, t, n, np0);

  // den = 16 * u^3 * v
  cgbn_mont_mul(env, den, X, v, n, np0);
  for (int i = 0; i < 4; i++)
    mod_add(env, den, den, den, n);

  // a24 = num / den.  A failed inversion means gcd(den, n) is a factor,
  // which is exactly what we are looking for.
  cgbn_mont2bn(env, t, den, n, np0);
  if (!cgbn_modular_inverse(env, inv, t, n)) {
    cgbn_gcd(env, w, t, n);
    cgbn_store(env, &out[idx], w);
    return;
  }
  cgbn_bn2mont(env, inv, inv, n);
  cgbn_mont_mul(env, a24, num, inv, n, np0);

  // Stage 1: multiply through every prime power below the bound.
  for (uint32_t i = 0; i < npowers; i++)
    xz_mul_ui(env, X, Z, a24, n, np0, powers[i]);

  // A nontrivial gcd(Z, n) is a factor.
  cgbn_mont2bn(env, t, Z, n, np0);
  cgbn_gcd(env, w, t, n);
  cgbn_store(env, &out[idx], w);
}

// ---------------------------------------------------------------- //
// Host side                                                          //
// ---------------------------------------------------------------- //

struct DeviceInfo {
  bool probed;
  bool usable;
  char name[256];
};

DeviceInfo &device_info() {
  static DeviceInfo info = {false, false, {0}};
  static std::once_flag once;
  std::call_once(once, [] {
    info.probed = true;
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        snprintf(info.name, sizeof info.name, "%s (sm_%d%d, %zu MiB)",
                 prop.name, prop.major, prop.minor,
                 (size_t)(prop.totalGlobalMem >> 20));
        info.usable = true;
      }
    }
  });
  return info;
}

template <uint32_t BITS>
void mpz_to_mem(cgbn_mem_t<BITS> &dst, mpz_srcptr v) {
  memset(&dst, 0, sizeof dst);
  size_t words = 0;
  mpz_export(dst._limbs, &words, -1, sizeof(uint32_t), 0, 0, v);
}

template <uint32_t BITS>
void mem_to_mpz(mpz_ptr out, const cgbn_mem_t<BITS> &src) {
  mpz_import(out, BITS / 32, -1, sizeof(uint32_t), 0, 0, src._limbs);
}

// Prime powers p^e <= b1, one entry per prime.
void build_powers(unsigned long b1, uint32_t **out, uint32_t *count) {
  mpz_t p;
  mpz_init_set_ui(p, 2);
  uint32_t cap = 1024, n = 0;
  uint32_t *v = (uint32_t *)malloc(cap * sizeof(uint32_t));
  while (mpz_cmp_ui(p, b1) <= 0) {
    unsigned long pi = mpz_get_ui(p);
    unsigned long pe = pi;
    while (pe <= b1 / pi)
      pe *= pi;
    if (n == cap) {
      cap *= 2;
      v = (uint32_t *)realloc(v, cap * sizeof(uint32_t));
    }
    v[n++] = (uint32_t)pe;
    mpz_nextprime(p, p);
  }
  mpz_clear(p);
  *out = v;
  *count = n;
}

template <uint32_t TPI, uint32_t BITS>
bool run_bits(mpz_srcptr n, unsigned long b1, unsigned int curves,
              unsigned long seed, mpz_ptr factor) {
  uint32_t *powers = NULL, npowers = 0;
  build_powers(b1, &powers, &npowers);
  if (!npowers) {
    free(powers);
    return false;
  }

  uint32_t *sigmas = (uint32_t *)malloc(curves * sizeof(uint32_t));
  gmp_randstate_t st;
  gmp_randinit_default(st);
  gmp_randseed_ui(st, seed);
  for (unsigned i = 0; i < curves; i++)
    sigmas[i] = 6 + (uint32_t)gmp_urandomm_ui(st, 1u << 30);
  gmp_randclear(st);

  cgbn_mem_t<BITS> h_n;
  mpz_to_mem<BITS>(h_n, n);

  cgbn_mem_t<BITS> *d_n = NULL, *d_out = NULL;
  uint32_t *d_sig = NULL, *d_pow = NULL;
  cgbn_error_report_t *report = NULL;
  bool ok = true;
  ok = ok && cudaMalloc(&d_n, sizeof h_n) == cudaSuccess;
  ok = ok && cudaMalloc(&d_out, sizeof(cgbn_mem_t<BITS>) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_sig, sizeof(uint32_t) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_pow, sizeof(uint32_t) * npowers) == cudaSuccess;
  ok = ok && cgbn_error_report_alloc(&report) == cudaSuccess;
  if (ok) {
    cudaMemcpy(d_n, &h_n, sizeof h_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sigmas, sizeof(uint32_t) * curves, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pow, powers, sizeof(uint32_t) * npowers, cudaMemcpyHostToDevice);

    const uint32_t tpb = 128;
    uint32_t threads = curves * TPI;
    uint32_t blocks = (threads + tpb - 1) / tpb;
    ecm_stage1_kernel<TPI, BITS><<<blocks, tpb>>>(report, d_n, d_sig, curves,
                                                  d_pow, npowers, d_out);
    ok = cudaDeviceSynchronize() == cudaSuccess;
    if (ok && cgbn_error_report_check(report))
      ok = false;
  }

  bool found = false;
  if (ok) {
    cgbn_mem_t<BITS> *h_out =
        (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>) * curves);
    cudaMemcpy(h_out, d_out, sizeof(cgbn_mem_t<BITS>) * curves,
               cudaMemcpyDeviceToHost);
    mpz_t g;
    mpz_init(g);
    for (unsigned i = 0; i < curves && !found; i++) {
      mem_to_mpz<BITS>(g, h_out[i]);
      if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, n) < 0) {
        mpz_set(factor, g);
        found = true;
      }
    }
    mpz_clear(g);
    free(h_out);
  }

  if (report) cgbn_error_report_free(report);
  if (d_n) cudaFree(d_n);
  if (d_out) cudaFree(d_out);
  if (d_sig) cudaFree(d_sig);
  if (d_pow) cudaFree(d_pow);
  free(sigmas);
  free(powers);
  return found;
}


// ---------------------------------------------------------------- //
// Resumable ECM (gmpecpp_cuda_ecm)                                   //
// ---------------------------------------------------------------- //
//
// The same curves and arithmetic as ecm_stage1_kernel, but each curve's
// state (X, Z, a24 in Montgomery form) lives in device memory between
// launches, so the work is cut into short launches:
//
//   k_init     sigma -> curve and starting point (Suyama)
//   k_stage1   [k]P for a slice of the prime powers <= B1
//   k_gcd      gcd(Z, n) of every curve
//
// Short launches let the caller stop a search between them (a GPU kernel
// cannot be interrupted), and remove any cap on B1: a long stage 1 is just
// more launches. Stage 2 continues from the saved points (see below).
//
// status[i] = 1 once curve i has a result in out[i] (a failed inversion in
// k_init already exposes a factor), so later kernels skip it.

template <uint32_t TPI, uint32_t BITS>
__global__ void k_init(cgbn_error_report_t *report, cgbn_mem_t<BITS> *n_in,
                       const uint32_t *sigmas, uint32_t ncurves,
                       cgbn_mem_t<BITS> *gX, cgbn_mem_t<BITS> *gZ,
                       cgbn_mem_t<BITS> *gA, uint32_t *status,
                       cgbn_mem_t<BITS> *out) {
  typedef cgbn_context_t<TPI> context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (idx >= ncurves)
    return;
  context_t ctx(cgbn_report_monitor, report, idx);
  env_t env(ctx);

  bn_t n, one, sig, u, v, t, w, X, Z, a24, num, den, inv;
  cgbn_load(env, n, n_in);
  cgbn_set_ui32(env, one, 1);
  uint32_t np0 = cgbn_bn2mont(env, one, one, n);

  cgbn_set_ui32(env, sig, sigmas[idx]);
  cgbn_bn2mont(env, sig, sig, n);
  cgbn_mont_sqr(env, u, sig, n, np0);
  cgbn_set_ui32(env, t, 5);
  cgbn_bn2mont(env, t, t, n);
  mod_sub(env, u, u, t, n);
  mod_add(env, v, sig, sig, n);
  mod_add(env, v, v, v, n);
  cgbn_mont_sqr(env, t, u, n, np0);
  cgbn_mont_mul(env, X, t, u, n, np0);
  cgbn_mont_sqr(env, w, v, n, np0);
  cgbn_mont_mul(env, Z, w, v, n, np0);
  mod_sub(env, t, v, u, n);
  cgbn_mont_sqr(env, w, t, n, np0);
  cgbn_mont_mul(env, num, w, t, n, np0);
  mod_add(env, t, u, u, n);
  mod_add(env, t, t, u, n);
  mod_add(env, t, t, v, n);
  cgbn_mont_mul(env, num, num, t, n, np0);
  cgbn_mont_mul(env, den, X, v, n, np0);
  for (int i = 0; i < 4; i++)
    mod_add(env, den, den, den, n);

  cgbn_mont2bn(env, t, den, n, np0);
  const bool lead = (threadIdx.x % TPI) == 0;
  if (!cgbn_modular_inverse(env, inv, t, n)) {
    cgbn_gcd(env, w, t, n);
    cgbn_store(env, &out[idx], w);
    if (lead) status[idx] = 1;
    return;
  }
  cgbn_bn2mont(env, inv, inv, n);
  cgbn_mont_mul(env, a24, num, inv, n, np0);
  cgbn_store(env, &gX[idx], X);
  cgbn_store(env, &gZ[idx], Z);
  cgbn_store(env, &gA[idx], a24);
  if (lead) status[idx] = 0;
}

template <uint32_t TPI, uint32_t BITS>
__global__ void k_stage1(cgbn_error_report_t *report, cgbn_mem_t<BITS> *n_in,
                         uint32_t ncurves, const uint32_t *powers,
                         uint32_t npowers, cgbn_mem_t<BITS> *gX,
                         cgbn_mem_t<BITS> *gZ, const cgbn_mem_t<BITS> *gA,
                         const uint32_t *status) {
  typedef cgbn_context_t<TPI> context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (idx >= ncurves || status[idx])
    return;
  context_t ctx(cgbn_report_monitor, report, idx);
  env_t env(ctx);

  bn_t n, t, X, Z, a24;
  cgbn_load(env, n, n_in);
  cgbn_set_ui32(env, t, 1);
  uint32_t np0 = cgbn_bn2mont(env, t, t, n);
  cgbn_load(env, X, &gX[idx]);
  cgbn_load(env, Z, &gZ[idx]);
  cgbn_load(env, a24, (cgbn_mem_t<BITS> *)&gA[idx]);
  for (uint32_t i = 0; i < npowers; i++)
    xz_mul_ui(env, X, Z, a24, n, np0, powers[i]);
  cgbn_store(env, &gX[idx], X);
  cgbn_store(env, &gZ[idx], Z);
}

// out[i] = gcd(V[i], n) for every curve still running (V in Montgomery form).
template <uint32_t TPI, uint32_t BITS>
__global__ void k_gcd(cgbn_error_report_t *report, cgbn_mem_t<BITS> *n_in,
                      uint32_t ncurves, const cgbn_mem_t<BITS> *gV,
                      const uint32_t *status, cgbn_mem_t<BITS> *out) {
  typedef cgbn_context_t<TPI> context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (idx >= ncurves || status[idx])
    return;
  context_t ctx(cgbn_report_monitor, report, idx);
  env_t env(ctx);

  bn_t n, t, v, g;
  cgbn_load(env, n, n_in);
  cgbn_set_ui32(env, t, 1);
  uint32_t np0 = cgbn_bn2mont(env, t, t, n);
  cgbn_load(env, v, (cgbn_mem_t<BITS> *)&gV[idx]);
  cgbn_mont2bn(env, t, v, n, np0);
  cgbn_gcd(env, g, t, n);
  cgbn_store(env, &out[idx], g);
}

// One wave: as many curves as the GPU runs at once for this kernel.
template <uint32_t TPI, uint32_t BITS>
unsigned int resident_curves() {
  int dev = 0, sms = 0, per_sm = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
          cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &per_sm, k_stage1<TPI, BITS>, 128, 0) != cudaSuccess)
    return 4096;
  unsigned int c = (unsigned int)per_sm * (unsigned int)sms * 128u / TPI;
  return c ? c : 4096;
}

// GMPECPP_CUDA_ECM_STATS=1 in the environment: report on stderr, per stage,
// how many curves found a proper divisor and how long the stage took (for
// measuring success rates; normal runs stop at the first divisor).
static bool stats_on() {
  static int on = -1;
  if (on < 0) {
    const char *e = getenv("GMPECPP_CUDA_ECM_STATS");
    on = (e && *e && *e != '0') ? 1 : 0;
  }
  return on == 1;
}

// Scan out[0, curves) for a proper divisor of n.
template <uint32_t BITS>
bool scan_out(const cgbn_mem_t<BITS> *d_out, unsigned int curves,
              mpz_srcptr n, mpz_ptr factor, cudaStream_t s,
              const char *stage = "", double secs = 0.0) {
  cgbn_mem_t<BITS> *h =
      (cgbn_mem_t<BITS> *)malloc(sizeof(cgbn_mem_t<BITS>) * curves);
  if (!h)
    return false;
  cudaMemcpyAsync(h, d_out, sizeof(cgbn_mem_t<BITS>) * curves,
                  cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  bool found = false;
  unsigned hits = 0;
  const bool count = stats_on();
  mpz_t g;
  mpz_init(g);
  for (unsigned i = 0; i < curves && (count || !found); i++) {
    mem_to_mpz<BITS>(g, h[i]);
    if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, n) < 0) {
      if (!found) mpz_set(factor, g);
      found = true;
      ++hits;
    }
  }
  mpz_clear(g);
  free(h);
  if (count && *stage)
    fprintf(stderr, "gmpecpp_cuda_ecm stats: %s: %u of %u curves found a factor (%.2f s)\n",
            stage, hits, curves, secs);
  return found;
}

// Time a launch in seconds (host wall clock around a stream sync).
static double now_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------- //
// Stage 2: prime continuation                                        //
// ---------------------------------------------------------------- //
//
// Q is a curve's point after stage 1. With D = 210, every prime q in
// (B1, B2] is m*D + j or m*D - j for one m and one of the 24 odd j < D/2
// that are coprime to D. [q]Q = O (mod p) exactly when [m*D]Q and [j]Q have
// the same x-coordinate mod p (x does not see the sign), i.e. when
// X_m*Z_j - X_j*Z_m = 0 (mod p). So each curve keeps its 24 baby points
// [j]Q in device memory, NORMALIZED once to x_j = X_j/Z_j (one inversion
// each in the setup; a failed inversion exposes gcd(Z_j, n) as a factor),
// walks the giant points [m*D]Q one differential addition at a time
// ([(m+1)D] = [mD] + [D], difference [(m-1)D]), and multiplies
// X_m - x_j*Z_m into an accumulator for every pair (m, j) that covers a
// prime. gcd(acc, n) at the end. Cost: 2 multiplications per prime (3 before
// the normalization: X_m*Z_j - X_j*Z_m) and one point addition per m,
// against ~16 multiplications per unit of B1 in stage 1. Which pairs to take is the same for every curve, so the
// host builds it once as one 24-bit mask per m.

#define S2_D 210u
#define S2_H 105u
#define S2_NB 24u

__host__ __device__ inline bool s2_coprime(uint32_t j) {
  return (j & 1u) && (j % 3u) && (j % 5u) && (j % 7u);
}

// Baby points x_j = X_j/Z_j, the step [D]Q, the giant pair [m0*D]Q,
// [(m0-1)*D]Q, acc = 1. Layouts: B[k*ncurves + i], G[i*4 + {Xm,Zm,Xp,Zp}],
// Dq[i*2 + {X,Z}], acc[i]; all Montgomery form.
template <uint32_t TPI, uint32_t BITS>
__device__ __forceinline__ bool
s2_store_baby(cgbn_env_t<cgbn_context_t<TPI>, BITS> env,
              const typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t &X,
              const typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t &Z,
              const typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t &n,
              uint32_t np0, cgbn_mem_t<BITS> *dst, cgbn_mem_t<BITS> *out) {
  typename cgbn_env_t<cgbn_context_t<TPI>, BITS>::cgbn_t t, inv, x;
  cgbn_mont2bn(env, t, Z, n, np0);
  if (!cgbn_modular_inverse(env, inv, t, n)) {
    cgbn_gcd(env, x, t, n);
    if (!cgbn_equals_ui32(env, x, 1) && cgbn_compare(env, x, n) != 0) {
      cgbn_store(env, out, x);                 // gcd(Z_j, n) is a factor
      return true;
    }
    cgbn_set_ui32(env, x, 0);                  // Z_j = 0 mod n: useless point
    cgbn_store(env, dst, x);
    return false;
  }
  cgbn_bn2mont(env, inv, inv, n);
  cgbn_mont_mul(env, x, X, inv, n, np0);       // x_j = X_j / Z_j
  cgbn_store(env, dst, x);
  return false;
}

template <uint32_t TPI, uint32_t BITS>
__global__ void k_s2_setup(cgbn_error_report_t *report, cgbn_mem_t<BITS> *n_in,
                           uint32_t ncurves, const cgbn_mem_t<BITS> *gX,
                           const cgbn_mem_t<BITS> *gZ,
                           const cgbn_mem_t<BITS> *gA, uint32_t *status,
                           uint32_t m0, cgbn_mem_t<BITS> *B,
                           cgbn_mem_t<BITS> *G, cgbn_mem_t<BITS> *Dq,
                           cgbn_mem_t<BITS> *acc, cgbn_mem_t<BITS> *out) {
  typedef cgbn_context_t<TPI> context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (idx >= ncurves || status[idx])
    return;
  context_t ctx(cgbn_report_monitor, report, idx);
  env_t env(ctx);

  bn_t n, one, X, Z, a24, X2, Z2, Xa, Za, Xb, Zb, Xc, Zc;
  cgbn_load(env, n, n_in);
  cgbn_set_ui32(env, one, 1);
  uint32_t np0 = cgbn_bn2mont(env, one, one, n);
  cgbn_load(env, X, (cgbn_mem_t<BITS> *)&gX[idx]);
  cgbn_load(env, Z, (cgbn_mem_t<BITS> *)&gZ[idx]);
  cgbn_load(env, a24, (cgbn_mem_t<BITS> *)&gA[idx]);

  // [1]Q, [2]Q, then [j+2] = [j] + [2] with difference [j-2].
  const bool lead = (threadIdx.x % TPI) == 0;
  xz_double(env, X2, Z2, X, Z, a24, n, np0);
  uint32_t k = 0;
  if (s2_store_baby<TPI, BITS>(env, X, Z, n, np0, &B[k * ncurves + idx],
                               &out[idx])) {                 // j = 1
    if (lead) status[idx] = 1;
    return;
  }
  ++k;
  cgbn_set(env, Xa, X);                                      // [j-2] = [1]
  cgbn_set(env, Za, Z);
  xz_add(env, Xb, Zb, X2, Z2, X, Z, X, Z, n, np0);           // [3]
  for (uint32_t j = 3; j < S2_H; j += 2) {
    if (s2_coprime(j)) {
      if (s2_store_baby<TPI, BITS>(env, Xb, Zb, n, np0, &B[k * ncurves + idx],
                                   &out[idx])) {
        if (lead) status[idx] = 1;
        return;
      }
      ++k;
    }
    xz_add(env, Xc, Zc, Xb, Zb, X2, Z2, Xa, Za, n, np0);     // [j+2]
    cgbn_set(env, Xa, Xb);
    cgbn_set(env, Za, Zb);
    cgbn_set(env, Xb, Xc);
    cgbn_set(env, Zb, Zc);
  }

  cgbn_set(env, Xa, X);
  cgbn_set(env, Za, Z);
  xz_mul_ui(env, Xa, Za, a24, n, np0, S2_D);                 // [D]Q
  cgbn_store(env, &Dq[idx * 2], Xa);
  cgbn_store(env, &Dq[idx * 2 + 1], Za);
  cgbn_set(env, Xb, X);
  cgbn_set(env, Zb, Z);
  xz_mul_ui(env, Xb, Zb, a24, n, np0, m0 * S2_D);            // [m0 D]Q
  cgbn_set(env, Xc, X);
  cgbn_set(env, Zc, Z);
  xz_mul_ui(env, Xc, Zc, a24, n, np0, (m0 - 1) * S2_D);      // [(m0-1) D]Q
  cgbn_store(env, &G[idx * 4], Xb);
  cgbn_store(env, &G[idx * 4 + 1], Zb);
  cgbn_store(env, &G[idx * 4 + 2], Xc);
  cgbn_store(env, &G[idx * 4 + 3], Zc);
  cgbn_store(env, &acc[idx], one);                           // R mod n
}

// Giant steps masks[0, mcount): pairs, then [m+1]D.
template <uint32_t TPI, uint32_t BITS>
__global__ void k_s2_chunk(cgbn_error_report_t *report, cgbn_mem_t<BITS> *n_in,
                           uint32_t ncurves, const uint32_t *status,
                           const cgbn_mem_t<BITS> *B, cgbn_mem_t<BITS> *G,
                           const cgbn_mem_t<BITS> *Dq, cgbn_mem_t<BITS> *acc,
                           const uint32_t *masks, uint32_t mcount) {
  typedef cgbn_context_t<TPI> context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (idx >= ncurves || status[idx])
    return;
  context_t ctx(cgbn_report_monitor, report, idx);
  env_t env(ctx);

  bn_t n, t, Xm, Zm, Xp, Zp, XD, ZD, a, xj, t1, Xn, Zn;
  cgbn_load(env, n, n_in);
  cgbn_set_ui32(env, t, 1);
  uint32_t np0 = cgbn_bn2mont(env, t, t, n);
  cgbn_load(env, Xm, &G[idx * 4]);
  cgbn_load(env, Zm, &G[idx * 4 + 1]);
  cgbn_load(env, Xp, &G[idx * 4 + 2]);
  cgbn_load(env, Zp, &G[idx * 4 + 3]);
  cgbn_load(env, XD, (cgbn_mem_t<BITS> *)&Dq[idx * 2]);
  cgbn_load(env, ZD, (cgbn_mem_t<BITS> *)&Dq[idx * 2 + 1]);
  cgbn_load(env, a, &acc[idx]);

  for (uint32_t i = 0; i < mcount; i++) {
    uint32_t mask = masks[i];
    while (mask) {
      const uint32_t k = __ffs(mask) - 1;
      mask &= mask - 1;
      cgbn_load(env, xj, (cgbn_mem_t<BITS> *)&B[k * ncurves + idx]);
      cgbn_mont_mul(env, t1, xj, Zm, n, np0);          // X_m - x_j * Z_m
      mod_sub(env, t1, Xm, t1, n);
      cgbn_mont_mul(env, a, a, t1, n, np0);
    }
    xz_add(env, Xn, Zn, Xm, Zm, XD, ZD, Xp, Zp, n, np0);     // [(m+1) D]
    cgbn_set(env, Xp, Xm);
    cgbn_set(env, Zp, Zm);
    cgbn_set(env, Xm, Xn);
    cgbn_set(env, Zm, Zn);
  }
  cgbn_store(env, &G[idx * 4], Xm);
  cgbn_store(env, &G[idx * 4 + 1], Zm);
  cgbn_store(env, &G[idx * 4 + 2], Xp);
  cgbn_store(env, &G[idx * 4 + 3], Zp);
  cgbn_store(env, &acc[idx], a);
}

// Pair masks for (b1, b2]: masks[m - m_lo] has bit k set when m*D +- j_k is
// a prime in range. Primes by a segmented sieve (memory: one segment).
static void s2_masks(unsigned long b1, unsigned long b2, uint32_t *m_lo_out,
                     uint32_t **masks_out, uint32_t *nm_out) {
  int8_t kof[S2_H + 1];
  memset(kof, -1, sizeof kof);
  for (uint32_t j = 1, k = 0; j < S2_H; j += 2)
    if (s2_coprime(j)) kof[j] = (int8_t)k++;

  const uint64_t lo = (uint64_t)b1 + 1, hi = b2;
  const uint32_t m_lo = (uint32_t)((lo + S2_H) / S2_D);
  const uint32_t m_hi = (uint32_t)((hi + S2_H) / S2_D);
  const uint32_t nm = m_hi - m_lo + 1;
  uint32_t *masks = (uint32_t *)calloc(nm, sizeof(uint32_t));

  uint64_t r = 1;
  while ((r + 1) * (r + 1) <= hi) r++;
  std::vector<uint32_t> base;
  {
    std::vector<uint8_t> c(r + 1, 0);
    for (uint64_t i = 2; i <= r; i++)
      if (!c[i]) {
        base.push_back((uint32_t)i);
        for (uint64_t j = i * i; j <= r; j += i) c[j] = 1;
      }
  }
  const uint64_t SEG = 1u << 20;
  std::vector<uint8_t> comp(SEG);
  for (uint64_t s0 = lo; s0 <= hi; s0 += SEG) {
    const uint64_t s1 = s0 + SEG - 1 < hi ? s0 + SEG - 1 : hi;
    std::fill(comp.begin(), comp.end(), 0);
    for (uint32_t p : base) {
      uint64_t st = ((s0 + p - 1) / p) * p;
      if (st < (uint64_t)p * p) st = (uint64_t)p * p;
      for (uint64_t x = st; x <= s1; x += p) comp[x - s0] = 1;
    }
    for (uint64_t q = s0; q <= s1; q++) {
      if (comp[q - s0] || q < 11) continue;          // 2,3,5,7 are below B1
      const uint32_t m = (uint32_t)((q + S2_H) / S2_D);
      const int64_t j = (int64_t)q - (int64_t)m * S2_D;
      const uint32_t a = (uint32_t)(j < 0 ? -j : j);
      masks[m - m_lo] |= 1u << kof[a];
    }
  }
  *m_lo_out = m_lo;
  *masks_out = masks;
  *nm_out = nm;
}

template <uint32_t TPI, uint32_t BITS>
int run_stage2(mpz_srcptr n, unsigned long b1, unsigned long b2,
               unsigned int curves, cgbn_mem_t<BITS> *d_n,
               cgbn_mem_t<BITS> *d_X, cgbn_mem_t<BITS> *d_Z,
               cgbn_mem_t<BITS> *d_A, uint32_t *d_status,
               cgbn_mem_t<BITS> *d_out, cgbn_error_report_t *report,
               cudaStream_t s, int (*stop)(void *), void *stop_arg,
               mpz_ptr factor) {
  typedef cgbn_mem_t<BITS> mem;
  if (b2 > 0xffffffffUL) b2 = 0xffffffffUL;
  uint32_t m_lo = 0, nm = 0, *masks = NULL;
  s2_masks(b1, b2, &m_lo, &masks, &nm);
  if (!masks)
    return -1;

  mem *d_B = NULL, *d_G = NULL, *d_Dq = NULL, *d_acc = NULL;
  uint32_t *d_mask = NULL;
  bool ok = cudaMalloc(&d_B, sizeof(mem) * curves * S2_NB) == cudaSuccess;
  ok = ok && cudaMalloc(&d_G, sizeof(mem) * curves * 4) == cudaSuccess;
  ok = ok && cudaMalloc(&d_Dq, sizeof(mem) * curves * 2) == cudaSuccess;
  ok = ok && cudaMalloc(&d_acc, sizeof(mem) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_mask, sizeof(uint32_t) * nm) == cudaSuccess;

  const uint32_t tpb = 128;
  const uint32_t blocks = (curves * TPI + tpb - 1) / tpb;
  int result = ok ? 0 : -1;
  bool stopped = false;
  if (ok) {
    cudaMemcpyAsync(d_mask, masks, sizeof(uint32_t) * nm,
                    cudaMemcpyHostToDevice, s);
    const double t_s2 = now_seconds();
    k_s2_setup<TPI, BITS><<<blocks, tpb, 0, s>>>(report, d_n, curves, d_X, d_Z,
                                                 d_A, d_status, m_lo, d_B,
                                                 d_G, d_Dq, d_acc, d_out);
    ok = cudaStreamSynchronize(s) == cudaSuccess;
    uint32_t i = 0, slice = 64;
    while (ok && i < nm) {
      if (stop && stop(stop_arg)) {
        stopped = true;
        break;
      }
      uint32_t cnt = slice < nm - i ? slice : nm - i;
      double t0 = now_seconds();
      k_s2_chunk<TPI, BITS><<<blocks, tpb, 0, s>>>(report, d_n, curves,
                                                   d_status, d_B, d_G, d_Dq,
                                                   d_acc, d_mask + i, cnt);
      ok = cudaStreamSynchronize(s) == cudaSuccess;
      double dt = now_seconds() - t0;
      i += cnt;
      double want = dt > 0 ? cnt * (0.25 / dt) : cnt * 4.0;
      if (want > cnt * 4.0) want = cnt * 4.0;
      slice = want < 1.0 ? 1u : (uint32_t)want;
    }
    if (ok && !stopped) {
      k_gcd<TPI, BITS><<<blocks, tpb, 0, s>>>(report, d_n, curves, d_acc,
                                              d_status, d_out);
      ok = cudaStreamSynchronize(s) == cudaSuccess;
      if (ok && cgbn_error_report_check(report))
        ok = false;
      if (ok && scan_out<BITS>(d_out, curves, n, factor, s, "stage 2",
                               now_seconds() - t_s2))
        result = 1;
    }
    if (!ok)
      result = -1;
  }
  if (d_B) cudaFree(d_B);
  if (d_G) cudaFree(d_G);
  if (d_Dq) cudaFree(d_Dq);
  if (d_acc) cudaFree(d_acc);
  if (d_mask) cudaFree(d_mask);
  free(masks);
  return result;
}

// Returns 1 = factor found, 0 = none (or stopped), -1 = CUDA error.
template <uint32_t TPI, uint32_t BITS>
int run_resumable(mpz_srcptr n, unsigned long b1, unsigned long b2,
                  unsigned int curves, unsigned long seed,
                  int (*stop)(void *), void *stop_arg, mpz_ptr factor) {
  if (curves == 0)
    curves = resident_curves<TPI, BITS>();

  uint32_t *powers = NULL, npowers = 0;
  build_powers(b1, &powers, &npowers);
  uint32_t *sigmas = (uint32_t *)malloc(curves * sizeof(uint32_t));
  if (!npowers || !sigmas) {
    free(powers);
    free(sigmas);
    return -1;
  }
  gmp_randstate_t st;
  gmp_randinit_default(st);
  gmp_randseed_ui(st, seed);
  for (unsigned i = 0; i < curves; i++)
    sigmas[i] = 6 + (uint32_t)gmp_urandomm_ui(st, 1u << 30);
  gmp_randclear(st);

  typedef cgbn_mem_t<BITS> mem;
  mem h_n;
  mpz_to_mem<BITS>(h_n, n);

  cudaStream_t s = NULL;
  mem *d_n = NULL, *d_X = NULL, *d_Z = NULL, *d_A = NULL, *d_out = NULL;
  uint32_t *d_sig = NULL, *d_pow = NULL, *d_status = NULL;
  cgbn_error_report_t *report = NULL;
  bool ok = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess;
  ok = ok && cudaMalloc(&d_n, sizeof(mem)) == cudaSuccess;
  ok = ok && cudaMalloc(&d_X, sizeof(mem) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_Z, sizeof(mem) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_A, sizeof(mem) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_out, sizeof(mem) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_sig, sizeof(uint32_t) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_status, sizeof(uint32_t) * curves) == cudaSuccess;
  ok = ok && cudaMalloc(&d_pow, sizeof(uint32_t) * npowers) == cudaSuccess;
  ok = ok && cgbn_error_report_alloc(&report) == cudaSuccess;

  const uint32_t tpb = 128;
  const uint32_t blocks = (curves * TPI + tpb - 1) / tpb;
  int result = ok ? 0 : -1;
  bool stopped = false;

  if (ok) {
    cudaMemcpyAsync(d_n, &h_n, sizeof h_n, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_sig, sigmas, sizeof(uint32_t) * curves,
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_pow, powers, sizeof(uint32_t) * npowers,
                    cudaMemcpyHostToDevice, s);
    k_init<TPI, BITS><<<blocks, tpb, 0, s>>>(report, d_n, d_sig, curves, d_X,
                                             d_Z, d_A, d_status, d_out);
    ok = cudaStreamSynchronize(s) == cudaSuccess;

    // Stage 1 in launches of ~0.25 s, the slice size adapted as we go.
    const double t_s1 = now_seconds();
    uint32_t i = 0, slice = 16;
    while (ok && i < npowers) {
      if (stop && stop(stop_arg)) {
        stopped = true;
        break;
      }
      uint32_t cnt = slice < npowers - i ? slice : npowers - i;
      double t0 = now_seconds();
      k_stage1<TPI, BITS><<<blocks, tpb, 0, s>>>(report, d_n, curves,
                                                 d_pow + i, cnt, d_X, d_Z,
                                                 d_A, d_status);
      ok = cudaStreamSynchronize(s) == cudaSuccess;
      double dt = now_seconds() - t0;
      i += cnt;
      double want = dt > 0 ? cnt * (0.25 / dt) : cnt * 4.0;
      if (want > cnt * 4.0) want = cnt * 4.0;
      slice = want < 1.0 ? 1u : (uint32_t)want;
    }
    if (ok && !stopped) {
      k_gcd<TPI, BITS><<<blocks, tpb, 0, s>>>(report, d_n, curves, d_Z,
                                              d_status, d_out);
      ok = cudaStreamSynchronize(s) == cudaSuccess;
    }
    if (ok && cgbn_error_report_check(report))
      ok = false;
    // A failed inversion in k_init counts even when stopped.
    if (ok && scan_out<BITS>(d_out, curves, n, factor, s, "stage 1",
                             now_seconds() - t_s1))
      result = 1;
    // Stats mode runs stage 2 even after a stage-1 find, to count its hits
    // (cumulative: a curve that found p in stage 1 finds it in stage 2 too).
    if (ok && !stopped && b2 > b1 && (result == 0 || stats_on())) {
      mpz_t f2;
      mpz_init(f2);
      const int r2 = run_stage2<TPI, BITS>(n, b1, b2, curves, d_n, d_X, d_Z,
                                           d_A, d_status, d_out, report, s,
                                           stop, stop_arg, f2);
      if (result == 0) {
        result = r2;
        if (r2 == 1) mpz_set(factor, f2);
      }
      mpz_clear(f2);
    }
    if (!ok)
      result = -1;
  }

  if (report) cgbn_error_report_free(report);
  if (d_n) cudaFree(d_n);
  if (d_X) cudaFree(d_X);
  if (d_Z) cudaFree(d_Z);
  if (d_A) cudaFree(d_A);
  if (d_out) cudaFree(d_out);
  if (d_sig) cudaFree(d_sig);
  if (d_status) cudaFree(d_status);
  if (d_pow) cudaFree(d_pow);
  if (s) cudaStreamDestroy(s);
  free(sigmas);
  free(powers);
  return result;
}

} // namespace

namespace gmpecpp_detail {

bool cuda_available() { return device_info().usable; }

const char *cuda_device_name() {
  DeviceInfo &d = device_info();
  return d.usable ? d.name : 0;
}

/*
 * Try to split n with `curves` concurrent ECM curves at bound b1.
 * Returns true and sets `factor` to a nontrivial divisor on success.
 *
 * The CGBN instance width is fixed at compile time, so pick the smallest
 * that comfortably holds n; a wider instance costs proportionally more.
 */
bool cuda_ecm_factor(mpz_srcptr n, unsigned long b1, unsigned int curves,
                     unsigned long seed, mpz_ptr factor) {
  if (!device_info().usable || curves == 0)
    return false;
  size_t bits = mpz_sizeinbase(n, 2);
  if (mpz_even_p(n) || mpz_cmp_ui(n, 3) <= 0)
    return false; // Montgomery arithmetic needs an odd modulus

  if (bits <= 480)
    return run_bits<8, 512>(n, b1, curves, seed, factor);
  if (bits <= 992)
    return run_bits<16, 1024>(n, b1, curves, seed, factor);
  if (bits <= 2016)
    return run_bits<32, 2048>(n, b1, curves, seed, factor);
  return false; // beyond what this backend is configured for
}

/*
 * gmpecpp_cuda_ecm(): curves on the GPU only, resumable, stage 1 to b1 and
 * stage 2 to b2.  Returns 1 = factor, 0 = none or stopped, -1 = CUDA error
 * or no device, -2 = n not supported (even, < 5, wider than 16352 bits) or
 * b1 outside [1000, 2^32).
 */
int cuda_ecm_run(mpz_srcptr n, unsigned long b1, unsigned long b2,
                 unsigned int curves, unsigned long seed, int (*stop)(void *),
                 void *stop_arg, mpz_ptr factor) {
  if (!device_info().usable)
    return -1;
  if (mpz_even_p(n) || mpz_cmp_ui(n, 3) <= 0 || b1 < 1000 ||
      b1 > 0xffffffffUL)
    return -2;
  size_t bits = mpz_sizeinbase(n, 2);
  if (bits <= 480)
    return run_resumable<8, 512>(n, b1, b2, curves, seed, stop, stop_arg, factor);
  if (bits <= 992)
    return run_resumable<16, 1024>(n, b1, b2, curves, seed, stop, stop_arg, factor);
  if (bits <= 2016)
    return run_resumable<32, 2048>(n, b1, b2, curves, seed, stop, stop_arg, factor);
  if (bits <= 4064)
    return run_resumable<32, 4096>(n, b1, b2, curves, seed, stop, stop_arg, factor);
  if (bits <= 8160)
    return run_resumable<32, 8192>(n, b1, b2, curves, seed, stop, stop_arg, factor);
  if (bits <= 16352)
    return run_resumable<32, 16384>(n, b1, b2, curves, seed, stop, stop_arg, factor);
  return -2;
}

// The curve count gmpecpp_cuda_ecm() uses by default for an n of `bits`.
unsigned int cuda_ecm_default_curves(size_t bits) {
  if (!device_info().usable)
    return 0;
  if (bits <= 480) return resident_curves<8, 512>();
  if (bits <= 992) return resident_curves<16, 1024>();
  if (bits <= 2016) return resident_curves<32, 2048>();
  if (bits <= 4064) return resident_curves<32, 4096>();
  if (bits <= 8160) return resident_curves<32, 8192>();
  if (bits <= 16352) return resident_curves<32, 16384>();
  return 0;
}

} // namespace gmpecpp_detail
