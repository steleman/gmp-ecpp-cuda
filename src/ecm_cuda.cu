//
// ecm_cuda.cu -- CUDA/CGBN backend for the ECM factoring stage.
//
// Lenstra's elliptic curve method is the one part of this prover that is
// embarrassingly parallel: every curve is an independent trial, and only
// one of them has to succeed.  That maps directly onto CGBN, which runs
// many fixed-width big integer instances at once, each spread across TPI
// cooperating threads.
//
// Stage 1 only.  Curves are Montgomery form B*y^2 = x^3 + A*x^2 + x with
// Suyama's parametrisation, worked in XZ coordinates so the inner loop
// needs no modular inversion -- the whole point, since an inversion per
// step would dominate.  All arithmetic is in Montgomery representation.
//
// Copyright (c) 2005 Paul Zimmermann and contributors (the CPU ECM this
// mirrors); GPL-3.0-or-later, as the rest of the program.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

// Montgomery curve doubling in XZ coordinates.
//   a = X+Z, b = X-Z, c = a^2 - b^2
//   X2 = a^2 * b^2
//   Z2 = c * (b^2 + a24*c)
// Output must not alias the input.
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

// Differential addition: given P1, P2 and their difference D = P1 - P2,
//   X3 = Zd * ((X1-Z1)(X2+Z2) + (X1+Z1)(X2-Z2))^2
//   Z3 = Xd * ((X1-Z1)(X2+Z2) - (X1+Z1)(X2-Z2))^2
// Output must not alias the inputs.
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

// Montgomery ladder: (X,Z) <- [k](X,Z), k a small scalar.
// Maintains R1 - R0 = P so the differential addition always has P as its
// difference.
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
  if (!info.probed) {
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
  }
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

} // namespace

namespace gmpecpp_detail {

bool cuda_available() { return device_info().usable; }

const char *cuda_device_name() {
  DeviceInfo &d = device_info();
  return d.usable ? d.name : 0;
}

// Try to split n with `curves` concurrent ECM curves at bound b1.
// Returns true and sets `factor` to a nontrivial divisor on success.
//
// The CGBN instance width is fixed at compile time, so pick the smallest
// that comfortably holds n; a wider instance costs proportionally more.
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

} // namespace gmpecpp_detail
