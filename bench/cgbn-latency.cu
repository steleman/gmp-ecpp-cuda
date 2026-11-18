/*
 * cgbn-latency.cu -- one GMP operation against one CGBN operation and
 * against a batch of 4096 (make bench-cgbn-latency).
 *
 * For 512, 1024 and 2048 bits (CGBN TPI 8, 16, 32): modular
 * exponentiation with a full-size exponent, and a multiplication modulo
 * n. GMP is one CPU thread. "CGBN one" is a full round trip for a single
 * instance -- copy the operands in, launch, synchronize, copy the result
 * out -- which is what replacing one GMP call with CGBN costs. "CGBN
 * batch" is 4096 independent instances in one launch (result copy
 * included), per operation. One result is checked against GMP.
 *
 * Copyright (c) 2026 Stefan Teleman and contributors. GPL-3.0-or-later.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <gmp.h>
#include <cuda_runtime.h>
#include <cgbn/cgbn.h>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <uint32_t BITS> void to_mem(cgbn_mem_t<BITS> &d, mpz_srcptr v) {
  memset(d._limbs, 0, sizeof d._limbs); size_t c = 0;
  mpz_export(d._limbs, &c, -1, sizeof(uint32_t), 0, 0, v);
}

template <uint32_t TPI, uint32_t BITS>
__global__ void k_powm(cgbn_mem_t<BITS> *a, cgbn_mem_t<BITS> *e, cgbn_mem_t<BITS> *n, cgbn_mem_t<BITS> *r, uint32_t cnt) {
  const uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (inst >= cnt) return;
  typedef cgbn_context_t<TPI> ctx_t; typedef cgbn_env_t<ctx_t, BITS> env_t;
  ctx_t ctx(cgbn_no_checks); env_t env(ctx.template env<env_t>());
  typename env_t::cgbn_t A, E, N, R;
  cgbn_load(env, A, &a[inst]); cgbn_load(env, E, &e[inst]); cgbn_load(env, N, &n[inst]);
  cgbn_modular_power(env, R, A, E, N);
  cgbn_store(env, &r[inst], R);
}
template <uint32_t TPI, uint32_t BITS>
__global__ void k_mulmod(cgbn_mem_t<BITS> *a, cgbn_mem_t<BITS> *b, cgbn_mem_t<BITS> *n, cgbn_mem_t<BITS> *r, uint32_t cnt) {
  const uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (inst >= cnt) return;
  typedef cgbn_context_t<TPI> ctx_t; typedef cgbn_env_t<ctx_t, BITS> env_t;
  ctx_t ctx(cgbn_no_checks); env_t env(ctx.template env<env_t>());
  typename env_t::cgbn_t A, B, N, R; typename env_t::cgbn_wide_t W;
  cgbn_load(env, A, &a[inst]); cgbn_load(env, B, &b[inst]); cgbn_load(env, N, &n[inst]);
  cgbn_mul_wide(env, W, A, B); cgbn_rem_wide(env, R, W, N);
  cgbn_store(env, &r[inst], R);
}

template <uint32_t TPI, uint32_t BITS> void run(gmp_randstate_t rs) {
  typedef cgbn_mem_t<BITS> mem;
  const unsigned bits = BITS - 32; // operands a little narrower than the width
  mpz_t n, a, b, e, r, g; mpz_inits(n, a, b, e, r, g, NULL);
  mpz_urandomb(n, rs, bits); mpz_setbit(n, bits - 1); mpz_setbit(n, 0);
  mpz_urandomm(a, rs, n); mpz_urandomm(b, rs, n); mpz_urandomb(e, rs, bits);
  // GMP, one thread
  int reps = 2000; double t0 = now();
  for (int i = 0; i < reps; i++) mpz_powm(r, a, e, n);
  const double g_pow = (now() - t0) / reps;
  reps = 200000; t0 = now();
  for (int i = 0; i < reps; i++) { mpz_mul(g, a, b); mpz_mod(r, g, n); }
  const double g_mul = (now() - t0) / reps;
  // CGBN buffers for up to 4096 instances
  const uint32_t B = 4096;
  std::vector<mem> ha(B), hb(B), he(B), hn(B), hr(B);
  for (uint32_t i = 0; i < B; i++) { to_mem(ha[i], a); to_mem(hb[i], b); to_mem(he[i], e); to_mem(hn[i], n); }
  mem *da, *db, *de, *dn, *dr;
  for (mem **p : {&da, &db, &de, &dn, &dr}) cudaMalloc((void **)p, sizeof(mem) * B);
  auto one = [&](bool pw) { // one instance, a full round trip
    cudaMemcpy(da, &ha[0], sizeof(mem), cudaMemcpyHostToDevice);
    cudaMemcpy(pw ? de : db, pw ? &he[0] : &hb[0], sizeof(mem), cudaMemcpyHostToDevice);
    cudaMemcpy(dn, &hn[0], sizeof(mem), cudaMemcpyHostToDevice);
    if (pw) k_powm<TPI, BITS><<<1, TPI>>>(da, de, dn, dr, 1); else k_mulmod<TPI, BITS><<<1, TPI>>>(da, db, dn, dr, 1);
    cudaMemcpy(&hr[0], dr, sizeof(mem), cudaMemcpyDeviceToHost);
  };
  for (int w = 0; w < 20; w++) { one(true); one(false); }
  reps = 300; t0 = now(); for (int i = 0; i < reps; i++) one(true); const double c1_pow = (now() - t0) / reps;
  reps = 2000; t0 = now(); for (int i = 0; i < reps; i++) one(false); const double c1_mul = (now() - t0) / reps;
  // check one result against GMP
  mpz_powm(g, a, e, n); mpz_t c; mpz_init(c); one(true);
  mpz_import(c, BITS / 32, -1, sizeof(uint32_t), 0, 0, hr[0]._limbs); const bool ok = mpz_cmp(c, g) == 0;
  // batch of B, copies included
  cudaMemcpy(da, ha.data(), sizeof(mem) * B, cudaMemcpyHostToDevice);
  cudaMemcpy(de, he.data(), sizeof(mem) * B, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb.data(), sizeof(mem) * B, cudaMemcpyHostToDevice);
  cudaMemcpy(dn, hn.data(), sizeof(mem) * B, cudaMemcpyHostToDevice);
  const uint32_t thr = 128, blocks = (B * TPI + thr - 1) / thr;
  k_powm<TPI, BITS><<<blocks, thr>>>(da, de, dn, dr, B); cudaDeviceSynchronize();
  reps = 5; t0 = now();
  for (int i = 0; i < reps; i++) { k_powm<TPI, BITS><<<blocks, thr>>>(da, de, dn, dr, B); cudaMemcpy(hr.data(), dr, sizeof(mem) * B, cudaMemcpyDeviceToHost); }
  const double cb_pow = (now() - t0) / reps / B;
  reps = 50; t0 = now();
  for (int i = 0; i < reps; i++) { k_mulmod<TPI, BITS><<<blocks, thr>>>(da, db, dn, dr, B); cudaMemcpy(hr.data(), dr, sizeof(mem) * B, cudaMemcpyDeviceToHost); }
  const double cb_mul = (now() - t0) / reps / B;
  printf("%5u bits | powm: GMP %8.1f us | CGBN one %8.1f us (%5.1fx) | CGBN batch %7.2f us/op (%6.1fx) | %s\n",
         BITS, g_pow * 1e6, c1_pow * 1e6, c1_pow / g_pow, cb_pow * 1e6, g_pow / cb_pow, ok ? "result ok" : "RESULT WRONG");
  printf("%5s      | mulmod: GMP %6.3f us | CGBN one %8.1f us (%5.0fx) | CGBN batch %7.3f us/op (%6.1fx)\n",
         "", g_mul * 1e6, c1_mul * 1e6, c1_mul / g_mul, cb_mul * 1e6, g_mul / cb_mul);
  for (mem *p : {da, db, de, dn, dr}) cudaFree(p);
  mpz_clears(n, a, b, e, r, g, c, NULL);
}

int main() {
  gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, 1);
  printf("(\"one\": copy in + launch + sync + copy out, slower by the factor shown; \"batch\": 4096 per launch, faster by the factor shown)\n");
  run<8, 512>(rs); run<16, 1024>(rs); run<32, 2048>(rs);
  return 0;
}
