/*
 * gmpecpp-verify -- independently check a certificate produced by
 * libgmpecpp.
 *
 * The point of a primality *proof* is that it can be checked without
 * trusting the prover, so this deliberately shares no code with it: the
 * elliptic curve arithmetic below is written against plain GMP.
 *
 * For each step it checks
 *   - chain linkage: n_0 is the input, and each n_(i+1) is the previous q
 *   - q divides m
 *   - q > (n^(1/4) + 1)^2                     (the Goldwasser-Kilian bound)
 *   - |m - (n+1)| <= 2*sqrt(n)                (Hasse)
 *   - P lies on y^2 = x^3 + ax + b mod n
 *   - [m/q]P == P2 and P2 != O
 *   - [q]P2 == O
 * and finally that the last q is small enough to settle by trial
 * division, and is prime.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>
#include "gmpecpp.h"

/* Affine point; z==0 marks the point at infinity. */
typedef struct { mpz_t x, y; int inf; } pt;

static void pt_init(pt *p) { mpz_inits(p->x, p->y, NULL); p->inf = 1; }
static void pt_clear(pt *p) { mpz_clears(p->x, p->y, NULL); }
static void pt_set(pt *r, const pt *a) {
  mpz_set(r->x, a->x); mpz_set(r->y, a->y); r->inf = a->inf;
}

/* r = a + b on y^2 = x^3 + A x + b over Z/nZ.  Returns 0 if an inverse
 * did not exist, which would mean n is composite. */
static int pt_add(pt *r, const pt *a, const pt *b, mpz_srcptr A, mpz_srcptr n) {
  if (a->inf) { pt_set(r, b); return 1; }
  if (b->inf) { pt_set(r, a); return 1; }

  mpz_t l, t, u, v;
  mpz_inits(l, t, u, v, NULL);
  int ok = 1;

  if (mpz_cmp(a->x, b->x) == 0) {
    mpz_add(t, a->y, b->y);
    mpz_mod(t, t, n);
    if (mpz_sgn(t) == 0) { r->inf = 1; goto done; }   /* P + (-P) = O */
    /* doubling: l = (3x^2 + A) / 2y */
    mpz_mul(u, a->x, a->x); mpz_mod(u, u, n);
    mpz_mul_ui(u, u, 3);    mpz_add(u, u, A); mpz_mod(u, u, n);
    mpz_mul_ui(v, a->y, 2); mpz_mod(v, v, n);
  } else {
    mpz_sub(u, b->y, a->y); mpz_mod(u, u, n);
    mpz_sub(v, b->x, a->x); mpz_mod(v, v, n);
  }
  if (!mpz_invert(t, v, n)) { ok = 0; goto done; }
  mpz_mul(l, u, t); mpz_mod(l, l, n);

  mpz_mul(t, l, l); mpz_mod(t, t, n);
  mpz_sub(t, t, a->x); mpz_sub(t, t, b->x); mpz_mod(t, t, n);
  mpz_sub(u, a->x, t); mpz_mul(u, u, l); mpz_sub(u, u, a->y); mpz_mod(u, u, n);
  mpz_set(r->x, t); mpz_set(r->y, u); r->inf = 0;

done:
  mpz_clears(l, t, u, v, NULL);
  return ok;
}

/* r = [k]p, binary ladder. */
static int pt_mul(pt *r, mpz_srcptr k, const pt *p, mpz_srcptr A, mpz_srcptr n) {
  pt acc, cur, tmp;
  pt_init(&acc); pt_init(&cur); pt_init(&tmp);
  pt_set(&cur, p);
  int ok = 1;
  for (mp_bitcnt_t i = 0, bits = mpz_sizeinbase(k, 2); i < bits && ok; i++) {
    if (mpz_tstbit(k, i)) { ok = pt_add(&tmp, &acc, &cur, A, n); pt_set(&acc, &tmp); }
    if (ok && i + 1 < bits) { ok = pt_add(&tmp, &cur, &cur, A, n); pt_set(&cur, &tmp); }
  }
  pt_set(r, &acc);
  pt_clear(&acc); pt_clear(&cur); pt_clear(&tmp);
  return ok;
}

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); bad++; } while (0)

int gmpecpp_verify(const char *input, const gmpecpp_certificate_t *cert,
                   int verbose) {
  size_t steps = gmpecpp_certificate_num_steps(cert);
  int bad = 0;
  mpz_t n, a, b, m, q, mq, t, u, prev_q;
  mpz_inits(n, a, b, m, q, mq, t, u, prev_q, NULL);
  mpz_set_str(prev_q, input, 10);

  for (size_t i = 0; i < steps; i++) {
    gmpecpp_step_t s;
    gmpecpp_certificate_get_step(cert, i, &s);
    mpz_set_str(n, s.n, 10); mpz_set_str(a, s.a, 10); mpz_set_str(b, s.b, 10);
    mpz_set_str(m, s.m, 10); mpz_set_str(q, s.q, 10);

    if (mpz_cmp(n, prev_q) != 0) FAIL("step %zu: n does not continue the chain", i);

    /* q | m */
    mpz_tdiv_qr(mq, t, m, q);
    if (mpz_sgn(t) != 0) FAIL("step %zu: q does not divide m", i);

    /* q > (n^(1/4)+1)^2 */
    mpz_sqrt(t, n); mpz_sqrt(t, t); mpz_add_ui(t, t, 1); mpz_mul(t, t, t);
    if (mpz_cmp(q, t) <= 0) FAIL("step %zu: q <= (n^(1/4)+1)^2", i);

    /* |m - (n+1)| <= 2 sqrt(n) */
    mpz_add_ui(t, n, 1); mpz_sub(t, m, t); mpz_abs(t, t);
    mpz_sqrt(u, n); mpz_mul_ui(u, u, 2); mpz_add_ui(u, u, 2);
    if (mpz_cmp(t, u) > 0) FAIL("step %zu: m violates the Hasse bound", i);

    /* P on the curve */
    pt P, P2, R;
    pt_init(&P); pt_init(&P2); pt_init(&R);
    mpz_set_str(P.x, s.px, 10); mpz_set_str(P.y, s.py, 10); P.inf = 0;
    mpz_mul(t, P.y, P.y); mpz_mod(t, t, n);
    mpz_mul(u, P.x, P.x); mpz_mod(u, u, n); mpz_mul(u, u, P.x);
    mpz_addmul(u, a, P.x); mpz_add(u, u, b); mpz_mod(u, u, n);
    if (mpz_cmp(t, u) != 0) FAIL("step %zu: P is not on the curve", i);

    /* [m/q]P == P2, and P2 != O */
    if (!pt_mul(&P2, mq, &P, a, n)) FAIL("step %zu: [m/q]P hit a zero divisor", i);
    if (P2.inf) FAIL("step %zu: [m/q]P is the point at infinity", i);
    else {
      mpz_set_str(t, s.p2x, 10); mpz_set_str(u, s.p2y, 10);
      if (mpz_cmp(P2.x, t) != 0 || mpz_cmp(P2.y, u) != 0)
        FAIL("step %zu: recomputed [m/q]P does not match P2", i);
    }

    /* [q]P2 == O */
    if (!pt_mul(&R, q, &P2, a, n)) FAIL("step %zu: [q]P2 hit a zero divisor", i);
    if (!R.inf) FAIL("step %zu: [q]P2 is not the point at infinity", i);

    if (verbose)
      printf("  step %zu: %zu-digit n, D=%ld  ok\n", i,
             strlen(s.n), s.discriminant);

    pt_clear(&P); pt_clear(&P2); pt_clear(&R);
    mpz_set(prev_q, q);
  }

  /* The chain has to bottom out somewhere provable directly. */
  if (mpz_sizeinbase(prev_q, 2) > 31)
    FAIL("final q is too large to settle by trial division");
  else if (!mpz_probab_prime_p(prev_q, 40))
    FAIL("final q is not prime");

  mpz_clears(n, a, b, m, q, mq, t, u, prev_q, NULL);
  return bad;
}

static void quiet(const char *m, void *u) { (void)m; (void)u; }

int main(int argc, char **argv) {
  int use_cuda = 0, verbose = 0, selftest = 0, i;
  unsigned threads = 1, sweep = 1;
  const char *num = NULL;
  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-C")) use_cuda = 1;
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else if (!strcmp(argv[i], "--selftest")) selftest = 1;
    else if (!strcmp(argv[i], "-T") && i + 1 < argc) threads = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-S") && i + 1 < argc) sweep = (unsigned)strtoul(argv[++i], NULL, 10);
    else num = argv[i];
  }
  if (!num) { fprintf(stderr, "usage: %s [-C] [-v] <number>\n", argv[0]); return 1; }

  gmpecpp_config_t *cfg = gmpecpp_config_new();
  gmpecpp_config_set_seed(cfg, 1);
  gmpecpp_config_set_verbosity(cfg, 0);
  gmpecpp_config_set_log(cfg, quiet, NULL);
  if (use_cuda) gmpecpp_config_set_backend(cfg, GMPECPP_BACKEND_CUDA);
  gmpecpp_config_set_threads(cfg, threads);
  gmpecpp_config_set_sweep_threads(cfg, sweep);

  gmpecpp_certificate_t *cert = NULL;
  gmpecpp_status_t st = gmpecpp_prove(cfg, num, &cert);
  printf("%s T=%-2u: %s (%zu steps, %.2fs, %s)\n", use_cuda ? "CUDA" : "CPU ", threads,
         gmpecpp_status_string(st), gmpecpp_certificate_num_steps(cert),
         gmpecpp_certificate_elapsed(cert),
         gmpecpp_certificate_backend(cert) == GMPECPP_BACKEND_CUDA ? "gpu" : "cpu");

  /* A verifier nobody has seen reject anything is not evidence.  Check
     the certificate against the wrong starting number: the chain linkage
     test must fire. */
  if (selftest) {
    printf("  selftest (cert checked against the wrong n):\n");
    int bad = gmpecpp_verify("39", cert, 0);
    printf("  -> %s\n", bad ? "correctly rejected" : "ACCEPTED -- verifier is broken");
    gmpecpp_certificate_free(cert);
    gmpecpp_config_free(cfg);
    return bad ? 0 : 1;
  }

  int rc = 0;
  if (st == GMPECPP_PROVEN_PRIME) {
    int bad = gmpecpp_verify(num, cert, verbose);
    printf("  certificate: %s\n", bad ? "INVALID" : "verified independently");
    rc = bad ? 1 : 0;
  }
  gmpecpp_certificate_free(cert);
  gmpecpp_config_free(cfg);
  return rc;
}
