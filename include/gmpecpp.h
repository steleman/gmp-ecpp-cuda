/*
 * gmpecpp.h -- public API for libgmpecpp, an elliptic curve primality
 * prover (ECPP, Atkin-Morain) built on GMP.
 *
 * Copyright (c) 1997 Pate Williams, (c) 2008-19 James Wanless,
 * (c) 2004 Chris Studholme, (c) 2003 Elisavet Konstantinou,
 * Yiannis Stamatiu and Christos Zaroliagis, (c) 1996 Scott Contini,
 * (c) 2005 Paul Zimmermann, (c) 2010 Timo Poikola.
 *
 * This file is part of the GMP-ECPP Math C++ Program.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.  See <http://www.gnu.org/licenses/>.
 */

#ifndef GMPECPP_H
#define GMPECPP_H

#include <stddef.h>

#define GMPECPP_VERSION_MAJOR 1
#define GMPECPP_VERSION_MINOR 0
#define GMPECPP_VERSION_PATCH 0

#if defined(_WIN32) || defined(__CYGWIN__)
#define GMPECPP_API __declspec(dllimport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#define GMPECPP_API __attribute__((visibility("default")))
#else
#define GMPECPP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Result of a proof attempt.
 *
 * Note that the command line tool collapses the last three of these onto
 * exit code 2, which cannot distinguish "this number is composite" from
 * "the prover gave up".  The library reports them separately.
 */
typedef enum {
  GMPECPP_PROVEN_PRIME = 0,     /* full certificate chain completed      */
  GMPECPP_PROBABLY_PRIME = 1,   /* one_step was set; chain not completed */
  GMPECPP_COMPOSITE = 2,        /* proved composite                      */
  GMPECPP_OUT_OF_DISCRIMINANTS = 3, /* gave up: Dmax ceiling reached     */
  GMPECPP_FACTORING_BOUND = 4,  /* gave up: Bmax ceiling reached         */
  GMPECPP_ERR_INVALID_INPUT = 5,/* not a positive integer, or < 2        */
  GMPECPP_ERR_NO_MEMORY = 6,
  GMPECPP_ERR_INTERNAL = 7
} gmpecpp_status_t;

/* Which arithmetic backend serviced the ECM factoring stage. */
typedef enum {
  GMPECPP_BACKEND_CPU = 0,  /* GMP on the host                      */
  GMPECPP_BACKEND_CUDA = 1  /* CGBN on the GPU, many curves at once */
} gmpecpp_backend_t;

typedef struct gmpecpp_config gmpecpp_config_t;
typedef struct gmpecpp_certificate gmpecpp_certificate_t;

/*
 * One link of the certificate.  All numbers are NUL-terminated decimal
 * strings owned by the certificate; they stay valid until it is freed.
 *
 * The step asserts: on the curve y^2 = x^3 + a*x + b over Z/nZ, the point
 * P has order divisible by q, because [m/q]P = p2 != O while [q]p2 == O.
 * With q > (n^(1/4) + 1)^2 that proves n prime -- provided q is prime,
 * which the next step in turn establishes.
 */
typedef struct {
  const char* n;   /* the number this step proves prime      */
  const char* a;   /* curve coefficient a                    */
  const char* b;   /* curve coefficient b                    */
  const char* m;   /* order of the curve                     */
  const char* q;   /* large prime factor of m; next step's n */
  const char* px;  /* P.x   */
  const char* py;  /* P.y   */
  const char* p1x; /* [q]P2 .x -- the point at infinity, 0   */
  const char* p1y; /* [q]P2 .y -- the point at infinity, 1   */
  const char* p2x; /* [m/q]P .x */
  const char* p2y; /* [m/q]P .y */
  long discriminant; /* the D that produced this curve       */
} gmpecpp_step_t;

/* ---------------------------------------------------------------- */
/* Configuration                                                     */
/* ---------------------------------------------------------------- */

GMPECPP_API gmpecpp_config_t* gmpecpp_config_new(void);
GMPECPP_API void gmpecpp_config_free(gmpecpp_config_t* cfg);

/* Floating point working precision in bits (default 1000).
 *
 * This is by far the most important performance knob: the class
 * polynomial layer is arbitrary precision floating point and its cost
 * grows steeply with this value.
 *
 * Too low, and polynomial construction fails a consistency check on the
 * constant coefficient; the discriminant is then abandoned and the sweep
 * moves on, so the failure mode is lost time rather than a wrong answer.
 * If proofs fail on discriminants that ought to work, raise this before
 * suspecting the number theory. */
GMPECPP_API void gmpecpp_config_set_precision(gmpecpp_config_t*, unsigned long bits);

/* Convergence threshold 2^-shift inside the Dedekind eta sum (default 1000). */
GMPECPP_API void gmpecpp_config_set_error_shift(gmpecpp_config_t*, unsigned long shift);

/* Pseudo-random seed.  0 means derive one from the clock. */
GMPECPP_API void gmpecpp_config_set_seed(gmpecpp_config_t*, unsigned long seed);

/* Pinning either bound disables the automatic escalation that otherwise
 * multiplies Bmax by 10 and Dmax by 5 on failure.  A pinned bound that is
 * too small makes a provable number fail rather than retry harder. */
GMPECPP_API void gmpecpp_config_set_bmax(gmpecpp_config_t*, unsigned long bmax);
GMPECPP_API void gmpecpp_config_set_dmax(gmpecpp_config_t*, unsigned long dmax);

/* Select class polynomial families.  Enabling neither enables both. */
GMPECPP_API void gmpecpp_config_set_hilbert(gmpecpp_config_t*, int enable);
GMPECPP_API void gmpecpp_config_set_weber(gmpecpp_config_t*, int enable);

/* Stop after one downrun step; yields GMPECPP_PROBABLY_PRIME. */
GMPECPP_API void gmpecpp_config_set_one_step(gmpecpp_config_t*, int enable);

/* Request a backend for the ECM factoring stage.  Requesting CUDA when it
 * is unavailable falls back to the CPU; call gmpecpp_backend_in_use() on
 * the resulting certificate to see what actually ran. */
GMPECPP_API void gmpecpp_config_set_backend(gmpecpp_config_t*, gmpecpp_backend_t);

/* Number of ECM curves attempted concurrently by the CUDA backend.
 * Ignored by the CPU backend.  0 selects a built-in default. */
GMPECPP_API void gmpecpp_config_set_cuda_curves(gmpecpp_config_t*, unsigned int curves);

/*
 * Smallest cofactor, in bits, worth handing to the GPU (default 192).
 *
 * The GPU kernel costs a fixed amount of wall time regardless of how hard
 * the number is -- it is latency bound on the serial chain of prime
 * powers -- whereas the CPU splits small cofactors in milliseconds.  Most
 * of a proof's ECM calls are on candidates that have no usable large
 * prime factor at all and are meant to fail fast, so handing every failed
 * CPU attempt to the GPU makes proofs slower, not faster.  This threshold
 * keeps the GPU for cases where it can actually win.  Set it to 0 to try
 * the GPU on every failed CPU attempt.
 */
GMPECPP_API void gmpecpp_config_set_cuda_min_bits(gmpecpp_config_t*, unsigned int bits);

/*
 * Worker threads for the ECM factoring stage (default 1).
 *
 * Curves are independent trials and one success ends the search, so this
 * is the one stage of the prover that parallelises cleanly.  Unlike the
 * CUDA backend it carries no per-launch cost, so it is worth enabling at
 * any input size.  0 means "one per online CPU".
 *
 * As with CUDA, more than one thread changes *which* factor is found, so
 * certificates differ from a single-threaded run while remaining valid.
 * Verify them rather than comparing them byte for byte.  Leaving this at
 * 1 reproduces the historical program exactly.
 */
GMPECPP_API void gmpecpp_config_set_threads(gmpecpp_config_t*, unsigned int threads);

/*
 * Worker threads for the discriminant sweep (default 1).
 *
 * Each discriminant is an independent trial and the sweep wants the
 * first that works, so this is speculative parallelism -- and it covers
 * essentially the whole proof, since both the factoring stage and the
 * class polynomial construction sit under it.
 *
 * The sweep stays deterministic: every attempt seeds its own random
 * stream from the discriminant, and the smallest successful discriminant
 * wins regardless of which thread reached it first. A run at sixteen
 * threads therefore reproduces a run at two. It still differs from a
 * single-threaded run, because that one draws from a single continuous
 * random stream instead.
 *
 * Prefer this over gmpecpp_config_set_threads() when both would apply;
 * setting both oversubscribes, since sweep workers each run their own
 * ECM. 0 means one per online CPU.
 */
GMPECPP_API void gmpecpp_config_set_sweep_threads(gmpecpp_config_t*, unsigned int threads);

/*
 * Progress reporting.  The library never writes to stdout or stderr on
 * its own; if you want the running commentary the command line tool
 * prints, install a callback here.  It is called with NUL-terminated
 * lines that do not include a trailing newline.
 */
typedef void (*gmpecpp_log_fn)(const char* message, void* user_data);
GMPECPP_API void gmpecpp_config_set_log(gmpecpp_config_t*, gmpecpp_log_fn, void* user_data);

/*
 * How much commentary reaches the log callback.  With no callback
 * installed nothing is emitted at any level.
 *
 *   0  silent   -- certificate steps and verdicts only (default)
 *   1  progress -- adds the seed/precision banner and Bmax/Dmax sweep
 *   2  normal   -- adds the computed values of PI, E and ln(1.9)
 *   3  verbose  -- adds a per-discriminant progress marker and a dump of
 *                  the first 20 class polynomials before proving
 *
 * These correspond to the command line tool's -Q, -q, (none) and -v.
 */
GMPECPP_API void gmpecpp_config_set_verbosity(gmpecpp_config_t*, int level);

/* ---------------------------------------------------------------- */
/* Proving                                                           */
/* ---------------------------------------------------------------- */

/*
 * Emit the banner, compute the transcendental constants, and (at
 * verbosity 3) dump the leading class polynomials -- everything the
 * prover does before it needs the number itself.
 *
 * Calling this is optional: gmpecpp_prove() does the same work if it has
 * not been done.  It exists so that an interactive caller can get the
 * banner out, and its own prompt, before blocking on input.
 *
 * Computing the constants at the default 10000 bits costs a few tenths
 * of a second.  The result is cached and reused by every subsequent call
 * at the same precision, so proving many numbers pays it once.
 */
GMPECPP_API void gmpecpp_preamble(const gmpecpp_config_t* cfg);

/*
 * Prove primality of the decimal string `n`.
 *
 * On return *cert receives a certificate, which the caller must free with
 * gmpecpp_certificate_free().  It is populated for every status: on
 * GMPECPP_COMPOSITE it may carry the factor that settled the question,
 * and on the give-up statuses it holds the steps completed so far.
 * Passing NULL for cert discards it.
 *
 * cfg may be NULL, meaning "all defaults".
 *
 * This call is thread safe in the sense that concurrent calls will not
 * corrupt each other, but they are serialised internally: the underlying
 * prover keeps global state.  For parallelism, run separate processes.
 */
GMPECPP_API gmpecpp_status_t gmpecpp_prove(const gmpecpp_config_t* cfg,
                                           const char* n,
                                           gmpecpp_certificate_t** cert);

/* As above, for a number already held in a GMP integer.  Declared only
 * when <gmp.h> has been included before this header. */
#ifdef __GMP_H__
GMPECPP_API gmpecpp_status_t gmpecpp_prove_mpz(const gmpecpp_config_t* cfg,
                                               mpz_srcptr n,
                                               gmpecpp_certificate_t** cert);
#endif

/*
 * Attempt to split `n` with one round of ECM at bound b1, honouring the
 * configured backend.  Returns GMPECPP_COMPOSITE having stored a
 * nontrivial factor in *factor (free it with gmpecpp_string_free), or
 * GMPECPP_PROBABLY_PRIME if no factor was found -- ECM failing to split
 * a number is not evidence that it is prime.
 *
 * Exposed mainly so the factoring stage can be measured on its own: it
 * is the one part of the prover the CUDA backend accelerates, and its
 * share of a whole proof is small enough that end-to-end timings alone
 * would not show what the backend is doing.
 */
GMPECPP_API gmpecpp_status_t gmpecpp_ecm_factor(const gmpecpp_config_t* cfg,
                                                const char* n,
                                                unsigned long b1,
                                                char** factor);

/*
 * ECM on the GPU alone: stage 1 to b1, then (if b2 > b1) stage 2 to b2, on
 * `curves` concurrent curves (0 = as many as the GPU runs at once; see
 * gmpecpp_cuda_ecm_curves).  Unlike gmpecpp_ecm_factor() there is no CPU
 * attempt first and no global lock, so several threads may call it at once
 * (each on its own device, or sharing one).  It runs on the calling thread's
 * current CUDA device.
 *
 * The work is cut into launches of about a quarter of a second, with every
 * curve's state kept on the device between them; `stop`, if not NULL, is
 * called between launches, and a non-zero return abandons the search
 * (returning GMPECPP_PROBABLY_PRIME).  That also means b1 is not capped.
 *
 * Returns GMPECPP_COMPOSITE with a nontrivial factor in *factor (free it with
 * gmpecpp_string_free), GMPECPP_PROBABLY_PRIME if no factor was found -- not
 * evidence of primality -- GMPECPP_ERR_INVALID_INPUT for an n that is even,
 * below 5 or wider than 16352 bits, or b1 outside [1000, 2^32), and
 * GMPECPP_ERR_INTERNAL when there is no usable CUDA device or a CUDA call
 * failed.  `seed` picks the curves (0 means 1).
 */
GMPECPP_API gmpecpp_status_t gmpecpp_cuda_ecm(const char* n,
                                              unsigned long b1,
                                              unsigned long b2,
                                              unsigned int curves,
                                              unsigned long seed,
                                              int (*stop)(void* arg),
                                              void* stop_arg,
                                              char** factor);

/* The curve count gmpecpp_cuda_ecm() uses for `n` when asked for 0: as many
 * curves as the current device runs at once at n's width.  0 when there is
 * no device or n is too wide. */
GMPECPP_API unsigned int gmpecpp_cuda_ecm_curves(const char* n);

/* Frees a string returned by this library. */
GMPECPP_API void gmpecpp_string_free(char* s);

/* ---------------------------------------------------------------- */
/* Certificates                                                      */
/* ---------------------------------------------------------------- */

GMPECPP_API void gmpecpp_certificate_free(gmpecpp_certificate_t*);
GMPECPP_API size_t gmpecpp_certificate_num_steps(const gmpecpp_certificate_t*);

/* Fills *out for step i.  Returns 0 on success, -1 if i is out of range. */
GMPECPP_API int gmpecpp_certificate_get_step(const gmpecpp_certificate_t*,
                                             size_t i, gmpecpp_step_t* out);

/* The nontrivial factor found, when the status was GMPECPP_COMPOSITE and
 * one was actually exhibited; NULL otherwise. */
GMPECPP_API const char* gmpecpp_certificate_factor(const gmpecpp_certificate_t*);

/* Which ECM backend actually ran for this proof. */
GMPECPP_API gmpecpp_backend_t gmpecpp_certificate_backend(const gmpecpp_certificate_t*);

/* Seed actually used, useful for reproducing a run that was seeded from
 * the clock. */
GMPECPP_API unsigned long gmpecpp_certificate_seed(const gmpecpp_certificate_t*);

/* Wall clock seconds spent inside gmpecpp_prove(), and of that, the time
 * spent in the ECM factoring stage. */
GMPECPP_API double gmpecpp_certificate_elapsed(const gmpecpp_certificate_t*);
GMPECPP_API double gmpecpp_certificate_ecm_elapsed(const gmpecpp_certificate_t*);

/* ---------------------------------------------------------------- */
/* Capabilities                                                      */
/* ---------------------------------------------------------------- */

/* Non-zero if this build has the CUDA/CGBN backend compiled in AND a
 * usable device is present. */
GMPECPP_API int gmpecpp_have_cuda(void);

/* Human readable device description, or NULL when there is none. */
GMPECPP_API const char* gmpecpp_cuda_device_name(void);

GMPECPP_API const char* gmpecpp_version_string(void);
GMPECPP_API const char* gmpecpp_status_string(gmpecpp_status_t);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GMPECPP_H */
