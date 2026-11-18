/*
 * gmpecmfft.h -- public API for libgmpecmfft, GPU ECM factoring at any
 * size: CGBN (through libgmpecpp) up to 16352 bits, cuFFT-based modular
 * arithmetic above that.
 *
 * Copyright (c) 2026 Stefan Teleman and contributors.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.  See <http://www.gnu.org/licenses/>.
 */

#ifndef GMPECMFFT_H
#define GMPECMFFT_H

#define GMPECMFFT_VERSION_MAJOR 1
#define GMPECMFFT_VERSION_MINOR 0
#define GMPECMFFT_VERSION_PATCH 0

/* Widest n the CGBN kernels take; gmpecmfft_ecm() uses FFT above it. */
#define GMPECMFFT_CGBN_MAX_BITS 16352

#if defined(__GNUC__) && __GNUC__ >= 4
#define GMPECMFFT_API __attribute__((visibility("default")))
#else
#define GMPECMFFT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Also the exit codes of the gmpecmfft command line tool. */
typedef enum {
  GMPECMFFT_FACTOR_FOUND = 0,      /* *factor holds a proper divisor   */
  GMPECMFFT_NO_FACTOR = 1,         /* not evidence of primality        */
  GMPECMFFT_STOPPED = 2,           /* stop() asked to abandon the run  */
  GMPECMFFT_ERR_INVALID_INPUT = 3, /* n even or < 5, b1 out of range   */
  GMPECMFFT_ERR_CUDA = 4,          /* no usable device, or a CUDA or
                                      cuFFT call failed                */
  GMPECMFFT_ERR_PRECISION = 5,     /* FFT rounding error too large: the
                                      arithmetic could not be trusted  */
  GMPECMFFT_ERR_NO_MEMORY = 6,     /* too many curves for the device   */
  GMPECMFFT_ERR_INTERNAL = 7       /* an exactness check failed (bug)  */
} gmpecmfft_status_t;

/*
 * ECM on the current CUDA device: stage 1 to b1 and, when b2 > b1, stage
 * 2 to b2. n is a decimal string; it must be odd and at least 5. b1 must
 * be in [1000, 2^32); b2 is capped at 2^32 - 1.
 *
 * Up to GMPECMFFT_CGBN_MAX_BITS bits this is gmpecpp_cuda_ecm() (CGBN,
 * libgmpecpp); above, every curve's arithmetic is done with cuFFT. Both
 * use Suyama's parametrisation with sigma = 6 + random(2^30) drawn from
 * `seed` (0 means 1) the same way, so the same seed gives the same
 * curves on either path.
 *
 * curves = 0 picks a default (gmpecmfft_ecm_curves()). stop, if not
 * NULL, is polled about every quarter second; a non-zero return
 * abandons the run with GMPECMFFT_STOPPED.
 *
 * On GMPECMFFT_FACTOR_FOUND, *factor (if factor is not NULL) is a proper
 * divisor of n as a decimal string; free it with gmpecmfft_string_free().
 *
 * GMPECMFFT_STATS=1 in the environment prints per-stage hit counts and
 * times on stderr (and runs stage 2 even after a stage-1 find, so the
 * counts are cumulative), plus the FFT geometry and rounding error.
 */
GMPECMFFT_API gmpecmfft_status_t
gmpecmfft_ecm(const char *n, unsigned long b1, unsigned long b2,
              unsigned int curves, unsigned long seed, int (*stop)(void *),
              void *stop_arg, char **factor);

/* As gmpecmfft_ecm(), but always on the FFT path, whatever n's size.
 * For testing and comparison against CGBN on the same curves. */
GMPECMFFT_API gmpecmfft_status_t
gmpecmfft_ecm_fft(const char *n, unsigned long b1, unsigned long b2,
                  unsigned int curves, unsigned long seed,
                  int (*stop)(void *), void *stop_arg, char **factor);

/* The curve count gmpecmfft_ecm() uses for n when asked for 0 (0 if n is
 * invalid or there is no device). With fft != 0, the FFT path's count
 * regardless of n's size. */
GMPECMFFT_API unsigned int gmpecmfft_ecm_curves(const char *n, int fft);

/*
 * Checks the FFT arithmetic mod n against GMP: `rounds` rounds of
 * multiplications, squarings, additions and subtractions on random
 * operands, for `curves` residues at a time (0: the default); and, for n
 * of up to 100000 bits, stage 2's polynomial arithmetic (products,
 * product and remainder trees, Newton inverses, remainders). Returns the
 * number of wrong results (0 is a pass), or -1 if the arithmetic could
 * not be set up. *max_err, if not NULL, receives the worst FFT rounding
 * error seen (anything near 0.5 would mean results cannot be trusted).
 */
GMPECMFFT_API int gmpecmfft_selftest(const char *n, unsigned int rounds,
                                     unsigned int curves, unsigned long seed,
                                     double *max_err);

GMPECMFFT_API const char *gmpecmfft_status_string(gmpecmfft_status_t s);
GMPECMFFT_API void gmpecmfft_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif /* GMPECMFFT_H */
