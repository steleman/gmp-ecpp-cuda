/*
 * gmpecmfft -- GPU ECM factoring of one number, any size: CGBN up to
 * 16352 bits, cuFFT arithmetic above. A client of libgmpecmfft.
 *
 *   echo N | gmpecmfft [-1 B1] [-2 B2] [-c curves] [-s seed] [-F] [-v]
 *   gmpecmfft --selftest K N
 *
 * Copyright (c) 2026 Stefan Teleman and contributors.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.  See <http://www.gnu.org/licenses/>.
 */

#include <ctype.h>
#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <gmp.h>

#include "gmpecmfft.h"

static volatile sig_atomic_t interrupted = 0;
static const char *stop_file = NULL;

static void on_signal(int sig) {
  (void)sig;
  interrupted = 1;
}

/* Polled by the library about every quarter second. */
static int stop_requested(void *arg) {
  (void)arg;
  return interrupted || (stop_file && access(stop_file, F_OK) == 0);
}

static void usage(FILE *f) {
  fprintf(f,
          "usage: gmpecmfft [options] [N]\n"
          "  N is read from stdin when not given on the command line.\n"
          "\n"
          "  -1, --b1 B1        stage 1 bound (default 10000)\n"
          "  -2, --b2 B2        stage 2 bound (default 100*B1 below B1 = "
          "50000,\n"
          "                     50*B1 from there; 0 = stage 1 only)\n"
          "  -c, --curves C     number of curves (default 0 = automatic)\n"
          "  -s, --seed S       curve seed (default 1)\n"
          "  -F, --fft          use the FFT arithmetic even below %d bits\n"
          "  -v, --verbose      per-stage statistics on stderr\n"
          "  -t, --selftest K   check K rounds of FFT arithmetic mod N "
          "against GMP\n"
          "      --sigmas K     print the first K curve sigmas for the seed "
          "and exit\n"
          "                     (GMP-ECM's -sigma 0:S builds the same curve)\n"
          "      --stop-file F  stop (exit 2) as soon as file F exists; "
          "checked\n"
          "                     about every quarter second (for job arrays)\n"
          "  -h, --help         this text\n"
          "\n"
          "Exit status: 0 factor found (printed on stdout), 1 none found,\n"
          "2 interrupted, 3 invalid input, 4 CUDA error, 5 FFT precision,\n"
          "6 out of GPU memory, 7 internal error.\n",
          GMPECMFFT_CGBN_MAX_BITS);
}

/* Reads one number from stdin, any length; strips whitespace. */
static char *read_number(void) {
  size_t cap = 4096, len = 0;
  char *s = malloc(cap);
  int ch;
  if (!s)
    return NULL;
  while ((ch = getchar()) != EOF) {
    if (isspace(ch)) {
      if (len)
        break;
      continue;
    }
    if (len + 1 >= cap) {
      char *t = realloc(s, cap *= 2);
      if (!t) {
        free(s);
        return NULL;
      }
      s = t;
    }
    s[len++] = (char)ch;
  }
  s[len] = 0;
  return s;
}

int main(int argc, char **argv) {
  static const struct option opts[] = {
      {"b1", required_argument, 0, '1'},
      {"b2", required_argument, 0, '2'},
      {"curves", required_argument, 0, 'c'},
      {"seed", required_argument, 0, 's'},
      {"fft", no_argument, 0, 'F'},
      {"verbose", no_argument, 0, 'v'},
      {"selftest", required_argument, 0, 't'},
      {"help", no_argument, 0, 'h'},
      {"sigmas", required_argument, 0, 'S'},
      {"stop-file", required_argument, 0, 'X'},
      {0, 0, 0, 0}};
  unsigned long b1 = 10000, b2 = 0, seed = 1;
  int have_b2 = 0, force_fft = 0, verbose = 0;
  unsigned curves = 0, selftest = 0, sigmas = 0;
  int c;
  while ((c = getopt_long(argc, argv, "1:2:c:s:Fvt:h", opts, NULL)) != -1) {
    switch (c) {
    case '1':
      b1 = (unsigned long)strtod(optarg, NULL);
      break;
    case '2':
      b2 = (unsigned long)strtod(optarg, NULL);
      have_b2 = 1;
      break;
    case 'c':
      curves = (unsigned)strtoul(optarg, NULL, 10);
      break;
    case 's':
      seed = strtoul(optarg, NULL, 10);
      break;
    case 'F':
      force_fft = 1;
      break;
    case 'v':
      verbose = 1;
      break;
    case 't':
      selftest = (unsigned)strtoul(optarg, NULL, 10);
      break;
    case 'h':
      usage(stdout);
      return 0;
    case 'S':
      sigmas = (unsigned)strtoul(optarg, NULL, 10);
      break;
    case 'X':
      stop_file = optarg;
      break;
    default:
      usage(stderr);
      return GMPECMFFT_ERR_INVALID_INPUT;
    }
  }
  if (!have_b2)
    b2 = (b1 >= 50000 ? 50 : 100) * b1;

  if (sigmas) {
    /* As the library draws them: sigma = 6 + random(2^30). */
    gmp_randstate_t st;
    gmp_randinit_default(st);
    gmp_randseed_ui(st, seed ? seed : 1);
    for (unsigned i = 0; i < sigmas; i++)
      printf("%lu\n", 6 + gmp_urandomm_ui(st, 1ul << 30));
    gmp_randclear(st);
    return 0;
  }

  char *n = optind < argc ? strdup(argv[optind]) : read_number();
  if (!n || !*n) {
    usage(stderr);
    free(n);
    return GMPECMFFT_ERR_INVALID_INPUT;
  }

  if (selftest) {
    double err = 0;
    int bad = gmpecmfft_selftest(n, selftest, curves, seed, &err);
    free(n);
    if (bad < 0) {
      fprintf(stderr, "gmpecmfft: selftest could not run\n");
      return GMPECMFFT_ERR_CUDA;
    }
    printf("selftest: %d wrong results, worst rounding error %.3g\n", bad,
           err);
    return bad ? GMPECMFFT_ERR_INTERNAL : 0;
  }

  if (verbose) {
    setenv("GMPECMFFT_STATS", "1", 1);
    setenv("GMPECPP_CUDA_ECM_STATS", "1", 1);
  }
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  char *f = NULL;
  gmpecmfft_status_t st =
      (force_fft ? gmpecmfft_ecm_fft : gmpecmfft_ecm)(
          n, b1, b2, curves, seed, stop_requested, NULL, &f);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  const double secs =
      (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  if (st == GMPECMFFT_FACTOR_FOUND)
    printf("%s\n", f);
  else if (st != GMPECMFFT_NO_FACTOR)
    fprintf(stderr, "gmpecmfft: %s\n", gmpecmfft_status_string(st));
  if (verbose)
    fprintf(stderr, "gmpecmfft: %s in %.2f s (B1 %lu, B2 %lu)\n",
            gmpecmfft_status_string(st), secs, b1, b2);
  gmpecmfft_string_free(f);
  free(n);
  return (int)st;
}
