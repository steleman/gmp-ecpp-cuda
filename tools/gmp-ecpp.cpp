/*
 * gmp-ecpp -- command line front end for libgmpecpp.
 *
 * This is the original program's main(), reimplemented against the
 * library's public API.  It exists both as the shipped tool and as
 * evidence that the API is sufficient to reproduce the original
 * behavior, output and exit codes exactly.
 */

#include "gmpecpp.h"
#include <getopt.h>
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *program_name;

static void print_usage(FILE *stream, int exit_code) {
  (void) fprintf(stream, "Usage: %s options [ < inputfile ]\n", program_name);
  (void) fprintf(
      stream,
      "	-h	--help			Display this usage information.\n"
      "	-Q	--Quiet			Print very reduced messages.\n"
      "	-q	--quiet			Print reduced messages.\n"
      "	-v	--verbose		Print verbose messages (including "
      "progress marker).\n"
      "	-o	--one			Run just one iteration before "
      "termination.\n"
      "	-H	--Hilbert		Use Hilbert polynomials (only).\n"
      "	-W	--Weber			Use Weber polynomials (only).\n"
      "	-s	--seed			Set (pseudo-)random seed.\n"
      "	-p	--precision		Set GMP FP precision (default 1000).\n"
      "	-e	--error_shift	Set precision in CP generation (default "
      "1000).\n"
      "	-B	--Bmax			Set max ECM factoring bound (default "
      "2000 autoincrement).\n"
      "	-D	--Dmax			Set max discriminant bound (default 20 "
      "autoincrement).\n"
      "	-C	--cuda			Use the CUDA/CGBN backend for ECM "
      "factoring.\n"
      "	-c	--curves		Concurrent ECM curves for the CUDA "
      "backend.\n");
  exit(exit_code);
}

/* The library hands back whole lines; the original wrote them to stdout. */
static void emit(const char *line, void *) {
  (void) fputs(line, stdout);
  (void) fputc('\n', stdout);
  (void) fflush(stdout);
}

int main(int argc, char *argv[]) {
  int next_option;
  int Quiet = 0, quiet = 0, verbose = 0;
  int use_cuda = 0;
  unsigned int curves = 0;

  const char *const short_options = "hQqvoHWs:p:e:B:D:Cc:";
  const struct option long_options[] = {
      {"help", 0, NULL, 'h'},      {"Quiet", 0, NULL, 'Q'},
      {"quiet", 0, NULL, 'q'},     {"verbose", 0, NULL, 'v'},
      {"one", 0, NULL, 'o'},       {"Hilbert", 0, NULL, 'H'},
      {"Weber", 0, NULL, 'W'},     {"seed", 1, NULL, 's'},
      {"precision", 1, NULL, 'p'}, {"error_shift", 1, NULL, 'e'},
      {"Bmax", 1, NULL, 'B'},      {"Dmax", 1, NULL, 'D'},
      {"cuda", 0, NULL, 'C'},      {"curves", 1, NULL, 'c'},
      {NULL, 0, NULL, 0}};

  program_name = argv[0];

  gmpecpp_config_t *cfg = gmpecpp_config_new();
  if (!cfg) {
    fprintf(stderr, "out of memory\n");
    return 7;
  }

  do {
    next_option = getopt_long(argc, argv, short_options, long_options, NULL);
    switch (next_option) {
    case 'h':
      print_usage(stdout, 0);
      break;
    case 'Q':
      Quiet = 1;
      break;
    case 'q':
      quiet = 1;
      break;
    case 'v':
      verbose = 1;
      break;
    case 'o':
      gmpecpp_config_set_one_step(cfg, 1);
      break;
    case 'H':
      gmpecpp_config_set_hilbert(cfg, 1);
      break;
    case 'W':
      gmpecpp_config_set_weber(cfg, 1);
      break;
    case 's':
      gmpecpp_config_set_seed(cfg, strtoul(optarg, NULL, 10));
      break;
    case 'p':
      gmpecpp_config_set_precision(cfg, strtoul(optarg, NULL, 10));
      break;
    case 'e':
      gmpecpp_config_set_error_shift(cfg, strtoul(optarg, NULL, 10));
      break;
    case 'B':
      gmpecpp_config_set_bmax(cfg, strtoul(optarg, NULL, 10));
      break;
    case 'D':
      gmpecpp_config_set_dmax(cfg, strtoul(optarg, NULL, 10));
      break;
    case 'C':
      use_cuda = 1;
      break;
    case 'c':
      curves = (unsigned)strtoul(optarg, NULL, 10);
      break;
    case '?':
      print_usage(stderr, 1);
      break;
    case -1:
      break;
    default:
      abort();
    }
  } while (next_option != -1);

  /* -Q, -q, neither, -v map onto the library's four verbosity levels. */
  gmpecpp_config_set_verbosity(cfg, Quiet ? 0 : verbose ? 3 : quiet ? 1 : 2);
  gmpecpp_config_set_log(cfg, emit, NULL);
  if (use_cuda) {
    gmpecpp_config_set_backend(cfg, GMPECPP_BACKEND_CUDA);
    gmpecpp_config_set_cuda_curves(cfg, curves);
  }

  /* Emit the banner and constants first, exactly as the original did,
   * then prompt, then read.  Without this split the prompt would appear
   * before the banner.
   */
  gmpecpp_preamble(cfg);

  if (!Quiet && !quiet) {
    (void) fprintf(stdout, "number to be tested:\n");
    (void) fflush(stdout);
  }

  char buf[65536];
  if (!fgets(buf, sizeof buf, stdin)) {
    gmpecpp_config_free(cfg);
    return 5;
  }

  buf[strcspn(buf, "\r\n")] = '\0';

  gmpecpp_certificate_t *cert = NULL;
  gmpecpp_status_t st = gmpecpp_prove(cfg, buf, &cert);

  if (st == GMPECPP_PROVEN_PRIME)
    (void) fprintf(stdout, "proven prime\n");
  else if (st == GMPECPP_COMPOSITE || st == GMPECPP_OUT_OF_DISCRIMINANTS ||
           st == GMPECPP_FACTORING_BOUND)
    (void) fprintf(stdout, "composite\n");
  (void) fflush(stdout);

  /* Exit codes are the original's: 0 proven, 1 probably prime, 2 both
   * composite and gave-up.  The library distinguishes the last two; the
   * tool deliberately does not, for compatibility.
   */
  int rc;
  switch (st) {
  case GMPECPP_PROVEN_PRIME:
    rc = 0;
    break;
  case GMPECPP_PROBABLY_PRIME:
    rc = 1;
    break;
  default:
    rc = 2;
    break;
  }

  gmpecpp_certificate_free(cert);
  gmpecpp_config_free(cfg);
  return rc;
}

