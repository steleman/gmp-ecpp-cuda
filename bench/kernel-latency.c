/*
 * kernel-latency -- is the GPU ECM kernel latency bound or throughput
 * bound?
 *
 * Sweeps the stage-one bound against the number of concurrent curves and
 * reports the GPU's marginal cost, obtained as the difference between a
 * CUDA run and a CPU-only run of the same work.  Everything goes through
 * the public API, so this measures what a caller would actually pay.
 *
 * Produces the kernel cost table in README.md and doc/gmpecpp.tex.  The
 * expected shape is cost tracking B1 and barely moving with curve count:
 * curves are nearly free, which is what makes ECM worth putting on a GPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gmpecpp.h"

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* A semiprime the host cannot split at these bounds, so that both
 * backends run to completion and we time the full attempt rather than an
 * early success.  Read from the data file rather than pasted inline: an
 * earlier version hardcoded a truncated literal, which the host split
 * instantly, and the GPU was never reached at all. */
static char HARD[1024];

static int load_hard(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char line[1024];
  int best = -1;
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    int d; char n[512], fac[256];
    if (sscanf(line, "%d %511s %255s", &d, n, fac) != 3) continue;
    if (d > best) { best = d; snprintf(HARD, sizeof HARD, "%s", n); }
  }
  fclose(f);
  if (best > 0)
    printf("hardest available: %d-digit factor, %zu-digit modulus\n\n",
           best, strlen(HARD));
  return best > 0;
}

static double run(gmpecpp_backend_t be, unsigned curves, unsigned long b1) {
  gmpecpp_config_t *c = gmpecpp_config_new();
  gmpecpp_config_set_backend(c, be);
  gmpecpp_config_set_cuda_curves(c, curves);
  gmpecpp_config_set_cuda_min_bits(c, 0);
  gmpecpp_config_set_seed(c, 1);
  gmpecpp_config_set_verbosity(c, 0);
  char *f = NULL;
  double t0 = now();
  gmpecpp_ecm_factor(c, HARD, b1, &f);
  double dt = now() - t0;
  gmpecpp_string_free(f);
  gmpecpp_config_free(c);
  return dt;
}

int main(int argc, char **argv) {
  const char *data = argc > 1 ? argv[1] : "bench/semiprimes.txt";
  static const unsigned long B1[] = {100, 1000, 10000, 50000};
  static const unsigned CURVES[] = {256, 1024, 4096};

  printf("%s\n", gmpecpp_version_string());
  printf("CUDA: %s\n\n", gmpecpp_have_cuda() ? gmpecpp_cuda_device_name() : "unavailable");
  if (!load_hard(data)) {
    fprintf(stderr, "cannot read %s\n", data);
    return 1;
  }
  if (!gmpecpp_have_cuda()) {
    printf("No GPU; the columns below would all be host time.\n");
    return 0;
  }

  /* Warm up: the first CUDA call in a process pays context creation,
     about 0.1s, which would otherwise land in the first cell. */
  run(GMPECPP_BACKEND_CUDA, 256, 100);

  printf("GPU marginal cost = (CUDA run) - (CPU-only run), same work.\n\n");
  printf("%10s %12s %12s %12s\n", "B1", "256 curves", "1024 curves", "4096 curves");
  for (size_t i = 0; i < sizeof B1 / sizeof B1[0]; i++) {
    double base = run(GMPECPP_BACKEND_CPU, 0, B1[i]);
    printf("%10lu", B1[i]);
    for (size_t j = 0; j < sizeof CURVES / sizeof CURVES[0]; j++) {
      double t = run(GMPECPP_BACKEND_CUDA, CURVES[j], B1[i]) - base;
      printf(" %11.3fs", t < 0 ? 0.0 : t);
      fflush(stdout);
    }
    printf("\n");
  }
  return 0;
}
