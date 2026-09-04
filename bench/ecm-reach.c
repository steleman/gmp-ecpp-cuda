/*
 * ecm-reach -- how large a factor can each backend actually find?
 *
 * ECM's cost depends on the size of the factor it is looking for, not on
 * the size of the number, so this indexes semiprimes by their smaller
 * factor and asks each backend to split them.  It produces the "Reach"
 * table in README.md.
 *
 * The size gate is disabled here so the GPU is always tried; in a real
 * proof it would decline the small cases.  Note that the CUDA column
 * includes the host attempt that precedes it, which is why it looks
 * slower on rows where both succeed -- the GPU is not reached there.
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

static gmpecpp_config_t *cfg(gmpecpp_backend_t be, unsigned curves) {
  gmpecpp_config_t *c = gmpecpp_config_new();
  gmpecpp_config_set_backend(c, be);
  gmpecpp_config_set_cuda_curves(c, curves);
  gmpecpp_config_set_cuda_min_bits(c, 0); /* always try the GPU */
  gmpecpp_config_set_seed(c, 1);
  gmpecpp_config_set_verbosity(c, 0);
  return c;
}

int main(int argc, char **argv) {
  unsigned curves = 4096;
  unsigned long b1 = 50000;
  const char *path = "bench/semiprimes.txt";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-c") && i + 1 < argc) curves = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-b") && i + 1 < argc) b1 = strtoul(argv[++i], NULL, 10);
    else path = argv[i];
  }

  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

  printf("%s\n", gmpecpp_version_string());
  printf("CUDA: %s\n", gmpecpp_have_cuda() ? gmpecpp_cuda_device_name() : "unavailable");
  printf("B1=%lu  curves=%u  (size gate disabled)\n\n", b1, curves);
  printf("%-8s %-12s %11s %11s\n", "factor", "found by", "CPU", "CUDA");

  char line[1024];
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    int d; char n[512], fac[256];
    if (sscanf(line, "%d %511s %255s", &d, n, fac) != 3) continue;

    char *f1 = NULL, *f2 = NULL;
    gmpecpp_config_t *c1 = cfg(GMPECPP_BACKEND_CPU, 0);
    gmpecpp_config_t *c2 = cfg(GMPECPP_BACKEND_CUDA, curves);

    double t0 = now();
    gmpecpp_status_t s1 = gmpecpp_ecm_factor(c1, n, b1, &f1);
    double tc = now() - t0;
    t0 = now();
    gmpecpp_status_t s2 = gmpecpp_ecm_factor(c2, n, b1, &f2);
    double tg = now() - t0;

    const char *who = (s1 == GMPECPP_COMPOSITE && s2 == GMPECPP_COMPOSITE) ? "both"
                    : (s2 == GMPECPP_COMPOSITE) ? "CUDA only"
                    : (s1 == GMPECPP_COMPOSITE) ? "CPU only" : "neither";
    printf("%-8d %-12s %10.3fs %10.3fs\n", d, who, tc, tg);
    fflush(stdout);

    gmpecpp_string_free(f1); gmpecpp_string_free(f2);
    gmpecpp_config_free(c1); gmpecpp_config_free(c2);
  }
  fclose(f);
  return 0;
}
