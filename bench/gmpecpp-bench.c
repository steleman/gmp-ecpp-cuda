/*
 * gmpecpp-bench -- measure libgmpecpp with and without the CUDA/CGBN
 * backend.
 *
 * Two things are measured, because they answer different questions:
 *
 *   ecm    the factoring stage on its own.  This is what CGBN actually
 *          accelerates, so it is where a speedup should show up.
 *
 *   prove  whole proofs.  This is what a user experiences, and it is
 *          bounded by how much of a proof the factoring stage represents
 *          -- which, for this prover, is not much.  Class polynomial
 *          construction is arbitrary precision *floating point* and CGBN
 *          is fixed width *integer*, so the dominant stage cannot be
 *          moved to the GPU at all.
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

/* Semiprimes whose smaller factor is of the stated size, for the ECM
 * benchmark.  ECM's cost depends on the size of the factor it is looking
 * for, not on the size of the number. */
static const struct {
  const char *n;
  const char *note;
} SEMIPRIMES[] = {
    {"1050809356352027836051", "11-digit factor"},
    {"4951760154835678088235319297", "10-digit factor"},
    {"999999000001000002999997000003", "12-digit factor"},
    {"1000000000000000012000000000000000027", "19-digit factor"},
};
static const int NSEMI = (int)(sizeof SEMIPRIMES / sizeof SEMIPRIMES[0]);

/* Primes spanning the range where the backends actually differ.
 *
 * The small ones are cheap and show the CUDA path costing nothing when it
 * is not needed.  The large ones are where the ECM stage dominates and
 * the GPU's thousands of concurrent curves can beat the CPU's twenty. */
static const struct {
  int digits;
  const char *n;
} PRIMES[] = {
    {20, "10000000000000000051"},
    {30, "100000000000000000000000000319"},
    {40, "1000000000000000000000000000000000000003"},
    {60, "100000000000000000000000000000000000000000000000000000000019"},
    {90, "850978769952618706291582893634341306902337029020791620185003731259211483772535233535392787"},
    {110, "13758869916262725973770761077076259686486928955706059785087763925515730455172862025624665316585541241687773293"},
    {140, "92029981302684663743473904934467692603279061032390677150857713906779421197715541285175611071400443375599009492234404616465821679975933404951"},
};
static const int NPRIME = (int)(sizeof PRIMES / sizeof PRIMES[0]);

static gmpecpp_config_t *make_cfg(gmpecpp_backend_t be, unsigned curves,
                                  unsigned long precision) {
  gmpecpp_config_t *c = gmpecpp_config_new();
  gmpecpp_config_set_backend(c, be);
  gmpecpp_config_set_cuda_curves(c, curves);
  gmpecpp_config_set_seed(c, 1);
  gmpecpp_config_set_verbosity(c, 0);
  if (precision)
    gmpecpp_config_set_precision(c, precision);
  return c;
}

static void bench_ecm(unsigned curves, unsigned long b1) {
  printf("\n== ECM factoring stage (B1=%lu, CUDA curves=%u) ==\n", b1, curves);
  printf("%-32s %10s %10s %9s   %s\n", "number", "CPU", "CUDA", "speedup",
         "factor agreed");
  for (int i = 0; i < NSEMI; i++) {
    char *f_cpu = NULL, *f_gpu = NULL;
    gmpecpp_config_t *cpu = make_cfg(GMPECPP_BACKEND_CPU, 0, 0);
    gmpecpp_config_t *gpu = make_cfg(GMPECPP_BACKEND_CUDA, curves, 0);

    double t0 = now();
    gmpecpp_status_t s1 = gmpecpp_ecm_factor(cpu, SEMIPRIMES[i].n, b1, &f_cpu);
    double t_cpu = now() - t0;

    t0 = now();
    gmpecpp_status_t s2 = gmpecpp_ecm_factor(gpu, SEMIPRIMES[i].n, b1, &f_gpu);
    double t_gpu = now() - t0;

    char label[40];
    snprintf(label, sizeof label, "%.28s", SEMIPRIMES[i].n);
    printf("%-32s %9.3fs %9.3fs %8.1fx   %s\n", label, t_cpu, t_gpu,
           t_gpu > 0 ? t_cpu / t_gpu : 0.0,
           (s1 == GMPECPP_COMPOSITE && s2 == GMPECPP_COMPOSITE && f_cpu &&
            f_gpu && strcmp(f_cpu, f_gpu) == 0)
               ? "yes"
               : (s1 == GMPECPP_COMPOSITE && s2 == GMPECPP_COMPOSITE)
                     ? "both found (differ)"
                     : "NOT BOTH FOUND");
    gmpecpp_string_free(f_cpu);
    gmpecpp_string_free(f_gpu);
    gmpecpp_config_free(cpu);
    gmpecpp_config_free(gpu);
  }
}

static void bench_prove(unsigned curves, unsigned long precision) {
  if (precision)
    printf("\n== whole proofs (precision=%lu) ==\n", precision);
  else
    printf("\n== whole proofs (library default precision) ==\n");
  printf("%-8s %10s %10s %9s  %10s %10s\n", "digits", "CPU", "CUDA", "speedup",
         "CPU ecm", "CUDA ecm");
  for (int i = 0; i < NPRIME; i++) {
    gmpecpp_certificate_t *c1 = NULL, *c2 = NULL;
    gmpecpp_config_t *cpu = make_cfg(GMPECPP_BACKEND_CPU, 0, precision);
    gmpecpp_config_t *gpu = make_cfg(GMPECPP_BACKEND_CUDA, curves, precision);

    gmpecpp_status_t s1 = gmpecpp_prove(cpu, PRIMES[i].n, &c1);
    gmpecpp_status_t s2 = gmpecpp_prove(gpu, PRIMES[i].n, &c2);

    double t1 = gmpecpp_certificate_elapsed(c1);
    double t2 = gmpecpp_certificate_elapsed(c2);
    printf("%-8d %9.3fs %9.3fs %8.2fx  %9.3fs %9.3fs%s\n", PRIMES[i].digits,
           t1, t2, t2 > 0 ? t1 / t2 : 0.0,
           gmpecpp_certificate_ecm_elapsed(c1),
           gmpecpp_certificate_ecm_elapsed(c2),
           (s1 == GMPECPP_PROVEN_PRIME && s2 == GMPECPP_PROVEN_PRIME)
               ? ""
               : "   <-- NOT BOTH PROVEN");
    gmpecpp_certificate_free(c1);
    gmpecpp_certificate_free(c2);
    gmpecpp_config_free(cpu);
    gmpecpp_config_free(gpu);
  }
}

int main(int argc, char **argv) {
  unsigned curves = 1024;
  unsigned long b1 = 10000, precision = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-c") && i + 1 < argc)
      curves = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-b") && i + 1 < argc)
      b1 = strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-p") && i + 1 < argc)
      precision = strtoul(argv[++i], NULL, 10);
    else {
      fprintf(stderr, "usage: %s [-c curves] [-b B1] [-p precision]\n", argv[0]);
      return 1;
    }
  }

  printf("%s\n", gmpecpp_version_string());
  printf("CUDA available: %s\n",
         gmpecpp_have_cuda() ? gmpecpp_cuda_device_name() : "no");
  if (!gmpecpp_have_cuda())
    printf("  (CUDA columns below will fall back to the CPU path)\n");

  bench_ecm(curves, b1);
  bench_prove(curves, precision);
  printf("\n");
  return 0;
}
