/*
 * gmpecpp-mpi -- prove a batch of numbers across MPI ranks, with POSIX
 * threads inside each rank.
 *
 * Why this shape.  An ECPP downrun is inherently sequential: step i+1
 * proves the prime that step i produced, so a single proof offers no
 * coarse-grained parallelism to distribute.  What it does offer is two
 * finer axes, and they suit different tools:
 *
 *   - within one proof, ECM curves are independent trials and one
 *     success ends the search.  That is shared-memory work with a tiny
 *     critical section, so POSIX threads fit and MPI would not.
 *
 *   - across proofs, whole numbers are independent.  That is
 *     embarrassingly parallel with no communication until the end, which
 *     is exactly what MPI is for.
 *
 * So this is a hybrid: MPI distributes numbers, threads accelerate each.
 * Work is handed out dynamically rather than block-partitioned, because
 * proof times vary by two orders of magnitude between neighbouring
 * inputs -- a static split would leave most ranks idle behind one
 * straggler.
 *
 * Usage:  mpirun -n R ./gmpecpp-mpi [-T threads] [-C] file
 *         where file holds one number per line ("digits value" also works).
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gmpecpp.h"

enum { TAG_REQUEST = 1, TAG_WORK = 2, TAG_RESULT = 3, TAG_STOP = 4 };
#define MAXNUM 4096

typedef struct {
  int index;
  int status;
  int steps;
  double seconds;
} Result;

static void quiet(const char *m, void *u) { (void)m; (void)u; }

static int read_numbers(const char *path, char (**out)[MAXNUM]) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  int cap = 64, n = 0;
  char (*v)[MAXNUM] = malloc((size_t)cap * MAXNUM);
  char line[MAXNUM];
  while (fgets(line, sizeof line, f)) {
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || !*p) continue;
    /* accept "<digits> <value>" as well as a bare value */
    char *sp = strchr(p, ' ');
    if (sp) { char *q = sp; while (*q == ' ') q++; if (*q >= '0' && *q <= '9') p = q; }
    p[strcspn(p, " \t\r\n")] = '\0';
    if (n == cap) { cap *= 2; v = realloc(v, (size_t)cap * MAXNUM); }
    snprintf(v[n++], MAXNUM, "%s", p);
  }
  fclose(f);
  *out = v;
  return n;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  unsigned threads = 1;
  int use_cuda = 0;
  const char *path = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-T") && i + 1 < argc) threads = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-C")) use_cuda = 1;
    else path = argv[i];
  }
  if (!path) {
    if (rank == 0) fprintf(stderr, "usage: mpirun -n R %s [-T threads] [-C] file\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  char (*nums)[MAXNUM] = NULL;
  int count = 0;
  if (rank == 0) {
    count = read_numbers(path, &nums);
    if (count < 0) { fprintf(stderr, "cannot read %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
  }
  MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) nums = malloc((size_t)count * MAXNUM);
  MPI_Bcast(nums, count * MAXNUM, MPI_CHAR, 0, MPI_COMM_WORLD);

  double t_start = MPI_Wtime();

  if (size == 1) {
    /* Degenerate case: no workers to hand out to, so just do the work. */
    gmpecpp_config_t *cfg = gmpecpp_config_new();
    gmpecpp_config_set_seed(cfg, 1);
    gmpecpp_config_set_verbosity(cfg, 0);
    gmpecpp_config_set_log(cfg, quiet, NULL);
    gmpecpp_config_set_threads(cfg, threads);
    if (use_cuda) gmpecpp_config_set_backend(cfg, GMPECPP_BACKEND_CUDA);
    for (int i = 0; i < count; i++) {
      gmpecpp_certificate_t *c = NULL;
      gmpecpp_status_t s = gmpecpp_prove(cfg, nums[i], &c);
      printf("[%3d] %-4zu digits  %-14s %2zu steps  %7.2fs\n", i, strlen(nums[i]),
             gmpecpp_status_string(s), gmpecpp_certificate_num_steps(c),
             gmpecpp_certificate_elapsed(c));
      gmpecpp_certificate_free(c);
    }
    gmpecpp_config_free(cfg);
  } else if (rank == 0) {
    /* Dynamic dispatch: hand out the next index whenever a rank asks. */
    int next = 0, live = size - 1;
    Result *res = calloc((size_t)count, sizeof *res);
    while (live > 0) {
      MPI_Status st;
      Result r;
      MPI_Recv(&r, sizeof r, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
      if (st.MPI_TAG == TAG_RESULT) res[r.index] = r;
      if (next < count) {
        MPI_Send(&next, 1, MPI_INT, st.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
        next++;
      } else {
        int stop = -1;
        MPI_Send(&stop, 1, MPI_INT, st.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
        live--;
      }
    }
    double busy = 0;
    for (int i = 0; i < count; i++) {
      printf("[%3d] %-4zu digits  %-14s %2d steps  %7.2fs\n", i, strlen(nums[i]),
             gmpecpp_status_string((gmpecpp_status_t)res[i].status),
             res[i].steps, res[i].seconds);
      busy += res[i].seconds;
    }
    double wall = MPI_Wtime() - t_start;
    printf("\n%d numbers over %d ranks x %u threads: wall %.2fs, "
           "summed proof time %.2fs, speedup %.2fx\n",
           count, size - 1, threads, wall, busy, busy / wall);
    free(res);
  } else {
    gmpecpp_config_t *cfg = gmpecpp_config_new();
    gmpecpp_config_set_seed(cfg, 1);
    gmpecpp_config_set_verbosity(cfg, 0);
    gmpecpp_config_set_log(cfg, quiet, NULL);
    gmpecpp_config_set_threads(cfg, threads);
    if (use_cuda) gmpecpp_config_set_backend(cfg, GMPECPP_BACKEND_CUDA);

    Result r;
    memset(&r, 0, sizeof r);
    r.index = -1;
    MPI_Send(&r, sizeof r, MPI_BYTE, 0, TAG_REQUEST, MPI_COMM_WORLD);
    for (;;) {
      int idx;
      MPI_Status st;
      MPI_Recv(&idx, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
      if (st.MPI_TAG == TAG_STOP) break;
      gmpecpp_certificate_t *c = NULL;
      gmpecpp_status_t s = gmpecpp_prove(cfg, nums[idx], &c);
      r.index = idx;
      r.status = (int)s;
      r.steps = (int)gmpecpp_certificate_num_steps(c);
      r.seconds = gmpecpp_certificate_elapsed(c);
      gmpecpp_certificate_free(c);
      MPI_Send(&r, sizeof r, MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
    gmpecpp_config_free(cfg);
  }

  free(nums);
  MPI_Finalize();
  return 0;
}
