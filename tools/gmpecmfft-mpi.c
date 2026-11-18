/*
 * gmpecmfft-mpi -- GPU ECM on one number across MPI ranks, one GPU per
 * rank: libgmpecmfft's curves handed out in batches, dynamically.
 *
 *   mpirun -n R gmpecmfft-mpi [-1 B1] [-2 B2] [-c curves] [-s seed]
 *                             [-b batches] [-F] [-v] [N]
 *
 * Why this shape.  ECM curves are independent trials, and the first one
 * to find a factor ends the search. Nothing but the number, the bounds
 * and which curves to run has to cross the network, so curves are the
 * unit to distribute -- not the arithmetic of one curve, which would be
 * communication bound and which fits on one GPU anyway. Expected time to
 * a factor falls in proportion to the curve rate, so in proportion to
 * the GPUs, until the cluster runs the expected number of curves for that
 * factor size in one batch each.
 *
 * Rank 0 coordinates and uses no GPU; every other rank is a worker on
 * one GPU (its index among the workers on its node, modulo the node's
 * GPU count). A batch is one gmpecmfft_ecm() call: batch i draws its
 * curves from seed S + i, so batches are distinct, and any batch can be
 * re-run on one GPU with `gmpecmfft -s S+i -c C` (batch 0 is exactly
 * `gmpecmfft -s S`). Work is handed out on request -- a worker's result
 * is its request for the next batch -- so a fast worker never waits for a
 * slow one. Use the same kind of GPU throughout: with a mix, the faster
 * GPUs finish their batches and then sit idle at the end while the slower
 * ones complete theirs.
 *
 * When a factor turns up (or the batches run out, or on SIGINT) the
 * coordinator sends every worker one STOP. Busy workers notice it through
 * the library's stop callback, polled about every quarter second, with
 * MPI_Iprobe; every worker then sends one last result and receives that
 * STOP, so no message is left unmatched.
 *
 * Exit status as gmpecmfft's: 0 factor found (printed on stdout), 1 none
 * found, 2 interrupted, 3 invalid input, 4 CUDA error (every worker
 * failed), and so on.
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

#include <cuda_runtime_api.h>
#include <mpi.h>

#include "gmpecmfft.h"

enum { TAG_RESULT = 1, TAG_FACTOR = 2, TAG_WORK = 3, TAG_STOP = 4 };

/* Worker -> coordinator, after each batch (batch < 0: the first
 * request). The factor, if any, follows as a TAG_FACTOR string. */
typedef struct {
  long batch;
  int status;      /* gmpecmfft_status_t */
  unsigned curves; /* curves in the batch */
  int device;      /* CUDA device of the worker */
  double seconds;
  int factor_len; /* bytes of the TAG_FACTOR message, 0 if none */
} Result;

static volatile sig_atomic_t interrupted = 0;

static void on_signal(int sig) {
  (void)sig;
  interrupted = 1;
}

/* Worker's stop callback: a STOP from the coordinator is waiting (left
 * in the queue for the main loop), or this process was interrupted. */
static int stop_requested(void *arg) {
  (void)arg;
  int flag = 0;
  MPI_Iprobe(0, TAG_STOP, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
  return flag || interrupted;
}

static void usage(FILE *f) {
  fprintf(f,
          "usage: mpirun -n R gmpecmfft-mpi [options] [N]\n"
          "  Rank 0 coordinates; ranks 1 .. R-1 each run ECM on one GPU.\n"
          "  N is read from stdin (on rank 0) when not given.\n"
          "\n"
          "  -1, --b1 B1        stage 1 bound (default 10000)\n"
          "  -2, --b2 B2        stage 2 bound (default 100*B1 below B1 = "
          "50000,\n"
          "                     50*B1 from there; 0 = stage 1 only)\n"
          "  -c, --curves C     curves per batch (default 0 = one wave of "
          "the\n"
          "                     worker's GPU)\n"
          "  -s, --seed S       base seed: batch i uses seed S + i (default "
          "1)\n"
          "  -b, --batches K    stop after K batches (default 0 = until a "
          "factor\n"
          "                     is found)\n"
          "  -F, --fft          use the FFT arithmetic even below %d bits\n"
          "  -v, --verbose      a line per batch, and a summary, on stderr\n"
          "  -h, --help         this text\n"
          "\n"
          "Batch i is `gmpecmfft -s S+i -c C` on one GPU. Use one kind of "
          "GPU:\n"
          "with a mix the faster ones sit idle while the slower finish.\n",
          GMPECMFFT_CGBN_MAX_BITS);
}

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

static double now(void) { return MPI_Wtime(); }

/* ------------------------------------------------------------------ */

static int coordinator(int nworkers, unsigned long b1, unsigned long b2,
                       unsigned long seed, long max_batches, int verbose) {
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  char *stop_sent = calloc((size_t)nworkers + 1, 1);
  long *redo = calloc((size_t)nworkers + 1, sizeof(long)); /* batches to
                                                    re-issue after errors */
  int nredo = 0;
  int alive = nworkers, stopping = 0, failed = 0;
  long next = 0, done = 0;
  unsigned long long curves = 0;
  double busy = 0.0;
  int found = 0, first_error = 0;
  char *factor = NULL;
  long fbatch = -1;
  int frank = -1;
  unsigned fcurves = 0;
  const double t0 = now();

  /* STOP to every worker that has not had one. */
  #define STOP_ALL()                                                        \
    do {                                                                    \
      stopping = 1;                                                         \
      for (int w = 1; w <= nworkers; w++)                                   \
        if (!stop_sent[w]) {                                                \
          MPI_Send(NULL, 0, MPI_BYTE, w, TAG_STOP, MPI_COMM_WORLD);         \
          stop_sent[w] = 1;                                                 \
        }                                                                   \
    } while (0)

  while (alive > 0) {
    if (interrupted && !stopping)
      STOP_ALL();
    int flag = 0;
    MPI_Status st;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &st);
    if (!flag) {
      struct timespec ts = {0, 5000000}; /* 5 ms */
      nanosleep(&ts, NULL);
      continue;
    }
    Result r;
    const int w = st.MPI_SOURCE;
    MPI_Recv(&r, sizeof r, MPI_BYTE, w, TAG_RESULT, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    char *f = NULL;
    if (r.factor_len > 0) {
      f = malloc((size_t)r.factor_len);
      MPI_Recv(f, r.factor_len, MPI_CHAR, w, TAG_FACTOR, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    if (r.batch < 0 && r.status != GMPECMFFT_FACTOR_FOUND) {
      /* Could not even start (no GPU, ...): retire this worker. */
      fprintf(stderr, "gmpecmfft-mpi: rank %d: %s\n", w,
              gmpecmfft_status_string((gmpecmfft_status_t)r.status));
      if (!first_error)
        first_error = r.status;
      failed++;
      if (!stop_sent[w]) {
        MPI_Send(NULL, 0, MPI_BYTE, w, TAG_STOP, MPI_COMM_WORLD);
        stop_sent[w] = 1;
      }
    } else if (r.batch >= 0) {
      const gmpecmfft_status_t s = (gmpecmfft_status_t)r.status;
      const int counted = s == GMPECMFFT_FACTOR_FOUND ||
                          s == GMPECMFFT_NO_FACTOR;
      if (counted) {
        done++;
        curves += r.curves;
      }
      busy += r.seconds;
      if (verbose)
        fprintf(stderr,
                "gmpecmfft-mpi: batch %ld (seed %lu, %u curves) on rank %d "
                "(GPU %d): %s, %.2f s\n",
                r.batch, seed + (unsigned long)r.batch, r.curves, w,
                r.device, gmpecmfft_status_string(s), r.seconds);
      if (s == GMPECMFFT_FACTOR_FOUND && f && !found) {
        found = 1;
        factor = f;
        f = NULL;
        fbatch = r.batch;
        frank = w;
        fcurves = r.curves;
        STOP_ALL();
      } else if (!counted && s != GMPECMFFT_STOPPED) {
        /* The batch failed on this worker (out of memory, ...): retire
         * the worker, and hand its batch to another. */
        fprintf(stderr, "gmpecmfft-mpi: rank %d, batch %ld: %s\n", w,
                r.batch, gmpecmfft_status_string(s));
        redo[nredo++] = r.batch;
        if (!first_error)
          first_error = r.status;
        failed++;
        if (!stop_sent[w]) {
          MPI_Send(NULL, 0, MPI_BYTE, w, TAG_STOP, MPI_COMM_WORLD);
          stop_sent[w] = 1;
        }
      }
    }
    free(f);
    if (stop_sent[w]) { /* that was its last result */
      alive--;
      continue;
    }
    if (!stopping && nredo > 0) {
      MPI_Send(&redo[--nredo], 1, MPI_LONG, w, TAG_WORK, MPI_COMM_WORLD);
      continue;
    }
    if (stopping || (max_batches > 0 && next >= max_batches)) {
      MPI_Send(NULL, 0, MPI_BYTE, w, TAG_STOP, MPI_COMM_WORLD);
      stop_sent[w] = 1;
      alive--;
      continue;
    }
    MPI_Send(&next, 1, MPI_LONG, w, TAG_WORK, MPI_COMM_WORLD);
    next++;
  }
  #undef STOP_ALL

  const double wall = now() - t0;
  if (found)
    printf("%s\n", factor);
  if (verbose) {
    if (found)
      fprintf(stderr,
              "gmpecmfft-mpi: factor found by rank %d in batch %ld (seed "
              "%lu, %u curves; on one GPU: gmpecmfft -s %lu -c %u)\n",
              frank, fbatch, seed + (unsigned long)fbatch, fcurves,
              seed + (unsigned long)fbatch, fcurves);
    fprintf(stderr,
            "gmpecmfft-mpi: %ld batches, %llu curves completed on %d "
            "workers in %.2f s wall (%.1f curves/s; %.2f s of batch time; "
            "B1 %lu, B2 %lu)\n",
            done, curves, nworkers - failed, wall,
            wall > 0 ? curves / wall : 0.0, busy, b1, b2);
  }
  free(factor);
  free(stop_sent);
  free(redo);
  if (found)
    return GMPECMFFT_FACTOR_FOUND;
  if (interrupted)
    return GMPECMFFT_STOPPED;
  if (failed == nworkers && first_error)
    return first_error;
  return GMPECMFFT_NO_FACTOR;
}

/* ------------------------------------------------------------------ */

static void worker(const char *n, unsigned long b1, unsigned long b2,
                   unsigned curves, unsigned long seed, int force_fft) {
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);

  /* GPU: this worker's index among the workers on its node. */
  MPI_Comm workers, node;
  MPI_Comm_split(MPI_COMM_WORLD, 1, 0, &workers);
  MPI_Comm_split_type(workers, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &node);
  int local = 0, nlocal = 1, ndev = 0, dev = -1;
  MPI_Comm_rank(node, &local);
  MPI_Comm_size(node, &nlocal);
  int setup = GMPECMFFT_FACTOR_FOUND; /* "no problem" */
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1)
    setup = GMPECMFFT_ERR_CUDA;
  else {
    dev = local % ndev;
    if (cudaSetDevice(dev) != cudaSuccess)
      setup = GMPECMFFT_ERR_CUDA;
    else if (local == 0 && nlocal > ndev)
      fprintf(stderr,
              "gmpecmfft-mpi: %d workers share %d GPU%s on one node; one "
              "worker per GPU is the intended layout\n",
              nlocal, ndev, ndev == 1 ? "" : "s");
  }
  const unsigned c =
      curves ? curves : (setup ? 0 : gmpecmfft_ecm_curves(n, force_fft));

  /* The first message is a request (batch -1); with a setup error it
   * carries that error instead, and the coordinator retires us. */
  Result r;
  memset(&r, 0, sizeof r);
  r.batch = -1;
  r.device = dev;
  r.curves = c;
  r.status = setup;
  char *f = NULL;
  for (;;) {
    MPI_Send(&r, sizeof r, MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    if (r.factor_len)
      MPI_Send(f, r.factor_len, MPI_CHAR, 0, TAG_FACTOR, MPI_COMM_WORLD);
    gmpecmfft_string_free(f);
    f = NULL;
    MPI_Status st;
    long batch = -1;
    MPI_Recv(&batch, 1, MPI_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    if (st.MPI_TAG == TAG_STOP)
      break;
    const double t0 = now();
    const gmpecmfft_status_t s =
        (force_fft ? gmpecmfft_ecm_fft : gmpecmfft_ecm)(
            n, b1, b2, c, seed + (unsigned long)batch, stop_requested, NULL,
            &f);
    r.batch = batch;
    r.status = (int)s;
    r.seconds = now() - t0;
    r.factor_len =
        (s == GMPECMFFT_FACTOR_FOUND && f) ? (int)strlen(f) + 1 : 0;
  }
  MPI_Comm_free(&node);
  MPI_Comm_free(&workers);
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  static const struct option opts[] = {
      {"b1", required_argument, 0, '1'},
      {"b2", required_argument, 0, '2'},
      {"curves", required_argument, 0, 'c'},
      {"seed", required_argument, 0, 's'},
      {"batches", required_argument, 0, 'b'},
      {"fft", no_argument, 0, 'F'},
      {"verbose", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};
  unsigned long b1 = 10000, b2 = 0, seed = 1;
  long max_batches = 0;
  int have_b2 = 0, force_fft = 0, verbose = 0, bad = 0, help = 0;
  unsigned curves = 0;
  int c;
  while ((c = getopt_long(argc, argv, "1:2:c:s:b:Fvh", opts, NULL)) != -1) {
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
    case 'b':
      max_batches = strtol(optarg, NULL, 10);
      break;
    case 'F':
      force_fft = 1;
      break;
    case 'v':
      verbose = 1;
      break;
    case 'h':
      help = 1;
      break;
    default:
      bad = 1;
    }
  }
  if (!have_b2)
    b2 = (b1 >= 50000 ? 50 : 100) * b1;
  if (!seed)
    seed = 1; /* the library reads seed 0 as 1: keep batch seeds distinct */
  if (help || bad || size < 2) {
    if (rank == 0) {
      usage(help ? stdout : stderr);
      if (size < 2 && !help && !bad)
        fprintf(stderr, "gmpecmfft-mpi: needs at least 2 ranks (a "
                        "coordinator and a worker)\n");
    }
    MPI_Finalize();
    return help ? 0 : GMPECMFFT_ERR_INVALID_INPUT;
  }

  /* The number, read on rank 0, to everyone. */
  char *n = NULL;
  long len = 0;
  if (rank == 0) {
    n = optind < argc ? strdup(argv[optind]) : read_number();
    len = n ? (long)strlen(n) + 1 : 0;
    if (len <= 1)
      len = 0;
  }
  MPI_Bcast(&len, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  if (!len) {
    if (rank == 0)
      usage(stderr);
    free(n);
    MPI_Finalize();
    return GMPECMFFT_ERR_INVALID_INPUT;
  }
  if (rank != 0)
    n = malloc((size_t)len);
  MPI_Bcast(n, (int)len, MPI_CHAR, 0, MPI_COMM_WORLD);

  int ret = 0;
  if (rank == 0) {
    /* The coordinator takes no part in the workers' node split. */
    MPI_Comm none;
    MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &none);
    ret = coordinator(size - 1, b1, b2, seed, max_batches, verbose);
  } else {
    worker(n, b1, b2, curves, seed, force_fft);
  }
  free(n);
  MPI_Finalize();
  return ret;
}
