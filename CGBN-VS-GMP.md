# GMP on the CPU or CGBN on the GPU? Measurements from gmp-ecpp

This report asks whether the multiple-precision arithmetic that gmp-ecpp
does on the host with GMP should move to the GPU with CGBN. The question
covers both programs: the ECPP prover (`libgmpecpp`, `gmpecpp-mpi`) and
the ECM factorizer (`libgmpecmfft`, `gmpecmfft-mpi`). Three experiments
answer it:

1. the latency and throughput of a single operation;
2. where each program's host time actually goes;
3. end-to-end batch proving over MPI, with and without the CGBN backend,
   across several layouts and seeds.

Every number below comes from a committed harness (`make
bench-cgbn-latency`, `make bench-gmp-profile`, `make bench-mpi-cgbn`); see
[Reproducing](#reproducing).

## Summary

- **One-for-one replacement of GMP calls with CGBN is slower, by 4× to
  172×.** A single CGBN instance runs on 8–32 threads of one SM. Every
  call pays a fixed round trip of about 26 µs: two copies in, a launch, a
  synchronization, a copy out. GMP finishes a 512-bit modular
  multiplication in 0.15 µs.
- **Batched, CGBN is 12× to 124× faster than GMP.** That takes thousands
  of independent operations of one width in a single launch.
- **The batchable host work has already moved; the rest cannot.**
  - In the factorizer, host GMP is ≤ 0.06% of host time, and above
    16352 bits CGBN cannot hold the numbers at all.
  - In the prover, the batchable part is the elliptic curve method
    (ECM), and `-C` already runs it on CGBN. What GMP still does is
    mostly affine point arithmetic (gcd-bound), in work that is small
    and sequential: the host's deliberate first attempt at each
    factorization, and the downrun's point checks. Experiment 1 rules
    out one-call-at-a-time offload of either.
- **End to end, the CGBN backend's benefit depends on the host's thread
  budget.**
  - It is 1.83× with one worker of 8 ECM threads, 1.37× with 16, and
    1.20× with 32.
  - With 4 or 8 workers sharing one GPU it is roughly even (0.98×,
    1.16×).
  - In every configuration it cuts variance: one worker's 18-proof batch
    ranged from 39 to 57 s on the host and from 24 to 28 s with CGBN.
- **Process binding can fake the result.** `mpirun -n 2` binds the
  worker to one core by default. All its ECM threads then share that
  core, the host path slows 4×, and a spurious 3.6× "GPU advantage"
  appears.

## System under test

| | |
|---|---|
| CPU | Intel Core i9-13900K: 8 P-cores × 2 threads (CPUs 0–15) + 16 E-cores (CPUs 16–31), up to 5.8 GHz |
| GPU | NVIDIA RTX PRO 4500 Blackwell (sm_120), 32 GB, driver 595.71.05 |
| CUDA | 13.2; CGBN (NVlabs) from `/usr/local/include` |
| GMP | 6.3.0 (Fedora 41 `gmp-6.3.0-2.fc41`) |
| MPI | Open MPI 5.0.5, rebuilt CUDA-aware (`5.0.5-2.1.cuda.fc41`; CUDA-awareness plays no part here, since all messages are small host messages) |
| OS | Fedora 41, Linux 6.17.10 |
| Compilers | GCC 14 (host, `-O3` for the tools, `-O2`-class flags for the libraries), `nvcc -O3 -arch=sm_120` |

## Background: what runs where

**CGBN** is NVlabs' fixed-width multiple-precision library.
- **Instances:** an instance of `BITS` bits is spread over `TPI` threads
  (8, 16 or 32; cooperative groups, warp-synchronous), each holding
  `BITS/(32·TPI)` 32-bit limbs.
- **Operations:** arithmetic is done cooperatively across those threads;
  modular multiplication is Montgomery's.
- **Throughput:** comes only from running many independent instances per
  launch.
- **Width ceiling:** in this project the widest instance is 16384 bits,
  because at `TPI = 32` the kernels already use 236–248 of 255
  registers.

**The prover** (Atkin–Morain ECPP) is mostly host GMP:
- `mpz` for the number theory;
- `mpf` for the class polynomials, which are built from
  arbitrary-precision complex arithmetic;
- polynomial root finding mod p;
- the elliptic curve method, which it uses to factor candidate curve
  orders.

Its host ECM (`LenstrasECM`) works in affine Weierstrass coordinates, so
every point addition needs a modular inverse, i.e. an extended gcd. With
`-T N` it runs N threads (`LenstrasECM_mt`):
- **Work split:** each thread takes curves `id, id+N, …` from
  `curves_per_bound = max(N, 20)` curves per bound, and each runs its own
  sequence of bounds without a barrier.
- **First find wins:** the first thread to find a factor sets an atomic
  flag, and the others stop.

With `-C`, the prover keeps the host attempt first and escalates to the
GPU only when the cofactor is at least 192 bits. The GPU then runs 4096
Montgomery-curve instances (no inversions) at once, stage 1 and stage 2,
with B1 clamped to [10³, 10⁵]. An earlier version escalated every failed
host attempt and made proofs 2× slower: most calls are meant to fail
fast, because they are rejecting candidate curve orders.

**The factorizer** runs ECM entirely on the GPU, on CGBN up to 16352
bits and on cuFFT arithmetic above. The host does curve setup, moves a
few values between engines, and takes one gcd per curve at the end.

**`gmpecpp-mpi`** distributes whole numbers:
- **Dispatch:** rank 0 hands out indices on request, and each worker
  calls `gmpecpp_prove()`.
- **Seeding:** every proof reseeds the prover's generator from the
  configured seed (`gmp_randseed_ui(rstate, seed)` at the start of
  `gmpecpp_prove`). The order in which ranks take numbers therefore does
  not affect any proof. `-s` (new) sets that seed.

## Experiment 1: one operation, one instance, or 4096

**Method** (`bench/cgbn-latency.cu`):
- **Operations:** for 512, 1024 and 2048 bits (CGBN `TPI` = 8, 16, 32):
  - modular exponentiation `a^e mod n` with a full-width exponent: the
    core of Rabin–Miller and of square roots mod p;
  - modular multiplication `a·b mod n`, done as `cgbn_mul_wide` +
    `cgbn_rem_wide`.
- **GMP:** one thread, `mpz_powm`, and `mpz_mul` + `mpz_mod`.
- **"CGBN one":** a full round trip for one instance:
  1. three `cudaMemcpy` host→device (operands and modulus);
  2. a `<<<1, TPI>>>` launch;
  3. a `cudaMemcpy` device→host, which synchronizes.

  This is the cost of replacing one GMP call.
- **"CGBN batch":** 4096 instances in one launch (128 threads per
  block), plus the result copy, divided by 4096. Contexts use
  `cgbn_no_checks`.
- **Correctness:** one result per width is compared with GMP; all
  matched.

| Operation | GMP | CGBN, one instance | CGBN, 4096 per launch |
|---|---:|---:|---:|
| `a^e mod n`, 512 bits | 49 µs | 1439 µs (29× slower) | 0.83 µs/op (59× faster) |
| `a^e mod n`, 1024 bits | 321 µs | 4774 µs (15× slower) | 3.14 µs/op (102× faster) |
| `a^e mod n`, 2048 bits | 2157 µs | 9210 µs (4.3× slower) | 17.4 µs/op (124× faster) |
| `a·b mod n`, 512 bits | 0.15 µs | 26.2 µs (172× slower) | 0.013 µs/op (12× faster) |
| `a·b mod n`, 1024 bits | 0.43 µs | 26.2 µs (61× slower) | 0.019 µs/op (22× faster) |
| `a·b mod n`, 2048 bits | 1.42 µs | 26.3 µs (19× slower) | 0.036 µs/op (40× faster) |

**Reading it:**

- **The fixed cost:** the multiplication rows are flat at about 26 µs
  for one instance at every width. That is the cost of the round trip
  (pageable `cudaMemcpy`, launch, synchronization), not of the
  arithmetic. Any design that hands single operations to the GPU pays it
  every time. GMP's 512-bit multiplication is 170 times shorter.
- **The parallelism:** a single exponentiation leaves the GPU almost
  idle. One instance is one warp, or a quarter of one, on one SM of many,
  executing a serial square-and-multiply chain. Against it is one host
  core at up to 5.8 GHz running GMP's hand-tuned x86-64 assembly. The
  single-instance gap narrows with width (29× at 512 bits, 4.3× at 2048)
  because more threads per instance share the work. It does not close.
- **The batch:** with 4096 instances per launch the device is full and
  the per-operation cost collapses. The advantage grows with width for
  exponentiation (59× → 124×), where more arithmetic sits behind each
  byte copied.
- **Not measured:** launch-latency tuning (CUDA graphs, pinned memory,
  persistent kernels) would shave part of the 26 µs. No such technique
  removes the gap between one warp and the whole GPU.

Consequence: CGBN is worth using only where an algorithm exposes
thousands of independent same-width operations at once. ECM curves do,
and so would, for example, primality tests of many candidates at once.
A sequential chain does not: a downrun step, a gcd, or Cornacchia on one
discriminant.

## Experiment 2: where the host time goes

**Method** (`bench/gmp-profile.sh`): `perf record -e cpu-clock -F 1999`
on each run, then `perf report --sort dso`, and for the prover also
`--sort dso,sym`, with libgmp's symbols grouped into gcd/division
(`*gcd*`, `*hgcd*`, `div2`, `*divrem*`, `*div_qr*`, `*bdiv*`,
`*invert*`), `mpf_*`, and the rest.
- **Why `cpu-clock`:** on this hybrid CPU the default `cycles` event is
  two PMU events, `cpu_core/cycles/` and `cpu_atom/cycles/`. `perf
  report` prints one table per event, and a thread that migrates between
  core types is split across both. `cpu-clock` is a software timer: it
  counts time, uniformly, in one table.
- **Interpretation:** the `libgmp` share bounds what any GPU offload of
  GMP could save.
- **Waiting:** a thread blocked in a CUDA synchronization spins inside
  `libcuda`, so that share is time spent waiting on the GPU.

| Run | Wall | `libgmp` | of which gcd/division | `mpf_*` | `libcuda` |
|---|---:|---:|---:|---:|---:|
| Prover, 140 digits, seed 1, host only, 1 thread | 67.6 s | 92.2% | 76.0% | 0.0% | — |
| Prover, same, `-C` | 2.9 s | 64.8% | 52.0% | 0.1% | 23.6% |
| `gmpecmfft`, 2000 bits, 2952 curves (CGBN path) | | 0.04% | | | 90% |
| `gmpecmfft`, 20000 bits, 64 curves (cuFFT path) | | 0.06% | | | 89% |

**Reading it:**

- **The factorizer** is essentially pure GPU. Its host GMP is at most
  0.06% of host time, and the host thread spends about 90% waiting on
  the device (most of the rest is `[vdso]`, the polling loop's clock
  reads). No GMP replacement can show up here: even a free one would be
  below run-to-run noise. Above 16352 bits there is nothing CGBN could
  do anyway, since that is the reason the cuFFT path exists.
- **The prover, host only:** the run is dominated by gcd and division
  code: `__gmpn_hgcd2` about 39% and `div2` about 22% of samples. The
  prover's point arithmetic (`multiply`) is affine, with an inversion at
  every point addition. It serves the host ECM and the downrun's point
  checks alike.
  - **What `-C` removes:** the ECM share is precisely the work `-C`
    replaces. CGBN does it with Montgomery's x-only curves, which need
    no inversion, across 4096 curves at once. Hence 67.6 s → 2.9 s, 23×,
    on this proof.
  - **Why the split is unknown:** a frame-pointer call graph cannot
    divide `multiply`'s time between its two callers. GMP's assembly
    keeps no frame pointer, so unwinding stops early and credits
    `LenstrasECM` with almost nothing although its inner loop calls
    `multiply`.
- **The prover, `-C`:** 65% of the time remains in GMP, still mostly
  gcd and division (52%); `mpf` is 0.1% at the default precision, and
  the libcuda 24% is the host waiting on the GPU's ECM. The affine
  arithmetic that remains belongs to small, sequential work:
  - **The host's first attempt at every factorization.** The dispatch
    policy keeps it on the host because most attempts are meant to fail
    quickly; escalating all of them was measured to make proofs 2×
    slower.
  - **The downrun's point checks**, a short chain per step.

  Neither offers thousands of independent same-width operations, and
  experiment 1 rules out offloading them one call at a time. Batching
  the host attempts themselves into a small GPU ECM is conceivable but
  untested here, and it would have to beat a fixed per-launch cost that
  most of these attempts, failing in milliseconds, cannot amortize.

## Experiment 3: batch proofs over MPI, with and without CGBN

**Method** (`bench/mpi-cgbn.sh`, `make bench-mpi-cgbn`):
- **Inputs:** the 18 primes of 90–180 digits in `bench/primes.txt` (4 ×
  90, 5 × 100, 2 × 110, 1 × 120, 2 × 130, 2 × 140, 1 × 150, 1 × 180).
- **Layouts:** workers × ECM threads of 1 × 8, 4 × 8 and 8 × 4, for
  seeds 1–5, each host-only and with `-C`. Plus 1 × 16 and 1 × 32 for
  seed 1, as a thread-count sweep.
- **Ranks:** rank 0 coordinates. All ranks are unbound (`mpirun
  --oversubscribe --bind-to none`), and every `-C` rank shares the one
  GPU.
- **Ordering:** host-only and `-C` runs alternate, so slow drift
  (clocks, temperature) affects both alike.
- **Completeness:** every one of the 2 × 37 runs proved all 18 numbers.

### Results

Batch wall time; mean (min–max) over seeds:

| Workers × threads | Seeds | Host only | `-C` (CGBN) | Ratio |
|---|---:|---:|---:|---:|
| 1 × 8 | 5 | 47.5 s (39.3–57.3) | 25.9 s (23.7–27.7) | 1.83× |
| 1 × 16 | 1 | 33.1 s | 24.2 s | 1.37× |
| 1 × 32 | 1 | 27.6 s | 23.1 s | 1.20× |
| 4 × 8 | 5 | 17.0 s (13.4–19.8) | 17.4 s (15.7–18.6) | 0.98× |
| 8 × 4 | 5 | 21.5 s (20.1–24.9) | 18.5 s (17.0–19.5) | 1.16× |
| 1 × 8, rank bound to one core | 1 | 185.3 s | 51.7 s | 3.59× (artifact) |

With several workers, the summed proof time (the sum over proofs of each
proof's own time) is 52.7 s host-only against 46.3 s with CGBN at 4 × 8
(1.14×), and 79.2 s against 79.0 s at 8 × 4 (1.00×).

Per proof, 1 × 8, median (min–max) over 5 seeds:

| # | Digits | Host only | `-C` (CGBN) | Host / CGBN |
|---:|---:|---:|---:|---:|
| 0 | 90 | 0.47 (0.38–0.50) | 0.51 (0.49–0.57) | 0.92× |
| 1 | 90 | 0.69 (0.65–0.85) | 1.46 (1.40–1.58) | 0.47× |
| 2 | 90 | 2.79 (2.68–2.97) | 0.37 (0.36–0.38) | 7.54× |
| 3 | 90 | 0.64 (0.55–0.86) | 0.38 (0.36–0.40) | 1.68× |
| 4 | 100 | 4.32 (4.19–4.49) | 1.59 (0.60–1.94) | 2.72× |
| 5 | 100 | 0.25 (0.23–0.31) | 0.45 (0.41–0.51) | 0.56× |
| 6 | 100 | 1.09 (0.50–5.80) | 0.46 (0.29–0.50) | 2.37× |
| 7 | 100 | 0.36 (0.34–0.74) | 0.70 (0.60–0.75) | 0.51× |
| 8 | 100 | 1.41 (1.13–1.60) | 0.34 (0.29–0.45) | 4.15× |
| 9 | 110 | 1.44 (1.39–1.51) | 0.93 (0.75–1.00) | 1.55× |
| 10 | 110 | 1.04 (0.89–1.31) | 0.55 (0.44–0.71) | 1.89× |
| 11 | 120 | 1.09 (0.82–1.52) | 1.37 (1.29–1.50) | 0.80× |
| 12 | 130 | 1.41 (1.33–5.56) | 2.89 (1.49–2.93) | 0.49× |
| 13 | 130 | 1.19 (1.17–1.23) | 2.55 (0.95–2.74) | 0.47× |
| 14 | 140 | 10.60 (7.56–13.12) | 1.38 (1.09–1.44) | 7.68× |
| 15 | 140 | 2.75 (1.46–5.86) | 1.05 (0.97–1.16) | 2.62× |
| 16 | 150 | 7.46 (5.03–7.76) | 1.43 (1.36–1.48) | 5.22× |
| 17 | 180 | 6.09 (3.42–9.26) | 8.50 (7.12–9.33) | 0.72× |
| | | 45.1 (sum of medians) | 26.9 | 1.68× |

### What it shows

1. **The GPU's value falls as the host's thread budget rises**: 1.83×,
   1.37× and 1.20× with 8, 16 and 32 ECM threads on one worker.
   - The prover escalates to the GPU only when the host attempt fails on
     a cofactor of 192 bits or more.
   - With more threads, the host attempt (which ends at the first factor
     any thread finds) succeeds more often, and sooner, before
     escalation.
   - The CGBN side barely moves (25.9 → 24.2 → 23.1 s), because its time
     is dominated by work the extra host threads do not touch.
2. **Several ranks on one GPU dilute it to nothing in wall time**: 0.98×
   at 4 × 8 and 1.16× at 8 × 4.
   - The ranks contend for one device, whose kernels are latency-bound
     but occupy it while they run.
   - Each rank also pays CUDA context creation and module loading.
   - Per-proof GPU times rise with more ranks. Proof #1 (90 digits)
     with `-C` takes 1.40–1.58 s on one worker, 3.23–3.58 s with four
     and 6.46–7.17 s with eight, across all five seeds.
3. **It works on the tail.**
   - **Where CGBN wins:** the hard proofs, those where the host would
     grind on a stubborn cofactor. #2 7.5×, #14 7.7×, #16 5.2×, #8 4.2×.
   - **Where it loses:** easy proofs, 8 of 18, by 0.47–0.92×. Each
     escalation pays fixed GPU costs (launches, a 4096-curve wave,
     transfers) that the host would not have.
   - **The result:** variance reduction. Across seeds the host-only batch
     ranged 39.3–57.3 s, the CGBN batch 23.7–27.7 s; per proof, #6 ranged
     0.50–5.80 s on the host and 0.29–0.50 s with CGBN.
   - **Earlier result:** this is also what the project's earlier
     single-thread sweep showed (`make bench-cuda`, 36 proofs, one ECM
     thread: 420.6 s → 120.8 s, 3.48×), in the configuration most
     favourable to the GPU.

### Determinism, and why runs vary

- **Per-proof reseeding:** `gmpecpp_prove()` reseeds its generator from
  the configured seed on entry, so a proof does not depend on which rank
  takes it or in what order. A seed fixes each proof's random choices:
  discriminant probing, points, curves.
- **The race:** with `-T > 1` the host ECM is a race. Each thread draws
  curves from its own deterministic stream (seeded from the proof seed
  and the thread id), and the first thread to find a factor wins, under
  a mutex and an atomic flag (`std::atomic<int> found`). Which factor
  wins depends on thread timing.
  - **Why it matters:** a different factor of a candidate order changes
    which candidates are accepted, and so the whole downrun: the step
    counts in the tables differ between modes for that reason.
  - **Under contention:** with several workers, or anything else
    competing for cores, the races can resolve differently from run to
    run. The seed ranges therefore include timing-induced variation, not
    just seed variation. This is inherent to `-T` (the certificates
    remain valid; `gmpecpp-verify` checks them, byte comparison cannot).

### The binding trap

Open MPI binds processes to cores by default for small jobs. With two
processes (coordinator plus one worker):

```
$ mpirun --report-bindings -n 2 ./gmpecpp-mpi -T 8 primes.txt
... Rank 0 bound to package[0][core:0]
... Rank 1 bound to package[0][core:1]
$ mpirun --oversubscribe --report-bindings -n 2 ./gmpecpp-mpi -T 8 primes.txt
... Rank 0 is not bound (or bound to all available processors)
... Rank 1 is not bound (or bound to all available processors)
```

**What binding does:** the worker's `pthread`s inherit its CPU affinity.
With `-T 8`, all eight ECM threads time-share core 1's two hardware
threads.

**The measured effect** (1 × 8, seed 1):

| | Host only | `-C` (CGBN) | Apparent ratio |
|---|---:|---:|---:|
| Unbound | 45.1 s | 26.0 s | 1.73× |
| Bound to one core | 185.3 s | 51.7 s | 3.59× |

The host path, nearly all threaded ECM, slows 4.1×; the CGBN path, which
needs little host CPU, slows 2×. The comparison then reports a 3.6× GPU
advantage that is mostly lost host parallelism. The trap is easy to fall
into, because two common launch forms differ only in this side effect:
- `--oversubscribe` disables binding;
- multi-rank jobs launched with it look fine.

The harness passes `--bind-to none` explicitly. For hybrid
multi-threaded MPI ranks generally, check `--report-bindings` (or
`taskset -p` on a worker) before trusting any thread-scaling number.

## Recommendations

- **Do not port scalar GMP code to CGBN.** Experiment 1 bounds it at 4×
  to 172× slower per operation.
- **Port only batched, same-width integer work.** ECM already is.
  Further candidates would have to be restructured into batches first
  (for example, testing many candidate orders or discriminants at once).
  Even then, Amdahl's law applies to what remains: in a `-C` proof, the
  `mpf` class polynomial work cannot move.
- **For batch proving on one GPU:** prefer few workers with many unbound
  ECM threads each, and `-C`.
  - CGBN then removes the pathological proofs.
  - With many ranks per GPU, give each GPU one rank (`gmpecpp-mpi` across
    nodes with one GPU each is the natural layout).
- **The factorizer gains nothing from moving its remaining host GMP**
  (≤ 0.06%).

## What was wrong along the way

Five intermediate results in this investigation were wrong.

1. **"CGBN is 3.6–4× faster with one worker."** The first single-worker
   runs used `mpirun -n 2` without `--oversubscribe`, so the worker was
   bound to one core (see [The binding trap](#the-binding-trap)).
2. **"`-T 32` is slower than `-T 8` because of E-cores and
   `curves_per_bound`."** The explanation was plausible:
   - `curves_per_bound = max(T, 20)` adds work above 20 threads;
   - threads beyond 8 land on SMT siblings and E-cores.

   The premise was the binding artifact, though. Unbound, 32 threads beat
   8 (27.6 s against 45.1 s).
3. **"Multi-rank runs differ because the numbers reach ranks in a
   different order and the random state carries over."** The prover
   reseeds at every proof. The real source is the threaded ECM race
   under contention ([Determinism](#determinism-and-why-runs-vary)).

4. **"With `-C`, the GMP that remains is the class polynomials'
   `mpf`."** This came from misreading an exploratory `perf` report. With
   `cpu-clock` and the symbols grouped, `mpf` is 0.1%, and gcd/division
   from the affine point arithmetic is 52%.
5. **Profile shares that add up to more than 100%.** On the hybrid CPU,
   `perf`'s default `cycles` event produced two report tables (P-core
   and E-core PMUs), and the first grouping summed across both.
   `cpu-clock` fixed it.

Each was caught by measuring again with a controlled setup, rather than
by reasoning about the first numbers.

## Reproducing

```sh
make                       # libraries, gmpecpp-mpi (needs Open MPI), gmpecmfft
make bench-cgbn-latency    # experiment 1, ~1 min; builds build/cgbn-latency
make bench-gmp-profile     # experiment 2, ~2 min; needs perf
make bench-mpi-cgbn        # experiment 3, ~40 min; results in build/bench-mpi-cgbn/
bash bench/mpi-cgbn.sh DIR "1 2 3" bound   # parts and seeds can be chosen
```

`bench-mpi-cgbn` writes each run's raw output to
`build/bench-mpi-cgbn/<layout>-<cpu|gpu>-s<seed>.txt`, and the tables
above to `summary.txt`. Numbers will differ on other hardware; the
ratios and the binding effect are what to compare.
