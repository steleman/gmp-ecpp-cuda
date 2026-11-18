# GMP-ECPP with CUDA and CGBN

An **elliptic curve primality prover** (ECPP, Atkin–Morain) built on
[GMP](https://gmplib.org/).

Given an integer, it does not merely report that the number is *probably* prime —
it constructs a **primality certificate**: a chain of elliptic curves that
together prove primality, and that can be checked independently of the program
that produced it. This is the difference between Miller–Rabin telling you a
number is prime with probability 1 − 4⁻ᵏ and a proof that it is.

The tradeoff is cost. A probabilistic test on a 100-digit number takes
microseconds; proving it here takes about 20 seconds.

## Requirements

- A C++ compiler (tested with g++ 14.3)
- GMP with the C++ wrapper — `libgmp` and `libgmpxx`, plus `gmp.h` / `gmpxx.h`
  (Debian/Ubuntu: `libgmp-dev`; Fedora: `gmp-devel`, `gmp-c++`)

## Build

```
%>> gmake
```

Or directly:

```
g++ -O3 -Wall -o gmp-ecpp gmp-ecpp.cpp -lgmpxx -lgmp
```

The conventional order is `-lgmpxx` before `-lgmp`. With shared libraries it
does not actually matter — `libgmpxx.so` carries a `NEEDED` entry on `libgmp.so`,
so the loader resolves the dependency either way — but the order is required if
you ever link the static archives, where the linker resolves left to right.

The build emits eight warnings — four `-Wregister`, three `-Wunused-variable`,
one `-Wreturn-type` — all inherited from the original sources and all benign.

## Usage

The number under test is read from **standard input**, not from the command
line:

```
echo 1000000000000000000000000000057 | ./gmp-ecpp -Q
```

### Exit codes

The exit code is the result, and is what you should branch on in a script:

| Code | Meaning |
|:---:|---|
| `0` | **Proven prime** — the certificate chain ran to completion |
| `1` | **Probably prime** — only under `-o`, which stops after one step |
| `2` | **Composite**, *or* the prover gave up |

Note the ambiguity in `2`. The prover returns it both for a number it has shown
to be composite and for one it could not handle — after exhausting discriminants
(`Dmax > 312500`) or factoring bounds (`Bmax > 2000000000`). The two are
distinguishable only by the message on stdout, which `-Q` suppresses:

```
ProvePrime: ran out of discriminants
ProvePrime: exceeded maximum factoring bounds
```

If you need to tell "composite" from "gave up", do not run with `-Q`.

## Reading the output

Each step of the proof prints a block:

```
N[0] = 1000000000000000000000000000057
a = 0
b = 1000000000000000000000000000056
m = 999999999999998058691781125968
q = 623786892692306107
P = (2615091852, 405570618194490443159465151948)
P1 = (0, 1)
P2 = (132044575022392478926729626792, 850103387158260846662653286509)
```

That block is one link in the certificate:

- `N[i]` — the number this step proves prime
- `a`, `b` — the curve *y*² = *x*³ + `a`*x* + `b` over ℤ/`N[i]`ℤ
- `m` — the order of that curve, with |`m` − (`N`+1)| ≤ 2√`N` (Hasse)
- `q` — a large prime factor of `m`, required to exceed (⁴√`N` + 1)²
- `P` — a point on the curve; `P2` = [`m`/`q`]`P`, and `P1` = [`q`]`P2`

The proof obligation is that `P2` ≠ *O* while `P1` = *O*, where *O* = `(0, 1)` is
the point at infinity. Together with `q` > (⁴√`N` + 1)², that implies `N[i]` is
prime **provided `q` is prime** — so the prover recurses, and `q` becomes
`N[i+1]`.

The chain therefore descends, each step shrinking the number, until it reaches
one below 2³⁰, which is settled directly by trial division. In the example above
the chain is 10³⁰+57 → 623786892692306107 → 7690057189 → 591555037, and the last
is small enough to finish. Then, and only then, does it print:

```
proven prime
```

## Options

| Flag | Long form | Effect |
|---|---|---|
| `-h` | `--help` | Usage summary |
| `-Q` | `--Quiet` | Print very reduced messages |
| `-q` | `--quiet` | Print reduced messages |
| `-v` | `--verbose` | Verbose, including a progress marker |
| `-o` | `--one` | Run a single step, then stop (exit 1) |
| `-H` | `--Hilbert` | Use Hilbert class polynomials only |
| `-W` | `--Weber` | Use Weber class polynomials only |
| `-s` | `--seed` | Pseudo-random seed |
| `-p` | `--precision` | GMP floating-point precision, in bits (default 1000) |
| `-e` | `--error_shift` | Precision used in class polynomial generation (default 1000) |
| `-B` | `--Bmax` | Max ECM factoring bound (default 2000, auto-escalating) |
| `-D` | `--Dmax` | Max discriminant bound (default 20, auto-escalating) |

With neither `-H` nor `-W`, **both** polynomial families are enabled. Weber
polynomials have much smaller coefficients and are generally the cheaper route.

### A note on verbosity

By default the program prints π, *e*, and ln(1.9) at full working precision —
at the default 1000 bits, that is three lines of roughly 300 digits each,
before it does any work. This is a diagnostic for the floating-point layer, not
a bug, but it makes the default output impractical to read or log.

- `-q` suppresses those constants and the input prompt
- `-Q` additionally suppresses the seed/precision banner and the `Bmax`/`Dmax`
  progress lines

Certificate blocks and the final verdict are printed at **every** verbosity
level, including `-Q`. For scripting, use `-Q` and read the exit code.

### Tuning

`-B` and `-D` normally **auto-escalate**: when no usable discriminant is found,
the prover multiplies `Bmax` by 10 and `Dmax` by 5 and sweeps again, up to hard
ceilings. Passing `-B` or `-D` explicitly *pins* the value and disables that
escalation — so a hand-set bound that is too small turns a provable number into
exit code 2 rather than making the prover try harder. Prefer leaving both alone
unless you are deliberately bounding runtime.

If proofs fail on discriminants that ought to work, suspect precision before
suspecting the number theory: raise `-p` and `-e`. The prover detects precision
loss heuristically and reports it as `HCP loss of precision` or `WCP loss of
precision`.

## Performance

Measured on one machine at `-O2`, proving the smallest prime with the
given number of digits, seed pinned, at the default precision:

| Digits | Time |
|---:|---:|
| 20 | 0.02 s |
| 30 | 0.10 s |
| 40 | 0.13 s |
| 50 | 0.19 s |
| 60 | 0.17 s |
| 80 | 0.92 s |
| 100 | 18.60 s |
| 120 | 4.91 s |
| 150 | 25.50 s |
| 180 | 14.81 s |

Runtime is **not** a smooth function of size — 120 digits costs a quarter
of what 100 digits does. Cost is dominated by how soon the discriminant
sweep finds a `D` whose curve order factors usefully, and that depends on
the specific number, not on its length. Expect order-of-magnitude scatter
between neighbouring inputs.

Changing the seed makes little difference; it selects curves and points,
not the order in which discriminants are tried, which is deterministic.

### Precision is the dominant knob

`-p` sets the working precision of the class polynomial layer, and that
layer is 66–89% of the runtime. The default is **1000 bits**:

`make bench-precision` runs all 33 primes in `bench/primes.txt` at both
settings. **Every one proves at either precision, and none is slower at
1000 bits.** A representative selection:

| Digits | `-p 1000` | `-p 10000` | Ratio |
|---:|---:|---:|---:|
| 20 | 0.02s | 0.43s | 26.6× |
| 30 | 0.06s | 3.00s | 48.5× |
| 40 | 0.26s | 2.55s | 9.7× |
| 50 | 0.17s | 3.08s | 18.4× |
| 60 | 3.66s | 12.49s | 3.4× |
| 70 | 0.27s | 0.69s | 2.5× |
| 80 | 0.93s | 6.96s | 7.4× |
| 90 | 1.22s | 1.65s | 1.4× |
| 100 | 19.06s | 21.13s | 1.1× |
| 110 | 5.08s | 10.91s | 2.1× |
| 120 | 4.97s | 7.89s | 1.6× |
| 130 | 7.12s | 11.88s | 1.7× |
| 140 | 52.51s | 61.58s | 1.2× |
| 150 | 25.97s | 31.61s | 1.2× |
| 180 | 14.99s | 18.05s | 1.2× |

Across all 33: median **1.70×**, range 1.1× to 48.5×, aggregate
263.9 s → 180.5 s.

The advantage is dramatic on small inputs and narrows to a few tens of
percent past a hundred digits, because the time then goes to factoring
rather than to polynomial construction. It never inverts, which is why
the lower setting is the default.

If a proof does start failing on discriminants that ought to work, the
symptom is a `HCP loss of precision` or `WCP loss of precision` message
and the remedy is a larger `-p`. That failure mode costs time, not
correctness: the discriminant is abandoned and the sweep moves on.

## Using it as a library

The prover is built as `libgmpecpp.so`, with the command line tool as a
thin client of it. The C API is in [`include/gmpecpp.h`](include/gmpecpp.h)
and works from C and C++.

```c
#include <gmpecpp.h>

gmpecpp_config_t *cfg = gmpecpp_config_new();
gmpecpp_config_set_seed(cfg, 1);

gmpecpp_certificate_t *cert = NULL;
gmpecpp_status_t st = gmpecpp_prove(cfg, "1000000000000000000000000000057", &cert);

if (st == GMPECPP_PROVEN_PRIME) {
    size_t n = gmpecpp_certificate_num_steps(cert);
    for (size_t i = 0; i < n; i++) {
        gmpecpp_step_t s;
        gmpecpp_certificate_get_step(cert, i, &s);
        printf("step %zu: n=%s q=%s D=%ld\n", i, s.n, s.q, s.discriminant);
    }
}
gmpecpp_certificate_free(cert);
gmpecpp_config_free(cfg);
```

Three things differ from the original program, all forced by the move
from a program to a library:

- **Nothing is written to stdout or stderr.** Install a
  `gmpecpp_config_set_log()` callback to receive the running commentary,
  which arrives one whole line at a time. With no callback the library is
  silent.
- **`exit()` is never called.** The original terminated the process from
  deep inside the polynomial code when a modular inverse did not exist.
  That is a proof of compositeness, not an error, so it now unwinds and
  is reported as `GMPECPP_COMPOSITE` with the factor attached.
- **The certificate comes back as data**, not as printed text, and the
  three outcomes the tool collapses onto exit code 2 -- composite, out of
  discriminants, exceeded factoring bounds -- are reported separately.

`gmpecpp_prove()` serialises internally. The prover keeps its
configuration in file scope globals, so concurrent calls are safe but not
parallel; for throughput, run separate processes.

### API surface

Link with `-lgmpecpp`. The library exports these 36 symbols and nothing
else — internals are hidden with `-fvisibility=hidden`.

**Configuration** — every setter is optional; passing a `NULL` config to
`gmpecpp_prove()` means all defaults.

| Function | Default |
|---|---|
| `gmpecpp_config_new`, `gmpecpp_config_free` | |
| `gmpecpp_config_set_precision` | 1000 bits |
| `gmpecpp_config_set_error_shift` | 1000 |
| `gmpecpp_config_set_seed` | 0 = from the clock |
| `gmpecpp_config_set_bmax`, `gmpecpp_config_set_dmax` | 2000 / 20, auto-escalating; setting either pins it |
| `gmpecpp_config_set_hilbert`, `gmpecpp_config_set_weber` | neither set = both enabled |
| `gmpecpp_config_set_one_step` | off |
| `gmpecpp_config_set_backend` | `GMPECPP_BACKEND_CPU` |
| `gmpecpp_config_set_cuda_curves` | 0 = 4096 |
| `gmpecpp_config_set_cuda_min_bits` | 192 |
| `gmpecpp_config_set_threads` | 1 (ECM threads; 0 = one per CPU) — see **Parallelism** |
| `gmpecpp_config_set_sweep_threads` | 1 (discriminant-sweep threads; 0 = one per CPU) |
| `gmpecpp_config_set_log` | none, so the library is silent |
| `gmpecpp_config_set_verbosity` | 0 = silent |

**Proving**

| Function | Notes |
|---|---|
| `gmpecpp_prove` | decimal string in, certificate out |
| `gmpecpp_prove_mpz` | same, for an `mpz_t`; declared only if `<gmp.h>` precedes the header |
| `gmpecpp_preamble` | banner and constants, so an interactive caller can prompt afterwards |
| `gmpecpp_ecm_factor` | one round of ECM on its own, mainly for measurement |
| `gmpecpp_cuda_ecm` | ECM on the GPU alone: stage 1 and 2, resumable, stoppable, no lock — see **Standalone GPU ECM** |
| `gmpecpp_cuda_ecm_curves` | the curve count `gmpecpp_cuda_ecm` uses by default for a number of that size |
| `gmpecpp_string_free` | frees what `gmpecpp_ecm_factor` and `gmpecpp_cuda_ecm` return |

**Certificates**

| Function | Returns |
|---|---|
| `gmpecpp_certificate_num_steps` | chain length |
| `gmpecpp_certificate_get_step` | one `gmpecpp_step_t`, all fields decimal strings owned by the certificate |
| `gmpecpp_certificate_factor` | the factor exhibited, when one was |
| `gmpecpp_certificate_backend` | which backend actually ran |
| `gmpecpp_certificate_seed` | the seed used, for reproducing a clock-seeded run |
| `gmpecpp_certificate_elapsed`, `gmpecpp_certificate_ecm_elapsed` | total and factoring-stage seconds |
| `gmpecpp_certificate_free` | |

**Capabilities** — `gmpecpp_have_cuda`, `gmpecpp_cuda_device_name`,
`gmpecpp_version_string`, `gmpecpp_status_string`.

`gmpecpp_have_cuda()` answers honestly in a CPU-only build rather than
failing to link, so a caller can probe at runtime:

```c
if (gmpecpp_have_cuda())
    gmpecpp_config_set_backend(cfg, GMPECPP_BACKEND_CUDA);
```

### Status codes

`gmpecpp_status_t` separates outcomes the tool's exit code cannot:

| Status | Meaning |
|---|---|
| `GMPECPP_PROVEN_PRIME` | certificate chain completed |
| `GMPECPP_PROBABLY_PRIME` | `one_step` was set; chain not completed |
| `GMPECPP_COMPOSITE` | proved composite |
| `GMPECPP_OUT_OF_DISCRIMINANTS` | gave up, `Dmax` ceiling |
| `GMPECPP_FACTORING_BOUND` | gave up, `Bmax` ceiling |
| `GMPECPP_ERR_INVALID_INPUT` | not an integer, or below 2 |
| `GMPECPP_ERR_NO_MEMORY`, `GMPECPP_ERR_INTERNAL` | |

The command line tool deliberately collapses the middle three onto exit
code 2 for compatibility with the original.

## CUDA and CGBN

Building with CUDA present compiles a [CGBN](https://github.com/NVlabs/CGBN)
backend for the ECM factoring stage. `make CUDA=0` forces a CPU-only
build; `make cuda-info` shows what was detected. The device code is built for
`CUDA_ARCH`, which defaults to `sm_120` (RTX PRO 4500 Blackwell; CUDA from
`/usr/local/cuda`, 13.2 here) — pass `CUDA_ARCH=sm_89` for an RTX 4080. At runtime the backend is
opt-in, via `-C` on the tool or `gmpecpp_config_set_backend()`.

Only the ECM stage is accelerated. That is not a design choice — it is
the only part of this prover CGBN can express, since CGBN provides
fixed-width *integer* arithmetic and the class polynomial layer is
arbitrary-precision *floating point*. But ECM is also the part that
matters: it dominates the runtime of every proof large enough to be
interesting.

### Where the time goes

Per-phase instrumentation, exclusive time, seed pinned. The split depends
strongly on `-p`, which is why the default matters so much:

| Phase | 30 digits | 60 digits | 100 digits |
|---|---:|---:|---:|
| **At `-p 1000` (the default)** | | | |
| ECM factoring | 51.0% | 98.2% | 99.5% |
| Hilbert class polynomials | 35.6% | 1.5% | 0.1% |
| everything else | 12.4% | 0.3% | 0.4% |
| total | 0.062 s | 3.859 s | 19.423 s |
| **At `-p 10000`** | | | |
| ECM factoring | 1.0% | 28.9% | 89.8% |
| Hilbert class polynomials | 84.3% | 67.5% | 7.8% |
| everything else | 14.7% | 3.6% | 2.4% |
| total | 3.015 s | 12.530 s | 21.085 s |

Regenerate with `make bench-profile`.

Read the ECM rows across: **0.032s, 3.790s, 19.320s** at 1000 bits versus
**0.032s, 3.619s, 18.934s** at 10000. ECM's absolute cost is unchanged,
because it is integer work and precision does not touch it. What changed
is everything around it — lowering the default made polynomial
construction cheap, and that is what turned ECM into the thing a proof
spends its time on.

Two consequences follow, and they point in opposite directions:

- The class polynomial layer, at 66–89% of runtime under the old default,
  was the obvious target for acceleration and is the one CGBN cannot
  touch: it is arbitrary-precision *floating point*, and CGBN is
  fixed-width *integer*.
- Under the current default that layer is 0.1–35% and ECM is the
  bottleneck — which CGBN handles well. Lowering the precision default
  is what made the GPU backend worth having.

### What the GPU does

Stage-1 ECM on Montgomery curves `B*y^2 = x^3 + A*x^2 + x` with Suyama's
parametrisation, worked in XZ coordinates so the inner loop needs no
modular inversion — an inversion per ladder step would dominate
everything else. All arithmetic is in Montgomery representation. One CGBN
instance per curve, spread over TPI cooperating threads, thousands of
curves per launch against the CPU implementation's twenty per bound.

Instance width is fixed at compile time, so the backend picks the
narrowest that holds the number: 512 bits up to 480-bit inputs, 1024 up to
992, 2048 up to 2016, 4096 up to 4064, 8192 up to 8160, 16384 up to
16352. Wider instances cost more than proportionally. The widths above 2048
are on the resumable kernels only (the prover's GPU path and
`gmpecpp_cuda_ecm()`); the unused one-shot stage-1 kernel stops at 2048.
16384 is the practical ceiling with CGBN: an instance spans at most one
32-thread warp, and at that width the hot kernels already use 236-248 of
the 255 registers a thread may have.

When the prover escalates to the GPU (`gmpecpp_ecm_factor()` and ECM inside a
proof, after the CPU attempt fails), it now runs **stage 1 and stage 2** on
the same resumable kernels as the standalone `gmpecpp_cuda_ecm()` below, with
B2 = 100·B1 below B1 = 5·10^4 and 50·B1 from there (B1 still clamped to
[1000, 10^5], 4096 curves by default). Measured over the 36 CUDA proofs of
the sweep below (90-180 digits, seeds 1 and 5, RTX PRO 4500): **117.9 s
against 145.9 s with stage 1 only**, all proven, every certificate verified
on both backends. The gain sits where the GPU is reached: 120 digits 4.8 →
2.9 s, 150 digits 9.9 → 3.2 s; small inputs are unchanged within ±0.3 s.
A full wave of curves (`-c 11808`) gave 121.5 s, no better than 4096.
The standalone `gmpecpp_cuda_ecm()` differs in having no CPU attempt, no
lock, no B1 clamp, and a stop callback.

#### The kernel is latency bound, not throughput bound

Stage 1's cost tracks `B1` — the serial chain of prime powers each curve
must walk — and barely notices how many curves you ask for. Measured with
stage 1 only (RTX 4080), sixteen times the curves cost 1.4× the time:
0.227 s for 256 curves and 0.354 s for 4096 at B1 = 10^4, 1.119 s and
1.802 s at 5·10^4.

The prover's GPU path now adds stage 2 when stage 1 fails, which blurs
that picture. `make bench-kernel` on the RTX PRO 4500 (GPU marginal cost:
a CUDA run minus a host-only run of the same work, on a 46-digit modulus
with a 20-digit factor):

| B1 | 256 curves | 1024 curves | 4096 curves |
|---:|---:|---:|---:|
| 100 | 0.057 s | 0.057 s | 0.069 s |
| 1 000 | 0.056 s | 0.057 s | 0.069 s |
| 10 000 | 0.483 s | 0.489 s | 0.337 s |
| 50 000 | 1.233 s | 1.209 s | 1.504 s |

At B1 = 10^4, 4096 curves came out *cheaper* than 256: with more curves
stage 1 more often finds the factor, and stage 2 is skipped. Cost still
tracks B1, and curves are still nearly free until the device is full
(about one wave, 11 808 curves at 512 bits here); `make bench-kernel`
warms the device first so that the one-off context creation does not land
in the first cell.

That shape is exactly what ECM wants: curves are independent trials and
one success ends the search, so being able to run thousands for the price
of hundreds is the whole game. It is also why raising `-c` is cheap and
raising `B1` is not.

### Measured results

All figures below were measured (November 2026, with GPU stage 2 in the
prover's path) on:

| | |
|---|---|
| GPU | NVIDIA RTX PRO 4500 Blackwell, sm_120, 32 104 MiB |
| CUDA | 13.2 (nvcc V13.2.78), driver 595.71.05 |
| Compiler | g++ 14.3.1, `-O2` |
| CGBN | headers from `/usr/local/include/cgbn` |
| Settings | seed pinned to 1, default precision, `-c 4096 -b 50000` |

Reproduce them with `make bench`, `make bench-cuda` and `make bench-reach`.
Reruns are only comparable at a pinned seed; treat small differences as
noise and large ones as real. The previous tables (RTX 4080 laptop, GPU
stage 1 only) showed the same pattern: 28.8× at 140 digits, the 36-proof
sweep 2.67× in aggregate. Speedups do not compare across the two machines,
whose CPUs differ (the 140-digit proof took 53 s on the host there, 68 s
here); the like-for-like comparison is on this machine, where the sweep's
36 CUDA proofs took 145.9 s with stage 1 only and 117.9-120.8 s with
stage 2.

#### Whole proofs — this is `make bench`

| Digits | CPU | CUDA | Speedup | CPU ECM | CUDA ECM |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.008 s | 0.008 s | 1.01× | 0.004 s | 0.004 s |
| 30 | 0.110 s | 0.108 s | 1.02× | 0.064 s | 0.065 s |
| 40 | 0.173 s | 0.175 s | 0.99× | 0.156 s | 0.158 s |
| 60 | 0.209 s | 0.209 s | 1.00× | 0.159 s | 0.159 s |
| 90 | 2.698 s | 1.067 s | 2.53× | 2.580 s | 0.973 s |
| 110 | 6.574 s | 2.009 s | 3.27× | 6.373 s | 1.856 s |
| 140 | 67.994 s | 2.660 s | **25.57×** | 67.787 s | 2.444 s |

The last two columns show the whole difference is the factoring stage,
which is the only thing the backend touches.

Below about 60 digits the CPU splits the cofactors in milliseconds and
there is nothing to win; the CUDA path costs nothing there because the
gate keeps it from being reached at all.

#### The full sweep --- 36 proofs, 90 to 180 digits

`make bench-cuda`, over every prime of 90 digits or more in
`bench/primes.txt`, at two seeds each:

| Digits | Seed | CPU | CUDA | Speedup |
|---:|---:|---:|---:|---:|
| 90 | 1 | 1.56 s | 1.06 s | 1.47× |
| 90 | 5 | 1.84 s | 1.23 s | 1.49× |
| 90 | 1 | 3.04 s | 3.23 s | 0.94× |
| 90 | 5 | 2.25 s | 3.53 s | 0.64× |
| 90 | 1 | 14.93 s | 1.09 s | 13.71× |
| 90 | 5 | 14.58 s | 1.16 s | 12.60× |
| 90 | 1 | 2.66 s | 1.25 s | 2.12× |
| 90 | 5 | 3.66 s | 1.41 s | 2.59× |
| 100 | 1 | 24.30 s | 1.67 s | 14.56× |
| 100 | 5 | 23.29 s | 3.37 s | 6.92× |
| 100 | 1 | 0.83 s | 1.04 s | 0.79× |
| 100 | 5 | 1.39 s | 1.73 s | 0.80× |
| 100 | 1 | 4.91 s | 1.57 s | 3.12× |
| 100 | 5 | 4.53 s | 1.61 s | 2.81× |
| 100 | 1 | 2.92 s | 2.21 s | 1.32× |
| 100 | 5 | 1.32 s | 2.29 s | 0.58× |
| 100 | 1 | 7.27 s | 0.91 s | 8.01× |
| 100 | 5 | 6.51 s | 0.94 s | 6.93× |
| 110 | 1 | 6.53 s | 2.34 s | 2.79× |
| 110 | 5 | 6.27 s | 2.28 s | 2.75× |
| 110 | 1 | 6.03 s | 1.81 s | 3.33× |
| 110 | 5 | 5.40 s | 1.25 s | 4.34× |
| 120 | 1 | 6.38 s | 3.00 s | 2.13× |
| 120 | 5 | 3.31 s | 3.47 s | 0.95× |
| 130 | 1 | 9.16 s | 6.46 s | 1.42× |
| 130 | 5 | 5.97 s | 6.46 s | 0.92× |
| 130 | 1 | 3.19 s | 5.58 s | 0.57× |
| 130 | 5 | 4.24 s | 2.64 s | 1.61× |
| 140 | 1 | 67.25 s | 2.98 s | **22.56×** |
| 140 | 5 | 58.30 s | 3.15 s | **18.50×** |
| 140 | 1 | 6.02 s | 2.81 s | 2.14× |
| 140 | 5 | 13.97 s | 2.47 s | 5.66× |
| 150 | 1 | 33.30 s | 3.31 s | 10.06× |
| 150 | 5 | 25.06 s | 3.37 s | 7.44× |
| 180 | 1 | 19.32 s | 19.70 s | 0.98× |
| 180 | 5 | 19.16 s | 16.37 s | 1.17× |

| | CPU | CUDA |
|---|---:|---:|
| Total | 420.6 s | **120.8 s** |
| Fastest run | 0.83 s | 0.91 s |
| Slowest run | **67.25 s** | **19.70 s** |
| Faster in | 9 of 36 | **27 of 36** |

Aggregate **3.48×**, median **2.37×**, range 0.57× to 22.56×.

The aggregate understates the mechanism. The real effect is **variance
reduction**: host times spread over almost two orders of magnitude, 0.83 s
to 67.3 s, while GPU-assisted times stay within 0.91 s to 19.7 s. The GPU
does not make every proof faster so much as it removes the pathological
ones, where the host's twenty curves per bound grind against a cofactor
that thousands of curves split immediately. Every run the backend loses
is one where the host took 6.5 s or less, except one 180-digit run
(19.3 s against 19.7 s).

At 180 digits the two are level (0.98× and 1.17×). Reach is not
unbounded: past some size the cofactors stop being ones these bounds
split, on either backend.

#### Reach: what each backend can split at all

Semiprimes, indexed by the size of the smaller factor, B1=50000, 4096
curves, size gate disabled so the GPU is always tried:

| Factor | CPU | CUDA |
|---:|---|---|
| 8 digits | found, 0.037 s | found, 0.037 s |
| 10 digits | found, 0.048 s | found, 0.047 s |
| 12 digits | found, 1.750 s | found, 1.749 s |
| 14 digits | **not found**, 2.078 s | found, 3.691 s |
| 16 digits | **not found**, 2.261 s | found, 3.771 s |
| 18 digits | **not found**, 2.931 s | found, 4.445 s |
| 20 digits | **not found**, 3.009 s | found, 4.501 s |

From roughly 14-digit factors upward the CPU implementation gives up
where the GPU succeeds. (The CUDA column includes the failed CPU attempt
that precedes it, which is why it looks slower on the rows where both
succeed — the GPU is not reached there at all. Where it is reached, it
runs stage 2 as well; a 25-digit factor, beyond this table, is found at
B1 = 5·10^4 on 0.39% of GPU curves with stage 2 against 0.013% without.)

This is what the whole-proof numbers are made of: in ECPP terms, a
discriminant the CPU would abandon for want of a factorisation is one the
GPU can use, which is why its proofs are sometimes *shorter* as well as
faster.

### Standalone GPU ECM: `gmpecpp_cuda_ecm()`

```c
gmpecpp_status_t gmpecpp_cuda_ecm(const char *n, unsigned long b1,
                                  unsigned long b2, unsigned int curves,
                                  unsigned long seed,
                                  int (*stop)(void *), void *stop_arg,
                                  char **factor);
unsigned int gmpecpp_cuda_ecm_curves(const char *n);
```

ECM on the GPU and nothing else, for programs that factor rather than
prove (it was added for the trial-division factorizers of
[primefactors](https://github.com/steleman/primefactors), which run it on
every GPU alongside GMP-ECM on the CPU). Differences from
`gmpecpp_ecm_factor()`:

- **No CPU attempt and no global lock.** Several threads may call it at
  once, each on its own device or sharing one; it runs on the calling
  thread's current CUDA device.
- **A full GPU of curves.** `curves = 0` means one wave: as many as the
  device runs at once, from the occupancy API (11 808 at 512 bits,
  5 904 at 1024, 2 952 at 2048, 1 968 at 4096, 984 at 8192, 656 at
  16384 on an RTX PRO 4500 Blackwell).
  `gmpecpp_cuda_ecm_curves()` reports that number.
- **Resumable launches.** Each curve's state stays in device memory and
  stage 1 runs as launches of about a quarter of a second, so `stop` is
  polled between launches (return non-zero to abandon the search) and
  `b1` is not capped (it must be in [1000, 2^32)).
- **Stage 2 when `b2 > b1`**: prime continuation with D = 210. Each curve
  keeps its 24 baby points [j]Q (odd j < 105, coprime to 210), normalized
  once to x_j = X_j/Z_j; it walks the giant points [mD]Q by one
  differential addition each and multiplies X_m − x_j·Z_m into an
  accumulator for every prime q = mD ± j in (B1, B2] — two modular
  multiplications per prime. The pairs to take are the same for every
  curve, so the host builds them once (one 24-bit mask per m, from a
  segmented sieve). A failed inversion while normalizing exposes a
  factor directly.

It returns `GMPECPP_COMPOSITE` with a factor (free it with
`gmpecpp_string_free`), `GMPECPP_PROBABLY_PRIME` when nothing was found —
which, as everywhere here, is not evidence of primality — and
`GMPECPP_ERR_INVALID_INPUT` (even, below 5, wider than 16352 bits, or `b1`
out of range) or `GMPECPP_ERR_INTERNAL` (no device, a CUDA error).

Measured on an RTX PRO 4500 Blackwell (CUDA 13.2, `sm_120`), 400-bit n:

| | `gmpecpp_ecm_factor`, CUDA backend | `gmpecpp_cuda_ecm` |
|---|---:|---:|
| B1 = 10^4, one call | 2.04 s (1.36 s of it the CPU attempt; kernel 0.34 s) | 0.65 s, 11 808 curves |
| B1 = 5·10^4, one call | 8.92 s (6.99 s CPU attempt; kernel 1.52 s) | 2.6 s |

Stage 2, on the same curves: per-curve success on a 20-digit factor at
B1 = 10^4 rises from 0.00068 (stage 1) to 0.0099 at B2 = 50·B1, matching
GMP-ECM's 0.0129 with the same B2. Normalizing the baby points (three
multiplications per prime down to two) left every hit unchanged and made
stage 2 1.4× faster. Factors found per second, one wave:

| | B2 = 25·B1 | 50·B1 | 100·B1 | 200·B1 |
|---|---:|---:|---:|---:|
| 20-digit factor, B1 = 10^4 (3 mults/prime) | 108 | 120 | **132** | 109 |
| 30-digit factor, B1 = 2.5·10^5 (2 mults/prime) | — | **0.47** | 0.41 | 0.42 |

The best B1 for each factor size is GMP-ECM's own (25 digits: 5·10^4;
30: 2.5·10^5, 1.7 s per find; 35: 10^6, 17 s per find), and at those
settings one GPU finds about 9× as many 30-digit factors per second as
GMP-ECM on 31 CPU threads, and 4–9× as many 35-digit ones. One wave is
the right size: half a wave finds 21% less per second, two waves the
same as one at twice the time per call.

`GMPECPP_CUDA_ECM_STATS=1` in the environment makes it report on stderr,
per stage, how many curves found a factor and how long the stage took —
in that mode stage 2 runs even after a stage-1 find, so the counts are
cumulative. That is how the success rates above were measured; normal
runs stop at the first factor.

### Why the GPU is not tried first

(This is about the prover's path; `gmpecpp_cuda_ecm()` above is GPU only by design.)
Because most ECM calls in a proof are *meant* to fail. `check_for_factor`
is asked about many candidate curve orders, most of which have no large
enough prime factor, and rejecting them quickly is the point. An early
version handed every failed CPU attempt to the GPU and made proofs 2×
slower.

The backend therefore runs the CPU attempt first and escalates to the GPU
only when it fails *and* the cofactor is at least `cuda_min_bits` (192 by
default) wide. Set it to 0 to try the GPU on every failed attempt.

### Beyond 16352 bits: `libgmpecmfft` and `gmpecmfft`

CGBN cannot go past 16384-bit instances (one warp per number, and the
registers are already full there). `libgmpecmfft` is a separate ECM
library, with a command line tool, for numbers of any size:

```sh
echo N | ./gmpecmfft -1 10000 -2 1000000 -v    # factor on stdout
./gmpecmfft --selftest 4 N                     # FFT arithmetic vs GMP
```

Up to 16352 bits it calls `gmpecpp_cuda_ecm()`, so CGBN does the work.
Above that, every modular multiplication is Montgomery's REDC with the
three integer products done as double precision cuFFT convolutions,
batched across all curves (`src/fftmod.cu`). The curves and the work on
them are the CGBN path's: the same Suyama sigmas from the same seed, the
prime powers up to B1, and a stage 2 over the same giant and baby
points.

Stage 2 is polynomial (Montgomery's FFT continuation, as in GMP-ECM)
when B2 is large enough: F(X) = prod (X - x_j) over the baby points,
the giant points in blocks G(X) = prod (Z_m X - X_m), H = prod G mod F,
and finally prod_j H(x_j) by a scaled remainder tree -- every pair of a giant
and a baby point, at a cost growing with sqrt(B2) rather than with the
number of primes. Polynomial products are Kronecker substitutions into
one long cuFFT convolution per curve (`src/fftpoly.cu`). It covers every
number coprime to D up to (a little past) B2, primes included, so it
finds at least what the per-prime stage 2 finds. Below that B2, or
when it does not fit in GPU memory, stage 2 goes prime by prime over
CGBN's exact pairs. `GMPECMFFT_STAGE2=prime` or `=poly` forces one.

The arithmetic checks itself as it goes. The digit size is chosen so a
convolution coefficient stays below 2^44 in the worst case, every
coefficient's rounding error is measured, and a run whose worst error
exceeds 0.25 fails with `GMPECMFFT_ERR_PRECISION` instead of returning
untrusted results (correct transforms measure around 0.0005). Every
returned factor is a proper divisor checked with GMP.

`make check-fft` validates it five ways, each on identical curves (and
`gmpecmfft-mpi` and `gmpecmfft-slurm`, below, two more):

- the FFT arithmetic against GMP, odd moduli from 200 to 10^6 bits;
- FFT path against CGBN at 2000 bits (`gmpecmfft_ecm_fft()` forces the
  FFT path below the CGBN limit): stage 1 and (per-prime) stage 2 hit
  counts must be identical;
- the polynomial stage 2 against a replay of the same computation on
  the host with GMP (`GMPECMFFT_S2_CHECK=1`): the product over all pairs
  must be identical, value for value;
- FFT path against GMP-ECM at 20000 bits, which CGBN cannot reach.
  `gmpecmfft --sigmas K` prints the curves' sigmas, and
  `gmp-ecm -sigma 0:S` builds the same curve.

The API (`include/gmpecmfft.h`) mirrors `gmpecpp_cuda_ecm()`:
`gmpecmfft_ecm()`, `gmpecmfft_ecm_fft()`, `gmpecmfft_ecm_curves()` and
`gmpecmfft_selftest()`, with status codes that are also the tool's exit
codes (0 factor found, 1 none, 2 stopped, 3 and up errors).
`GMPECMFFT_STATS=1` prints per-stage hit counts, the FFT geometry and the
worst rounding error.

- polynomial stage 2 in curve groups (as D = 30030 needs when all
  curves do not fit at once): the same hits in groups of 7 as in one;

Its speed has not been benchmarked with a committed harness yet, so no
figures are given here. The FFTs are double precision, so cuFFT's FP64
throughput is what bounds it: that is low on the RTX PRO 4500 and 6000
and full rate on the B200.

#### Across GPUs and nodes: `gmpecmfft-mpi`

ECM curves are independent trials, and the first to find a factor ends
the search, so a cluster's GPUs can share one number's curves with
almost nothing crossing the network -- the number, the bounds, and which
curves to run. `gmpecmfft-mpi` (Open MPI) does that: rank 0 coordinates,
every other rank runs ECM on one GPU (its index among the workers on its
node).

```sh
echo N | mpirun -n 9 ./gmpecmfft-mpi -1 1e6 -2 5e7 -v   # 8 GPUs
```

- **Batches, handed out on request.** A batch is one `gmpecmfft_ecm()`
  call on a worker's GPU, one wave of curves by default (`-c` sets it).
  A worker's result is its request for the next batch, so no GPU waits
  for another. Batch *i* draws its curves from seed *S* + *i* (`-s S`,
  default 1): batches are distinct, and any one is reproduced on a
  single GPU with `gmpecmfft -s S+i -c C` -- batch 0 *is* `gmpecmfft -s
  S`.
- **Stopping.** When a factor turns up (or after `-b K` batches, or on
  Ctrl-C) every worker is told to stop, and a busy one abandons its batch
  within about a quarter second (the library's stop callback polls for
  the message). The factor goes to stdout; `-v` adds a line per batch and
  a summary naming the batch and the single-GPU command that reproduces
  it.
- **Failures.** A worker without a usable GPU, or whose batch fails (out
  of memory, say), is retired and its batch handed to another; only if
  every worker fails does the run fail. Exit codes are `gmpecmfft`'s.
- **Use one kind of GPU throughout.** With a mix, the faster GPUs finish
  and then sit idle at the end while the slower ones complete their
  batches.

Expected time to a factor falls in proportion to the number of GPUs
until the cluster runs, in one batch each, the expected number of curves
for that factor size; past that, more GPUs no longer help, since a
batch cannot finish faster than one curve. `make check-fft` runs it with
two workers on the local GPU and requires the factor from exactly the
batch a single-GPU run predicts.

#### From a Slurm job array: `tools/gmpecmfft-slurm`

The same batches without MPI or a coordinator: one array task per GPU,
coordinated through a run directory on a filesystem every node sees.

```sh
echo N | tools/gmpecmfft-slurm submit -d /shared/run1 -n 8 -1 1e6 -2 5e7 -c 256 \
           -- --partition=gpu --time=24:00:00
tools/gmpecmfft-slurm status /shared/run1    # factor, batches, curves
```

- `submit` records the number and bounds in the directory and runs
  `sbatch --array=0-(n-1) --gres=gpu:1` (anything after `--` goes to
  sbatch; `--dry-run` prints the command instead).
- Array task *t* of *T* runs batches *t*, *t* + *T*, *t* + 2*T*, ... --
  batch *b* is `gmpecmfft -s S+b -c C`, the very batches
  `gmpecmfft-mpi` runs -- for `-b K` batches each, or until a factor is
  found. Each appends a line per batch to `DIR/results.t`.
- The first task to find a factor claims it (`mkdir DIR/found`, atomic),
  writes `DIR/found/factor`, creates `DIR/STOP`, and `scancel`s the
  array; running tasks stop on Slurm's SIGTERM. Where `scancel` is not
  allowed, running batches stop within a quarter second of `DIR/STOP`
  appearing anyway (`gmpecmfft --stop-file`), and later tasks do not
  start a batch.
- Fix `-c`: with the default (one wave of whatever GPU a task lands on)
  a batch's curves depend on the GPU, and reproducing it on one GPU
  needs the same count.

Compared with `gmpecmfft-mpi` it needs nothing but Slurm and a shared
filesystem, and tasks can start and finish independently (or be
requeued); the split is static, though, so it suits a cluster of one GPU
type even more strictly. `make check-fft` runs two of its tasks by hand
(no Slurm needed) and requires the factor from the predicted batch.
[`CGBN-VS-GMP.md`](CGBN-VS-GMP.md) measures whether the host's remaining
GMP work should move to CGBN (it should not: only batched work gains),
and what `-C` is worth to `gmpecpp-mpi` in batch proving.
[`SLURM.md`](SLURM.md) gives Slurm setup steps for a one-GPU machine and
for a multi-node GPU cluster, for this tool and for running
`gmpecmfft-mpi` under Slurm.

## Parallelism

An ECPP downrun is inherently sequential — step *i*+1 proves the prime
that step *i* produced — so there is no coarse-grained parallelism inside
a single proof. What the algorithm does offer is two finer axes, and they
suit different tools.

| Site | Independent? | Share of runtime | Fan-out | Used |
|---|---|---:|---|---|
| ECM curves | fully; one success ends the search | 52–99.5% | unbounded | **POSIX threads** (`-T`) |
| Discriminant sweep | until the first success | ~100% | up to `Dmax` | **POSIX threads** (`-S`) |
| Whole numbers | fully | — | unbounded | **MPI** |
| *m* candidates per *D* | fully | within ECM | 2–6 | not yet |
| *j*(τ) per reduced form | fully, then a reduction | 0.1–85% | *h(D)* | not yet |
| Root finding (`Recurse`) | two subtrees | <1% | 2 | not worth it |
| The downrun itself | **no** — each step needs the last | — | 1 | — |

### Threads: `-T N`

ECM is where the time goes and its curves are independent trials, so the
twenty attempts per bound become twenty parallel attempts. `-T 0` uses
one thread per CPU. Each worker owns a `gmp_randstate_t`, which is the
only GMP state the prover mutates rather than merely reads; everything
else in the inner loop works on locals.

Measured on 32 cores over all eighteen primes of 90 digits or more in
`bench/primes.txt`, `make bench-threads`:

| Threads | Total | Speedup |
|---:|---:|---:|
| 1 | 171.6 s | 1.00× |
| 2 | 135.0 s | 1.27× |
| 4 | 54.9 s | 3.12× |
| 8 | 41.9 s | 4.09× |
| 16 | 31.9 s | 5.38× |
| 32 | 28.7 s | **5.98×** |

Scaling is sublinear in aggregate and flattens past sixteen threads,
but *individual* inputs frequently beat the thread count — one 90-digit
prime goes 11.63 s to 2.52 s on two threads, and 52.63 s to 2.72 s on
four. Both facts have the same cause: ECM is a Las Vegas algorithm, so
concurrent curves finish in the time of the *luckiest* rather than the
average. That produces enormous wins where the sequential search was
unlucky and nothing at all where it was not, which averages out to the
column above. Past roughly twenty curves per bound the extra trials are
mostly speculative and stop paying, which is where the flattening comes
from.

An earlier version of this table, measured over only five primes,
reported 6.96×. The eighteen-prime figure above is the honest one; a
small sample of a high-variance quantity flatters whichever end it
happens to catch.

Unlike the CUDA backend this has no per-launch cost, so it is worth
enabling at any size. Like the CUDA backend it changes *which* factor is
found, so certificates differ from a single-threaded run while remaining
valid — verify them rather than diffing them. `-T 1` reproduces the
historical program byte for byte.

### The discriminant sweep: `-S N`

Each discriminant is an independent trial, and the work under one is
essentially a whole proof step — the factoring stage and the class
polynomial construction both sit beneath it. So this is speculative
parallelism: run several *D* at once and keep the best.

Two properties make it usable rather than merely fast.

**It stays deterministic.** Each attempt seeds its own random stream from
the discriminant, so `try_discriminant(D)` is a pure function of
(*D*, *N*, seed). Output does not depend on scheduling: a run at 32
threads is byte-identical to a run at 2.

**The smallest *D* still wins.** Workers draw monotonically increasing
indices; a success at index *k* stops new work beyond *k* while
in-flight attempts below it finish. Taking the smallest success means the
sweep picks the discriminant the sequential walk would have picked —
which matters, because a larger |*D*| means a larger class number and a
costlier proof step. Racing to "any success" would often choose a worse
curve and give the speedup straight back.

`make bench-sweep`, over the same eighteen primes as the thread table
above so the two are directly comparable:

| Threads | Total | Speedup |
|---:|---:|---:|
| 1 | 172.3 s | 1.00× |
| 2 | 133.6 s | 1.29× |
| 4 | 122.3 s | 1.41× |
| 8 | 119.9 s | 1.44× |
| 16 | 118.7 s | 1.45× |
| 32 | 118.5 s | **1.45×** |

**This is a much weaker result than threading ECM — 1.45× against 5.98×
on identical inputs — and the reason is structural.** Most inputs succeed
on one of the first few discriminants, so there is simply not much sweep
to parallelise; and because the smallest success wins, everything the
other workers computed is discarded. That discarding is deliberate, not
waste to be optimised away: it is what buys the cheaper curve.

The gain is concentrated and the losses are real. Where the sweep is long
it pays well — 52.98 s to 23.68 s at 140 digits, 19.08 s to 10.14 s at
100 — and one 100-digit case improves 2.29 s to 0.93 s. But several
inputs get slightly *slower*, by a few percent to about twenty:
1.21 → 1.41 s, 2.39 → 2.85 s, 5.11 → 5.78 s. Those are the ones whose
first discriminant already worked, where the fan-out has nothing to find
and only costs thread setup.

It also plateaus at four threads, and nothing above that helps. If you
have cores to spend, spend them on `-T`.

An earlier version was far worse, turning 0.66 s into 2.29 s, because
thread creation cost more than the sweep it was accelerating. The
smallest few discriminants are now probed on the calling thread before
any fan-out, which costs nothing in the parallel case and keeps `-S` from
losing badly to `-S 1`.

Prefer `-T` over `-S` when only one applies; setting both oversubscribes,
since each sweep worker runs its own ECM, so `-S` wins and forces `-T 1`.

### MPI: batch mode

Whole numbers are embarrassingly parallel with no communication until the
end, which is what MPI is for. `gmpecpp-mpi` distributes them, handing
out work dynamically rather than partitioning it up front — proof times
vary by two orders of magnitude between neighbouring inputs, so a static
split would leave most ranks idle behind one straggler.

```sh
mpirun -n 10 ./gmpecpp-mpi -T 4 bench/primes.txt
```

Ten primes of 90–110 digits, wall clock:

| Configuration | Wall | Speedup |
|---|---:|---:|
| 1 rank, 1 thread | 53.78 s | 1.00× |
| 5 ranks, 1 thread | 20.47 s | 2.63× |
| 9 ranks, 1 thread | 20.45 s | 2.63× |
| 5 ranks, 4 threads | 5.68 s | 9.47× |
| 9 ranks, 4 threads | 4.89 s | **11.00×** |

**MPI alone plateaus at 2.63×, and adding ranks past five changes
nothing.** The batch's wall time is bounded by its slowest single proof,
and no number of ranks can split one proof. Threads *can*, because they
work inside it — which is why the hybrid reaches 11× where either tool
alone stalls. This is the case for combining them rather than picking
one.

### What is not parallelised, and why

**Class polynomial construction** parallelises as a map over reduced
forms followed by a reduction, and each *j*(τ) is expensive. It is worth
doing at high `-p`, where that layer dominates; at the 1000-bit default
it is 0.1–37% of runtime and ECM is the better target.

## Tests

```
make check              # exit-code assertions plus certificate verification
make check SEED=7       # same, under a different seed
```

`make check` covers the trial-division path, a full ECPP downrun, a
composite, the same three through the CUDA backend when it is built, and
independent verification of a 60-digit certificate.

### Verifying certificates

The point of a primality *proof* is that it can be checked without
trusting the prover, so `gmpecpp-verify` shares no code with it — its
elliptic curve arithmetic is written against plain GMP:

```
./gmpecpp-verify -v 100000000000000000000000000000000000000000000000000000000019
./gmpecpp-verify -C <number>          # prove with CUDA, check on the CPU
./gmpecpp-verify --selftest <number>  # confirm it rejects a bad certificate
```

It re-derives every step: chain linkage, `q | m`, the Goldwasser–Kilian
bound `q > (n^(1/4)+1)^2`, Hasse's bound on `m`, that `P` is on the curve,
that `[m/q]P` equals the recorded `P2` and is not the point at infinity,
that `[q]P2` is, and finally that the chain bottoms out at a `q` small
enough to settle by trial division. `--selftest` checks a certificate
against the wrong starting number and requires the rejection — a verifier
nobody has watched reject anything is not evidence.

#### How the CUDA backend is checked

Two regimes, and they need different tests.

**Below the 192-bit gate** the GPU is never reached, so output must be
byte-identical to the original program. It is: 156 runs across 13 flag
combinations, 4 inputs and 3 seeds, comparing stdout, stderr and exit
code, with `-C` on and with it off.

**Above the gate** the GPU finds *different* factorisations, so the
certificates legitimately differ in content and in length. Byte
comparison is the wrong test there and would report failures that are
not. Validity is the right test:

| Digits | CPU steps | CPU time | CUDA steps | CUDA time | Both verified |
|---:|---:|---:|---:|---:|:---:|
| 90 | 11 | 2.02s | 12 | 0.84s | yes |
| 90 | 8 | 11.39s | 8 | 0.69s | yes |
| 100 | 10 | 0.62s | 10 | 0.70s | yes |
| 100 | 10 | 3.75s | 7 | 1.10s | yes |
| 110 | 12 | 4.97s | 8 | 1.62s | yes |
| 110 | 12 | 4.61s | 13 | 1.22s | yes |
| 130 | 13 | 2.43s | 12 | 5.04s | yes |
| 130 | 14 | 6.99s | 12 | 4.71s | yes |
| 140 | **15** | **51.88s** | **8** | **1.90s** | yes |
| 140 | 13 | 4.57s | 10 | 3.61s | yes |

Twenty certificates, all independently verified. The step counts are the
interesting column: a discriminant the CPU abandons for want of a
factorisation is one the GPU can use, so its proofs are often *shorter*
as well as faster. At 140 digits the CPU needed 15 steps and 51.9s where
the GPU found a better factorisation in 8 steps and 1.9s.

### Reproducing the measurements

Every table in this file is produced by a harness in `bench/`, run against
the primes and semiprimes in `bench/primes.txt` and
`bench/semiprimes.txt`. The data is committed so the numbers are
reproducible rather than merely asserted.

| Target | Regenerates | Runtime |
|---|---|---|
| `make bench` | whole-proof CPU vs CUDA | ~1 min |
| `make bench-reach` | largest factor each backend finds | ~20 s |
| `make bench-kernel` | GPU cost vs B1 and curve count | ~1 min |
| `make bench-profile` | per-phase time at both precisions | ~1 min |
| `make bench-precision` | the precision comparison | minutes |
| `make bench-cuda` | the full CPU vs CUDA sweep | many minutes |
| `make bench-all` | all of the above | |
| `make parity` | byte-identity vs the original program | ~2 min |
| `make verify-all` | certificates from both backends | minutes |

Each honours environment overrides, so a subset can be run without
editing anything:

```
make parity EXTRA=-C              # check the CUDA backend stays inert
DATA=my-primes.txt bash bench/cuda-sweep.sh
SEEDS="1 2 3" bash bench/precision-sweep.sh
python3 bench/profile-phases.py -p 4000
```

`bench/profile-phases.py` works by injecting an exclusive-time profiler
into a throwaway copy of the prover and building it, so it needs no
instrumentation in the shipped sources.

### Coverage gaps

`make check` passes even when the library was built for the wrong GPU
architecture: the CUDA smoke test quietly falls back to the CPU. Check the
build with `cuobjdump --list-elf libgmpecpp.so.1.0.0` (it must list the
device's `sm_` value) or a real GPU run such as `gmpecpp_cuda_ecm` with
`GMPECPP_CUDA_ECM_STATS=1`.

Inputs below 2^30 **never exercise ECPP** — they are settled by trial
division, so they prove nothing about the elliptic curve or class
polynomial code. Any change to that machinery needs an input above 2^30,
and preferably 30 or more digits so the chain actually recurses.

`make check` also pins the seed for reproducibility, which by
construction cannot expose seed-dependent bugs. Sweeping `SEED` is the
cheapest way to look for those.

## Limitations

- Class polynomial degree — the class number *h*(*D*) — must stay below 8192.
  There is no bounds check; exceeding it corrupts memory rather than reporting
  an error.
- Polynomials are fixed-size stack arrays, and the root-finding routine is
  recursive, at roughly 512 KB per frame.
- The prover gives up past `Dmax > 312500` or `Bmax > 2000000000`, reporting
  this as exit code 2 (see above).

## How it works

The implementation follows Cohen, *A Course in Computational Algebraic Number
Theory*, and is cross-referenced to it throughout: Atkin's test is Algorithm
9.2.4, with square roots mod *p* (1.5.1), modified Cornacchia (1.5.3), roots of a
polynomial mod *p* (1.6.1), Hilbert class polynomials (7.6.1), and Lenstra's ECM
(10.3.3). Weber polynomials follow Kaltofen–Yui and Atkin–Morain, with the
IEEE P1363 helper predicates, and Weber roots are converted to *j*-invariants by
the Kaltofen–Valente method.

## Provenance and licence

Descended from the GMP-ECPP program by Pate Williams (1997), James Wanless
(2008–19), Chris Studholme (2004), Elisavet Konstantinou, Yiannis Stamatiu and
Christos Zaroliagis (2003), Scott Contini (1996), Paul Zimmermann (2005), and
Timo Poikola (2010).

This copy has been reformatted to LLVM style at 70 columns with C++ comments;
the reformat was verified token-for-token against the original, so it is
functionally identical to its upstream.

Licensed under the **GNU General Public License, version 3 or later**
(`GPL-3.0-or-later`). The full text is in [LICENSE](LICENSE); the per-file
notice electing "version 3 ... or (at your option) any later version" is in the
header of `gmp-ecpp.cpp`. See also <https://www.gnu.org/licenses/>.
