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
| 20 | 0.02 s | 0.44 s | 29.2× |
| 30 | 0.06 s | 2.91 s | **46.6×** |
| 40 | 0.27 s | 2.50 s | 9.3× |
| 50 | 0.16 s | 3.01 s | 18.4× |
| 60 | 3.59 s | 12.18 s | 3.4× |
| 80 | 0.91 s | 6.82 s | 7.5× |
| 90 | 11.41 s | 21.27 s | 1.9× |
| 100 | 18.63 s | 20.56 s | 1.1× |
| 110 | 4.99 s | 10.67 s | 2.1× |
| 140 | 51.44 s | 60.37 s | 1.2× |
| 150 | 25.59 s | 30.98 s | 1.2× |
| 180 | 14.73 s | 17.71 s | 1.2× |

Across all 33: median **1.60×**, range 1.1× to 46.6×, aggregate
258.6 s → 177.3 s.

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

Link with `-lgmpecpp`. The library exports these 32 symbols and nothing
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
| `gmpecpp_config_set_log` | none, so the library is silent |
| `gmpecpp_config_set_verbosity` | 0 = silent |

**Proving**

| Function | Notes |
|---|---|
| `gmpecpp_prove` | decimal string in, certificate out |
| `gmpecpp_prove_mpz` | same, for an `mpz_t`; declared only if `<gmp.h>` precedes the header |
| `gmpecpp_preamble` | banner and constants, so an interactive caller can prompt afterwards |
| `gmpecpp_ecm_factor` | one round of ECM on its own, mainly for measurement |
| `gmpecpp_string_free` | frees what `gmpecpp_ecm_factor` returns |

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
build; `make cuda-info` shows what was detected. At runtime the backend is
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
| ECM factoring | 52.2% | 98.3% | 99.5% |
| Hilbert class polynomials | 37.1% | 1.4% | 0.1% |
| everything else | 9.4% | 0.3% | 0.4% |
| total | 0.056 s | 3.598 s | 18.746 s |
| **At `-p 10000`** | | | |
| ECM factoring | 1.0% | 28.8% | 89.9% |
| Hilbert class polynomials | 84.6% | 67.7% | 7.7% |
| everything else | 14.4% | 3.5% | 2.4% |
| total | 2.923 s | 12.239 s | 20.677 s |

Regenerate with `make bench-profile`.

Read the ECM rows across: **0.029s, 3.536s, 18.644s** at 1000 bits versus
**0.029s, 3.527s, 18.590s** at 10000. ECM's absolute cost is unchanged,
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
992, 2048 up to 2016. Wider instances cost proportionally more.

#### The kernel is latency bound, not throughput bound

Cost tracks `B1` — the serial chain of prime powers each curve must walk —
and barely notices how many curves you ask for:

| B1 | 256 curves | 1024 curves | 4096 curves |
|---:|---:|---:|---:|
| 100 | 0.022 s | 0.026 s | 0.033 s |
| 1 000 | 0.023 s | 0.024 s | 0.034 s |
| 10 000 | 0.242 s | 0.242 s | 0.346 s |
| 50 000 | 1.119 s | 1.186 s | 1.742 s |

Sixteen times the curves for 1.4× the time. Cost is reported as the GPU's
marginal contribution, measured as a CUDA run minus a host-only run of
the same work; `make bench-kernel` regenerates it, warming the device
first so that the one-off ~0.1 s of context creation does not land in the
first cell.

That shape is exactly what ECM wants: curves are independent trials and
one success ends the search, so being able to run thousands for the price
of hundreds is the whole game. It is also why raising `-c` is cheap and
raising `B1` is not.

### Measured results

All figures below were measured on:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 Laptop, sm_89, 11848 MiB |
| CUDA | 12.9 (nvcc V12.9.86), driver 575.64.03 |
| Compiler | g++ 14.3.1, `-O2` |
| CGBN | headers from `/usr/local/include/cgbn` |
| Settings | seed pinned to 1, default precision, `-c 4096 -b 50000` |

Reproduce them with `make bench`. Run-to-run variation is a percent or
two — a repeat of the table below gave 28.27× where it says 28.52× — so
treat the small differences as noise and the large ones as real. Reruns
are only comparable at a pinned seed.

#### Whole proofs — this is `make bench`

| Digits | CPU | CUDA | Speedup | CPU ECM | CUDA ECM |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.006 s | 0.006 s | 1.02× | 0.003 s | 0.003 s |
| 30 | 0.089 s | 0.084 s | 1.06× | 0.052 s | 0.050 s |
| 40 | 0.133 s | 0.133 s | 1.00× | 0.120 s | 0.120 s |
| 60 | 0.161 s | 0.161 s | 1.00× | 0.122 s | 0.122 s |
| 90 | 2.090 s | 0.853 s | 2.45× | 1.999 s | 0.780 s |
| 110 | 5.064 s | 1.566 s | 3.23× | 4.909 s | 1.450 s |
| 140 | 52.292 s | 1.840 s | **28.42×** | 52.132 s | 1.674 s |

The last two columns show the whole difference is the factoring stage,
which is the only thing the backend touches.

Below about 60 digits the CPU splits the cofactors in milliseconds and
there is nothing to win; the CUDA path costs nothing there because the
gate keeps it from being reached at all.

#### The full sweep --- 36 proofs, 90 to 180 digits

`make bench-cuda`, over every prime of 90 digits or more in
`bench/primes.txt`, at two seeds each:

| Digits | Seed | CPU | CUDA | Speedup | | Digits | Seed | CPU | CUDA | Speedup |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 90 | 1 | 1.19 s | 0.80 s | 1.48× | | 100 | 1 | 2.22 s | 1.67 s | 1.33× |
| 90 | 5 | 1.41 s | 1.28 s | 1.11× | | 100 | 5 | 1.02 s | 1.76 s | 0.58× |
| 90 | 1 | 2.36 s | 2.37 s | 0.99× | | 100 | 1 | 5.58 s | 0.66 s | 8.50× |
| 90 | 5 | 1.72 s | 2.59 s | 0.67× | | 100 | 5 | 4.97 s | 0.68 s | 7.26× |
| 90 | 1 | 11.40 s | 0.76 s | **15.01×** | | 110 | 1 | 5.00 s | 1.74 s | 2.88× |
| 90 | 5 | 11.19 s | 0.80 s | **13.94×** | | 110 | 5 | 4.78 s | 1.68 s | 2.85× |
| 90 | 1 | 2.04 s | 0.92 s | 2.21× | | 110 | 1 | 4.61 s | 1.30 s | 3.56× |
| 90 | 5 | 2.81 s | 1.02 s | 2.75× | | 110 | 5 | 4.14 s | 1.44 s | 2.87× |
| 100 | 1 | 18.58 s | 3.01 s | 6.18× | | 120 | 1 | 4.89 s | 3.90 s | 1.25× |
| 100 | 5 | 17.84 s | 3.29 s | 5.43× | | 120 | 5 | 2.54 s | 4.22 s | 0.60× |
| 100 | 1 | 0.65 s | 0.76 s | 0.85× | | 130 | 1 | 7.00 s | 4.79 s | 1.46× |
| 100 | 5 | 1.08 s | 1.28 s | 0.84× | | 130 | 5 | 4.56 s | 6.12 s | 0.74× |
| 100 | 1 | 3.76 s | 1.16 s | 3.24× | | 130 | 1 | 2.44 s | 5.12 s | 0.47× |
| 100 | 5 | 3.48 s | 1.21 s | 2.88× | | 130 | 5 | 3.24 s | 4.86 s | 0.67× |
| 140 | 1 | 51.49 s | 1.96 s | **26.25×** | | 150 | 1 | 25.52 s | 8.13 s | 3.14× |
| 140 | 5 | 44.66 s | 2.25 s | **19.87×** | | 150 | 5 | 19.19 s | 7.59 s | 2.53× |
| 140 | 1 | 4.58 s | 3.71 s | 1.24× | | 180 | 1 | 14.75 s | 15.70 s | 0.94× |
| 140 | 5 | 10.72 s | 4.10 s | 2.61× | | 180 | 5 | 14.66 s | 16.44 s | 0.89× |

| | CPU | CUDA |
|---|---:|---:|
| Total | 322.1 s | **121.1 s** |
| Fastest run | 0.65 s | 0.66 s |
| Slowest run | **51.49 s** | **16.44 s** |
| Faster in | 11 of 36 | **25 of 36** |

Aggregate **2.66×**, median **2.37×**, range 0.47× to 26.25×.

The aggregate understates the mechanism. The real effect is **variance
reduction**: host times span almost two orders of magnitude, 0.65 s to
51.5 s, while GPU-assisted times stay within a factor of 25, 0.66 s to
16.4 s. The GPU does not make every proof faster so much as it removes
the pathological ones, where the host's twenty curves per bound grind
against a cofactor that thousands of curves split immediately. Nearly
every run the backend loses is one where the host was already fast.

The 180-digit rows are the exception worth noting: there the backend
loses slightly on both seeds. Reach is not unbounded, and past some size
the cofactors stop being ones a fixed stage-one bound can split.

#### Reach: what each backend can split at all

Semiprimes, indexed by the size of the smaller factor, B1=50000, 4096
curves, size gate disabled so the GPU is always tried:

| Factor | CPU | CUDA |
|---:|---|---|
| 8 digits | found, 0.029 s | found, 0.028 s |
| 10 digits | found, 0.037 s | found, 0.037 s |
| 12 digits | found, 1.350 s | found, 1.351 s |
| 14 digits | **not found**, 1.655 s | found, 3.441 s |
| 16 digits | **not found**, 1.752 s | found, 3.486 s |
| 18 digits | **not found**, 2.247 s | found, 3.993 s |
| 20 digits | **not found**, 2.306 s | found, 4.046 s |

From roughly 14-digit factors upward the CPU implementation gives up
where the GPU succeeds. (The CUDA column includes the failed CPU attempt
that precedes it, which is why it looks slower on the rows where both
succeed — the GPU is not reached there at all.)

This is what the whole-proof numbers are made of: in ECPP terms, a
discriminant the CPU would abandon for want of a factorisation is one the
GPU can use, which is why its proofs are sometimes *shorter* as well as
faster.

### Why the GPU is not tried first

Because most ECM calls in a proof are *meant* to fail. `check_for_factor`
is asked about many candidate curve orders, most of which have no large
enough prime factor, and rejecting them quickly is the point. An early
version handed every failed CPU attempt to the GPU and made proofs 2×
slower.

The backend therefore runs the CPU attempt first and escalates to the GPU
only when it fails *and* the cofactor is at least `cuda_min_bits` (192 by
default) wide. Set it to 0 to try the GPU on every failed attempt.

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
