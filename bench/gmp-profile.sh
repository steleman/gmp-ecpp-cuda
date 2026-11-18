#!/bin/bash
# Where CPU time goes, by library, for the prover and for gmpecmfft
# (make bench-gmp-profile; needs perf, ~2 min). The share in libgmp is
# the most any GPU replacement of GMP could save.
#
#   - the prover on the first 140-digit prime of bench/primes.txt, seed
#     1, CPU only and with -C (CGBN ECM);
#   - gmpecmfft at 2000 bits (CGBN path, one wave of 2952 curves) and at
#     20000 bits (cuFFT path, 64 curves), B1 = 10^4, B2 = 10^6.
#
# For the prover, libgmp's samples are also split by kind: gcd and
# division (the affine point arithmetic, which inverts at every point
# addition), mpf_* (class polynomials), and the rest.
#
# Samples are taken at 1999 Hz of cpu-clock (a software timer: time, not
# cycles) on the calling thread and any it creates. On a hybrid CPU the
# default cycles event is two events, one per core type (cpu_core,
# cpu_atom), which perf report shows as separate tables; cpu-clock gives
# one.
# A thread blocked in a CUDA synchronization spins in libcuda, so the
# libcuda share is time spent waiting on the GPU.
set -u
cd "$(dirname "$0")/.."
command -v perf >/dev/null || { echo "perf not installed" >&2; exit 1; }
OUT=${1:-build/bench-gmp-profile}
mkdir -p "$OUT"
p=$(grep -v '^#' bench/primes.txt | awk '$1 == 140 {print $2; exit}')
share() { # perf data file: libraries above 0.5%
  perf report -q -i "$1" --sort dso 2>/dev/null | awk '$1+0 >= 0.5' |
    head -6 | sed 's/^/    /'
}
kinds() { # perf data file: libgmp samples by kind of work
  perf report -q -i "$1" --sort dso,sym 2>/dev/null | awk '
    NF >= 4 && $1 ~ /%$/ {
      p = $1 + 0; dso = $2; sym = $4
      if (dso ~ /libgmp/) {
        if (sym ~ /gcd|div2|hgcd|divrem|div_qr|bdiv|invert/) g += p
        else if (sym ~ /gmpf_|mpf/) f += p
        else o += p
      } else if (dso ~ /libcuda/) c += p
      else x += p
    }
    END {
      printf "    by kind: libgmp gcd/division %.1f%%, libgmp mpf_* %.1f%%, other libgmp %.1f%%, libcuda %.1f%%, the rest %.1f%%\n", g, f, o, c, x
    }'
}
for mode in cpu gpu; do
  f=""; [ $mode = gpu ] && f="-C"
  s=$( { /usr/bin/time -f %e perf record -q -e cpu-clock -F 1999 -o "$OUT/prover-$mode.data" \
        ./gmp-ecpp -Q -s 1 $f <<<"$p" >/dev/null; } 2>&1 | tail -1)
  echo "prover, 140 digits, $mode: $s s"
  share "$OUT/prover-$mode.data"
  kinds "$OUT/prover-$mode.data"
done
for b in 2000 20000; do
  n=$(awk -v b=$b '$1 == b {print $3}' bench/fft-numbers.txt)
  c=$([ $b = 2000 ] && echo 2952 || echo 64)
  s=$( { /usr/bin/time -f %e perf record -q -e cpu-clock -F 1999 -o "$OUT/fft-$b.data" \
        ./gmpecmfft -c $c -s 5 -1 10000 -2 1000000 <<<"$n" >/dev/null; } 2>&1 | tail -1)
  echo "gmpecmfft, $b bits, $c curves: $s s"
  share "$OUT/fft-$b.data"
  echo "    libgmp: $(perf report -q -i "$OUT/fft-$b.data" --sort dso 2>/dev/null |
    awk '/libgmp/ {s += $1} END {printf "%.2f%%", s}')"
done
