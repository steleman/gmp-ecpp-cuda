#!/bin/bash
# Correctness checks for libgmpecmfft (make check-fft). A few minutes.
#
#  1. FFT Montgomery arithmetic vs GMP, odd moduli from 200 to 10^6 bits.
#  2. ECM, FFT path vs CGBN on the same curves at 2000 bits: the stage 1
#     and stage 2 hit counts must be identical.
#  3. Polynomial stage 2 (B2 large enough to choose it, and D = 210
#     forced): its product over all pairs, for two curves, must be
#     identical to a host replay of the same computation with GMP.
#  4. Polynomial stage 2 in curve groups (GMPECMFFT_S2_GROUP=7, as
#     when D = 30030 does not fit all curves at once): the hit counts
#     must be the same as with all curves in one group.
#  5. ECM, FFT path vs GMP-ECM (-sigma 0:S, the same Suyama curves) at
#     20000 bits, where CGBN cannot go: stage 1 hit counts must match.
#     Skipped if gmp-ecm is not installed.
#  6. gmpecmfft-mpi, two workers on this host: the 2000-bit number at
#     B1 = 2000, 32 curves a batch, must be factored in batch 10 (seed
#     11) -- the one single-GPU seed of 1..12 that finds it. Skipped
#     without gmpecmfft-mpi or mpirun.
#  7. tools/gmpecmfft-slurm's array tasks, two run here by hand (as
#     Slurm would, without scancel): the same factor from the same
#     batch, the other task stopped through the stop file, and `status`
#     reporting it.
set -u
cd "$(dirname "$0")/.."
BIN=./gmpecmfft
NUMS=bench/fft-numbers.txt
fail=0
num() { awk -v b="$1" '$1==b {print $3}' "$NUMS"; }

echo "== 1. FFT arithmetic vs GMP =="
for bits in 200 4000 16384 20000 100000 1000000; do
  n=$(python3 -I -c "import random,sys; sys.set_int_max_str_digits(0); \
random.seed($bits); print(random.getrandbits($bits) | 1 | (1 << ($bits - 1)))")
  r=$(echo "$n" | $BIN -t 4 -c 16)
  echo "  $bits bits: $r"
  case "$r" in *" 0 wrong"*) ;; *) fail=1 ;; esac
done

echo "== 2. FFT vs CGBN, 2000 bits, 256 curves, B1 = 10^4, B2 = 10^5 =="
echo "   (per-prime stage 2 at this B2: the same pairs as CGBN)"
n=$(num 2000)
a=$(echo "$n" | $BIN -v -1 10000 -2 100000 -c 256 -s 1 2>&1 |
    grep -o '[0-9]* of 256' | tr '\n' ' ')
b=$(echo "$n" | $BIN -F -v -1 10000 -2 100000 -c 256 -s 1 2>&1 |
    grep -o '[0-9]* of 256' | tr '\n' ' ')
echo "  CGBN: stage 1, 2 hits: $a"
echo "  FFT : stage 1, 2 hits: $b"
[ -n "$a" ] && [ "$a" = "$b" ] || fail=1

echo "== 3. polynomial stage 2 vs host replay, 2000 bits =="
for spec in "D=2310: -1 10000 -2 1000000" "D=210 (forced): -1 2000 -2 100000"; do
  args=${spec#*: }
  force=; case "$spec" in *forced*) force=poly ;; esac
  r=$(echo "$n" | GMPECMFFT_STAGE2=$force GMPECMFFT_S2_CHECK=1 \
      $BIN -F -v $args -c 32 -s 5 2>&1)
  ok=$(echo "$r" | grep -c 'host check.*identical')
  bad=$(echo "$r" | grep -c 'host check.*MISMATCH')
  echo "  ${spec%%:*}: $ok identical, $bad mismatched"
  echo "$r" | grep -q 'stage 2: polynomial' || { echo "  (not polynomial)"; fail=1; }
  [ "$ok" -ge 1 ] && [ "$bad" -eq 0 ] || fail=1
done

echo "== 4. polynomial stage 2 in curve groups, 2000 bits, 128 curves =="
a=$(echo "$n" | $BIN -F -v -1 10000 -2 1000000 -c 128 -s 5 2>&1 |
    grep -o '[0-9]* of 128' | tr '\n' ' ')
r=$(echo "$n" | GMPECMFFT_S2_GROUP=7 $BIN -F -v -1 10000 -2 1000000 \
    -c 128 -s 5 2>&1)
b=$(echo "$r" | grep -o '[0-9]* of 128' | tr '\n' ' ')
echo "  one group:     stage 1, 2 hits: $a"
echo "  groups of 7:   stage 1, 2 hits: $b"
echo "$r" | grep -q '19 groups' || { echo "  (not grouped)"; fail=1; }
[ -n "$a" ] && [ "$a" = "$b" ] || fail=1

if command -v gmp-ecm >/dev/null 2>&1; then
  echo "== 5. FFT vs GMP-ECM, 20000 bits, 256 curves, B1 = 5000, stage 1 =="
  n=$(num 20000)
  f=$(echo "$n" | $BIN -v -1 5000 -2 0 -c 256 -s 1 2>&1 |
      grep -o '[0-9]* of 256' | cut -d' ' -f1)
  # GMP-ECM exits with an odd code when it found no factor.
  g=$($BIN --sigmas 256 -s 1 | xargs -P "$(nproc)" -I{} sh -c \
      "echo $n | gmp-ecm -q -sigma 0:{} 5000 5000 >/dev/null 2>&1; echo \$?" |
      awk '$1 % 2 == 0 && $1 != 0 {c++} END {print c+0}')
  echo "  GMP-ECM: $g of 256;  FFT: $f of 256"
  [ -n "$f" ] && [ "$f" = "$g" ] || fail=1
else
  echo "== 5. skipped: gmp-ecm not installed =="
fi

MPIRUN=$(command -v mpirun || ls /usr/lib64/openmpi/bin/mpirun 2>/dev/null)
if [ -x ./gmpecmfft-mpi ] && [ -n "$MPIRUN" ]; then
  # Open MPI 5's mpirun finds prterun on PATH.
  PATH="$(dirname "$MPIRUN"):$PATH"
  echo "== 6. gmpecmfft-mpi, 2 workers, 2000 bits, B1 = 2000, 32 curves/batch =="
  n=$(num 2000)
  r=$(echo "$n" | "$MPIRUN" --oversubscribe -n 3 ./gmpecmfft-mpi -c 32 \
      -1 2000 -2 0 -v 2>&1)
  f=$(echo "$r" | grep -x '[0-9]*' | head -1)
  b=$(echo "$r" | grep -o 'factor found by rank [0-9]* in batch [0-9]*' |
      awk '{print $NF}')
  echo "  factor $f in batch $b (expected 170286825566383 in batch 10)"
  [ "$f" = 170286825566383 ] && [ "$b" = 10 ] || fail=1
else
  echo "== 6. skipped: no gmpecmfft-mpi or mpirun =="
fi

echo "== 7. gmpecmfft-slurm, 2 array tasks run locally, same bounds =="
SD=$(mktemp -d)
n=$(num 2000)
echo "$n" | ./tools/gmpecmfft-slurm submit -d "$SD/run" -n 2 -c 32 -1 2000 \
  -2 0 --dry-run >/dev/null
for t in 0 1; do
  SLURM_ARRAY_TASK_ID=$t SLURM_ARRAY_TASK_COUNT=2 \
    ./tools/gmpecmfft-slurm task "$SD/run" >/dev/null 2>&1 &
done
wait
st=$(./tools/gmpecmfft-slurm status "$SD/run")
echo "$st" | sed 's/^/  /'
echo "$st" | grep -q '^factor 170286825566383 (task 0 batch 10 seed 11' || fail=1
rm -rf "$SD"

if [ $fail -eq 0 ]; then echo "== check-fft passed =="; else
  echo "== check-fft FAILED =="; exit 1; fi
