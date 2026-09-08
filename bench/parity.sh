#!/bin/bash

# parity -- is the library-backed tool still the original program?
#
# Compares stdout, stderr and exit status against a reference build of
# the historical single-file program across a matrix of flags, inputs and
# seeds.  Any difference is a regression: below the CUDA size gate the
# GPU is never reached, so even -C must not perturb anything.
#
# Above the gate byte comparison is the WRONG test -- the GPU finds
# different factorisations and so different, equally valid certificates.
# Use gmpecpp-verify for that regime.

set -u
REF=${REF:-/tmp/gmp-ecpp-reference}
SRC=${SRC:-gmp-ecpp.cpp}
BIN=${BIN:-./gmp-ecpp}
EXTRA=${EXTRA:-}          # e.g. EXTRA=-C to check the backend is inert

if [ ! -x "$REF" ]; then
  echo "building reference from $SRC"
  g++ -O2 -w -o "$REF" "$SRC" -lgmpxx -lgmp || exit 1
fi

# DEEP=1 additionally replays every prime in bench/primes.txt. The small
# matrix below misses regressions that only appear once the downrun runs
# many steps -- a stale flag between steps stayed hidden until a
# 140-digit input exposed it.
if [ "${DEEP:-0}" = 1 ]; then
  n=0; fail=0
  while read -r d v; do
    case "$d" in \#*|"") continue;; esac
    a=$(echo "$v" | "$REF" -Q -s 1 2>&1; echo "rc=$?")
    b=$(echo "$v" | "$BIN" $EXTRA -Q -s 1 2>&1; echo "rc=$?")
    n=$((n+1))
    if [ "$a" != "$b" ]; then
      fail=$((fail+1)); echo "MISMATCH ${d}-digit ${v:0:20}..."
    fi
  done < "${DATA:-bench/primes.txt}"
  echo "=== deep: $n primes, $fail mismatches ==="
  [ "$fail" = 0 ] || exit 1
  exit 0
fi

FLAGS=("-Q" "-q" "" "-Q -H" "-Q -W" "-Q -o" "-Q -p 2000" "-q -p 1500"
       "-Q -D 40" "-Q -B 5000" "-v -p 300" "-Q -o -H" "-Q -o -W")
NUMS=(1000003 4294967311 1000000000000000000000000000057
      1000000000000000012000000000000000027)
SEEDS=${SEEDS:-"1 2 7"}

n=0; fail=0
for s in $SEEDS; do
  for f in "${FLAGS[@]}"; do
    read -r -a F <<< "$f"
    for N in "${NUMS[@]}"; do
      a=$(echo "$N" | "$REF" "${F[@]}" -s "$s" 2>&1; echo "rc=$?")
      b=$(echo "$N" | "$BIN" $EXTRA "${F[@]}" -s "$s" 2>&1; echo "rc=$?")
      n=$((n+1))
      if [ "$a" != "$b" ]; then
        fail=$((fail+1))
        echo "MISMATCH flags='$f' seed=$s N=${N:0:14}"
        diff <(echo "$a") <(echo "$b") | cut -c1-100 | head -6
      fi
    done
  done
done
echo "=== $n runs, $fail mismatches ==="
[ "$fail" = 0 ] || exit 1
