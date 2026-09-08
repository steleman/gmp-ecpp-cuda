#!/bin/bash

# sweep-threads -- how does the discriminant sweep scale?
#
# Speculative parallelism with a catch: the sweep keeps the *smallest*
# successful discriminant, because a larger |D| means a larger class
# number and a costlier proof step. So workers beyond the one that finds
# it are doing work that gets discarded, and the ceiling is set by how
# many discriminants the input actually needs -- an input whose first
# discriminant works has nothing to parallelise.

set -u
BIN=${BIN:-./gmp-ecpp}
DATA=${DATA:-bench/primes.txt}
SWEEPS=${SWEEPS:-"1 2 4 8 16 32"}
SEED=${SEED:-1}
MIN=${MIN:-90}

printf '%6s' digits; for s in $SWEEPS; do printf '\tS=%s' "$s"; done; echo
declare -A tot
while read -r d n; do
  case "$d" in \#*|"") continue;; esac
  [ "$d" -lt "$MIN" ] && continue
  printf '%6s' "$d"
  for s in $SWEEPS; do
    t0=$(date +%s.%N)
    echo "$n" | timeout 900 "$BIN" -Q -s "$SEED" -S "$s" >/dev/null 2>&1; rc=$?
    t1=$(date +%s.%N); e=$(echo "$t1-$t0" | bc)
    [ "$rc" = 0 ] || e=FAIL
    printf '\t%.2f' "$e"
    tot[$s]=$(echo "${tot[$s]:-0}+$e" | bc)
  done
  echo
done < "$DATA"
echo; printf 'totals:'
base=""
for s in $SWEEPS; do
  [ -z "$base" ] && base=${tot[$s]}
  printf '\tS=%s %.1fs (%.2fx)' "$s" "${tot[$s]}" "$(echo "scale=3; $base/${tot[$s]}" | bc)"
done; echo
