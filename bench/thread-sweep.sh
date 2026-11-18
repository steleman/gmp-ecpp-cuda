#!/bin/bash
# thread-sweep -- how does the ECM stage scale with POSIX threads?
#
# ECM is a Las Vegas algorithm: curves are independent and the search
# stops at the first success, so parallel speedup can exceed the thread
# count -- running twenty curves at once finds a factor in the time of
# the luckiest, not the average.  Expect scatter, and read the aggregate.
set -u
BIN=${BIN:-./gmp-ecpp}
DATA=${DATA:-bench/primes.txt}
THREADS=${THREADS:-"1 2 4 8 16 32"}
SEED=${SEED:-1}
MIN=${MIN:-90}

hdr="digits"; for t in $THREADS; do hdr="$hdr|T=$t"; done
echo "$hdr" | tr '|' '\t'
declare -A tot
while read -r d n; do
  case "$d" in \#*|"") continue;; esac
  [ "$d" -lt "$MIN" ] && continue
  line="$d"
  for t in $THREADS; do
    t0=$(date +%s.%N)
    echo "$n" | timeout 900 "$BIN" -Q -s "$SEED" -T "$t" >/dev/null 2>&1; rc=$?
    t1=$(date +%s.%N)
    e=$(echo "$t1-$t0" | bc)
    [ "$rc" = 0 ] || e="FAIL"
    line="$line	$e"
    tot[$t]=$(echo "${tot[$t]:-0}+$e" | bc 2>/dev/null || echo "${tot[$t]:-0}")
  done
  echo "$line"
done < "$DATA"
echo
echo -n "totals:"
base=""
for t in $THREADS; do
  [ -z "$base" ] && base=${tot[$t]}
  printf '\tT=%s %.1fs (%.2fx)' "$t" "${tot[$t]}" "$(echo "scale=3; $base/${tot[$t]}" | bc)"
done
echo
