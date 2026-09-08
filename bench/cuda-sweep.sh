#!/bin/bash

# cuda-sweep -- whole-proof CPU vs CUDA across sizes and seeds.
#
# Produces the "full sweep" table in README.md.  The spread matters more
# than the mean: the backend's value is in removing the pathological
# cases, so watch the slowest run in each column rather than the average.

set -u
BIN=${BIN:-./gmp-ecpp}
DATA=${DATA:-bench/primes.txt}
SEEDS=${SEEDS:-"1 5"}
MIN=${MIN:-90}          # below this the GPU is gated off anyway

printf '%6s %5s %11s %11s %9s\n' digits seed CPU CUDA speedup
tc_all=0; tg_all=0; wins=0; runs=0
while read -r d n; do
  case "$d" in \#*|"") continue;; esac
  [ "$d" -lt "$MIN" ] && continue
  for s in $SEEDS; do
    t0=$(date +%s.%N); echo "$n" | timeout 900 "$BIN" -Q    -s "$s" >/dev/null 2>&1; a=$?; t1=$(date +%s.%N)
    tc=$(echo "$t1-$t0" | bc)
    t0=$(date +%s.%N); echo "$n" | timeout 900 "$BIN" -Q -C -s "$s" >/dev/null 2>&1; b=$?; t1=$(date +%s.%N)
    tg=$(echo "$t1-$t0" | bc)
    printf '%6s %5s %10.2fs %10.2fs %8.2fx%s\n' "$d" "$s" "$tc" "$tg" \
      "$(echo "scale=3; $tc/$tg" | bc)" \
      "$([ "$a" = "$b" ] || echo '  <-- verdicts differ')"
    tc_all=$(echo "$tc_all+$tc" | bc); tg_all=$(echo "$tg_all+$tg" | bc)
    runs=$((runs+1))
    [ "$(echo "$tc > $tg" | bc)" = 1 ] && wins=$((wins+1))
  done
done < "$DATA"
echo
printf 'aggregate: CPU %.1fs, CUDA %.1fs -> %.2fx over %d runs; CUDA faster in %d\n' \
  "$tc_all" "$tg_all" "$(echo "scale=3; $tc_all/$tg_all" | bc)" "$runs" "$wins"
