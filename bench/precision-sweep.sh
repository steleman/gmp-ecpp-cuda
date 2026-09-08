#!/bin/bash

# precision-sweep -- compare working precisions across input sizes.
#
# Produces the "Precision is the dominant knob" table in README.md, and
# is the evidence behind the default having been lowered to 1000 bits.
# Every input must still prove at the lower precision; a nonzero exit
# status in either column is a failure, not a datapoint.

set -u
BIN=${BIN:-./gmp-ecpp}
DATA=${DATA:-bench/primes.txt}
LOW=${LOW:-1000}
HIGH=${HIGH:-10000}
SEED=${SEED:-1}

printf '%6s %11s %11s %9s  %s\n' digits "p=$LOW" "p=$HIGH" ratio status
fail=0
while read -r d n; do
  case "$d" in \#*|"") continue;; esac
  for p in "$LOW" "$HIGH"; do
    t0=$(date +%s.%N)
    echo "$n" | timeout 600 "$BIN" -Q -s "$SEED" -p "$p" >/dev/null 2>&1; rc=$?
    t1=$(date +%s.%N)
    eval "t_$p=\$(echo \"\$t1-\$t0\" | bc)"
    eval "rc_$p=\$rc"
  done
  eval "tl=\$t_$LOW; th=\$t_$HIGH; rl=\$rc_$LOW; rh=\$rc_$HIGH"
  st=ok
  if [ "$rl" != 0 ] || [ "$rh" != 0 ]; then st="FAIL rc=$rl/$rh"; fail=$((fail+1)); fi
  printf '%6s %10.2fs %10.2fs %8.1fx  %s\n' "$d" "$tl" "$th" \
    "$(echo "scale=3; $th/$tl" | bc)" "$st"
done < "$DATA"
echo
[ "$fail" = 0 ] && echo "all inputs proved at both precisions" \
                || { echo "$fail failures"; exit 1; }
