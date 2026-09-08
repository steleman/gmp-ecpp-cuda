#!/bin/bash

# verify-both -- prove with each backend and check both certificates.
#
# This is the correct correctness test above the CUDA size gate, where
# the two backends legitimately produce different certificates.  Watch
# the step counts: a discriminant the host abandons is one the GPU can
# use, so GPU proofs are often shorter as well as faster.

set -u
VERIFY=${VERIFY:-./gmpecpp-verify}
DATA=${DATA:-bench/primes.txt}
MIN=${MIN:-90}

fail=0
while read -r d n; do
  case "$d" in \#*|"") continue;; esac
  [ "$d" -lt "$MIN" ] && continue
  echo "---------- $d digits ----------"
  "$VERIFY"    "$n"; a=$?
  "$VERIFY" -C "$n"; b=$?
  if [ "$a" != 0 ] || [ "$b" != 0 ]; then
    echo "  *** VERIFICATION FAILED (cpu=$a cuda=$b) ***"; fail=$((fail+1))
  fi
done < "$DATA"
echo
[ "$fail" = 0 ] && echo "all certificates verified on both backends" \
                || { echo "$fail failures"; exit 1; }
