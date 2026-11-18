#!/bin/bash
# gmpecpp-mpi: CPU-only ECM vs the CGBN backend (-C), on the 18 primes of
# 90 to 180 digits in bench/primes.txt (make bench-mpi-cgbn; ~40 min).
#
#   bench/mpi-cgbn.sh OUTDIR [SEEDS] [PARTS]
#       SEEDS default "1 2 3 4 5"; PARTS default "seeds sweep bound"
#
# Configurations (workers x ECM threads; rank 0 coordinates). Every rank
# is unbound (--bind-to none):
#   seeds  1x8, 4x8, 8x4   every seed, CPU and -C
#   sweep  1x16, 1x32      the first seed only, CPU and -C (thread count)
#   bound  1x8b            the first seed only, CPU and -C, but with Open
#                          MPI's default binding for a 2-process job --
#                          each rank bound to one core -- which puts all
#                          of a rank's ECM threads on that core. Shown
#                          because it is easy to fall into: mpirun binds
#                          by default unless oversubscribing.
# CPU and -C runs alternate, so slow drift (clocks, temperature) hits
# both alike. Raw output goes to OUTDIR/<config>-<mode>-s<seed>.txt and
# the summary to OUTDIR/summary.txt (and stdout).
#
# What varies between seeds: the prover reseeds at every proof, so a seed
# fixes each proof's random choices. With -T > 1 a run can still differ
# from a repeat of itself: the threaded ECM is a race (the first thread to
# find a factor wins), and contention from other ranks changes who wins.
set -u
cd "$(dirname "$0")/.."
OUT=${1:?usage: bench/mpi-cgbn.sh OUTDIR [SEEDS]}
SEEDS=${2:-"1 2 3 4 5"}
PARTS=${3:-"seeds sweep bound"}
want() { [[ " $PARTS " == *" $1 "* ]]; }
mkdir -p "$OUT"
MPIRUN=$(command -v mpirun || ls /usr/lib64/openmpi/bin/mpirun 2>/dev/null)
[ -n "$MPIRUN" ] || { echo "no mpirun" >&2; exit 1; }
PATH="$(dirname "$MPIRUN"):$PATH" # Open MPI 5's mpirun finds prterun on PATH
[ -x ./gmpecpp-mpi ] || { echo "build gmpecpp-mpi first (make)" >&2; exit 1; }
IN="$OUT/primes90.txt"
grep -v '^#' bench/primes.txt | awk '$1 >= 90' >"$IN"

run() { # workers threads seed mode [bound]
  local w=$1 t=$2 s=$3 m=$4 f="" tag="" bind="--bind-to none"
  [ "$m" = gpu ] && f="-C"
  [ "${5:-}" = bound ] && { tag=b; bind="--bind-to core"; }
  local o="$OUT/${w}x${t}${tag}-$m-s$s.txt"
  "$MPIRUN" --oversubscribe $bind -n $((w + 1)) ./gmpecpp-mpi -T "$t" -s "$s" $f \
    "$IN" >"$o" 2>&1
  echo "  ${w}x${t}${tag} $m seed $s: $(grep -o 'wall [0-9.]*s, summed proof time [0-9.]*s' "$o")"
}

first=${SEEDS%% *}
if want seeds; then
  for s in $SEEDS; do
    for cfg in "1 8" "4 8" "8 4"; do
      set -- $cfg
      run "$1" "$2" "$s" cpu
      run "$1" "$2" "$s" gpu
    done
  done
fi
if want sweep; then
  for t in 16 32; do
    run 1 "$t" "$first" cpu
    run 1 "$t" "$first" gpu
  done
fi
if want bound; then
  run 1 8 "$first" cpu bound
  run 1 8 "$first" gpu bound
fi

python3 - "$OUT" <<'EOF' | tee "$OUT/summary.txt"
import glob, re, statistics as st, sys
out = sys.argv[1]
runs = {}  # (cfg, mode) -> list of (seed, wall, summed, {idx: (digits, t)})
for f in glob.glob(f"{out}/*x*-*-s*.txt"):
    if not re.match(r".*/\d+x\d+b?-(cpu|gpu)-s\d+\.txt$", f):
        continue
    cfg, mode, seed = re.match(r".*/(\d+x\d+b?)-(cpu|gpu)-s(\d+)\.txt$", f).groups()
    t = open(f).read()
    m = re.search(r"wall ([0-9.]+)s, summed proof time ([0-9.]+)s", t)
    proofs = {int(a): (int(b), float(c)) for a, b, c in re.findall(
        r"\[\s*(\d+)\]\s+(\d+)\s+digits\s+proven prime\s+\d+ steps\s+([0-9.]+)s", t)}
    bad = len(re.findall(r"\[\s*\d+\]", t)) - len(proofs)
    runs.setdefault((cfg, mode), []).append(
        (int(seed), float(m.group(1)), float(m.group(2)), proofs, bad))

def stats(v):
    return f"{st.mean(v):7.1f} ({min(v):6.1f}-{max(v):6.1f})" if len(v) > 1 else f"{v[0]:7.1f}{'':16}"

print("Batch wall and summed proof time, seconds: mean (min-max) over seeds\n")
print(f"{'config':7} {'seeds':>5}  {'CPU wall':>24}  {'CGBN wall':>24}  {'ratio':>6}   "
      f"{'CPU summed':>24}  {'CGBN summed':>24}  {'ratio':>6}")
order = ["1x8", "1x16", "1x32", "4x8", "8x4", "1x8b"]
for cfg in order:
    c, g = runs.get((cfg, "cpu"), []), runs.get((cfg, "gpu"), [])
    if not c or not g:
        continue
    cw, gw = [r[1] for r in c], [r[1] for r in g]
    cs, gs = [r[2] for r in c], [r[2] for r in g]
    print(f"{cfg:7} {len(c):5}  {stats(cw)}  {stats(gw)}  {st.mean(cw)/st.mean(gw):5.2f}x   "
          f"{stats(cs)}  {stats(gs)}  {st.mean(cs)/st.mean(gs):5.2f}x")
bad = sum(r[4] for v in runs.values() for r in v)
print("\n1x8b: as 1x8, but each rank bound to one core (Open MPI's default")
print("for this job size) -- all 8 ECM threads of the worker on one core.")
print(f"\nproofs not completed as 'proven prime': {bad}")

for cfg in ("1x8",):
    c, g = runs.get((cfg, "cpu"), []), runs.get((cfg, "gpu"), [])
    if not c or not g:
        continue
    print(f"\nPer number, {cfg}: median (min-max) over {len(c)} seeds, seconds\n")
    print(f"{'#':>2} {'digits':>6}  {'CPU':>24}  {'CGBN':>24}  {'CPU/CGBN':>8}")
    tc = tg = 0.0
    for i in sorted(c[0][3]):
        a = [r[3][i][1] for r in c]; b = [r[3][i][1] for r in g]
        d = c[0][3][i][0]
        tc += st.median(a); tg += st.median(b)
        print(f"{i:2} {d:6}  {st.median(a):7.2f} ({min(a):6.2f}-{max(a):6.2f})  "
              f"{st.median(b):7.2f} ({min(b):6.2f}-{max(b):6.2f})  {st.median(a)/st.median(b):7.2f}x")
    print(f"{'sum of medians':>9}  {tc:7.1f}{'':17}  {tg:7.1f}{'':17}  {tc/tg:7.2f}x")
EOF
