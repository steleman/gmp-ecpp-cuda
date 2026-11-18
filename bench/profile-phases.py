#!/usr/bin/env python3
"""profile-phases -- where does an ECPP proof actually spend its time?

Injects an exclusive-time profiler into a throwaway copy of the prover,
builds it, and reports the split between the class-polynomial layer and
the factoring layer.  Produces the "Where the time goes" tables in
README.md and doc/gmpecpp.tex.

This matters more than it might look.  The split is not a property of the
program but of the -p setting: at 10000 bits the class-polynomial layer
dominates and CGBN cannot help it, while at the 1000-bit default the
factoring stage dominates and CGBN can.  Profile at the precision you
actually intend to run.

Usage:
    python3 bench/profile-phases.py [-p BITS] [-s SEED] [NUMBER ...]

With no numbers, uses one prime of each of 30, 60 and 100 digits from
bench/primes.txt.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import io

SOURCE = os.environ.get('GMPECPP_SOURCE', 'gmp-ecpp.cpp')
src = io.open(SOURCE, encoding='utf-8').read()

prof = r'''
// ---- injected exclusive-time profiler ----
#include <time.h>
namespace prof {
enum R { CORNACCHIA, CHECKFACTOR, HILBERT, WEBER, ROOTS, CURVEMUL, SQRTMOD, NREG };
static const char* NAMES[NREG] = {"modified_Cornacchia","check_for_factor(ECM)",
  "Hilbert","Weber","FindRootsModuloAPrime","curve point multiply","square_root_mod"};
static double acc[NREG];
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                     return t.tv_sec + 1e-9*t.tv_nsec; }
static double t_start;
struct Scope {
  int r; double t0; Scope* parent; static Scope* cur;
  Scope(int r_):r(r_),t0(now()),parent(cur){ if(parent) acc[parent->r]+=t0-parent->t0; cur=this; }
  ~Scope(){ double t=now(); acc[r]+=t-t0; cur=parent; if(parent) parent->t0=t; }
};
Scope* Scope::cur = 0;
struct Init { Init(){ t_start = now(); } } _init;
__attribute__((destructor)) static void dump(){
  double total = now() - t_start, sum = 0;
  fprintf(stderr, "\n=== phase profile (exclusive) ===\n");
  for (int i=0;i<NREG;i++) sum += acc[i];
  for (int i=0;i<NREG;i++)
    fprintf(stderr, "%-24s %8.3fs  %5.1f%%\n", NAMES[i], acc[i], 100.0*acc[i]/total);
  fprintf(stderr, "%-24s %8.3fs  %5.1f%%\n", "(everything else)", total-sum, 100.0*(total-sum)/total);
  fprintf(stderr, "%-24s %8.3fs\n", "TOTAL", total);
}
}
// ---- end profiler ----
'''
src = src.replace('#define HAVE_GETTIMEOFDAY 1', '#define HAVE_GETTIMEOFDAY 1\n' + prof, 1)

def inject(pattern, region):
    global src
    m = re.search(pattern, src)
    if not m:
        sys.stderr.write("MISS: %s\n" % pattern); return
    end = src.index('{', m.end()-1)
    src = src[:end+1] + ('\n  prof::Scope _ps(prof::%s);' % region) + src[end+1:]

inject(r'int32_t modified_Cornacchia\(mpz_class D[^{]*', 'CORNACCHIA')
inject(r'bool check_for_factor\(mpz_class \*q[^{]*', 'CHECKFACTOR')
inject(r'void Hilbert\(int64_t D[^{]*', 'HILBERT')
inject(r'void Weber\(int64_t D[^{]*', 'WEBER')
inject(r'void FindRootsModuloAPrime\(int64_t degreeP[^{]*', 'ROOTS')
inject(r'mpz_class square_root_mod\(mpz_class a[^{]*', 'SQRTMOD')



# ---------------------------------------------------------------- #
# Driver                                                            #
# ---------------------------------------------------------------- #

def build_and_run(numbers, precision, seed):
    tmp = tempfile.mkdtemp(prefix='gmpecpp-prof-')
    cpp = os.path.join(tmp, 'prof.cpp')
    exe = os.path.join(tmp, 'prof')
    io.open(cpp, 'w', encoding='utf-8').write(src)

    cc = os.environ.get('CXX', 'g++')
    subprocess.check_call([cc, '-O2', '-w', '-o', exe, cpp, '-lgmpxx', '-lgmp'])

    for n in numbers:
        print('---------- %d digits, -p %d ----------' % (len(n), precision))
        p = subprocess.run([exe, '-Q', '-s', str(seed), '-p', str(precision)],
                           input=n + '\n', capture_output=True, text=True)
        # the profiler writes its report to stderr on exit
        for line in p.stderr.splitlines():
            if line.strip() and not line.startswith('==='):
                print('  ' + line)
        print()


def default_numbers():
    want, out = [30, 60, 100], []
    try:
        for line in io.open('bench/primes.txt', encoding='utf-8'):
            if line.startswith('#') or not line.strip():
                continue
            d, v = line.split()
            if int(d) in want:
                want.remove(int(d))
                out.append(v)
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-p', '--precision', type=int, default=1000)
    ap.add_argument('-s', '--seed', type=int, default=1)
    ap.add_argument('numbers', nargs='*')
    a = ap.parse_args()
    nums = a.numbers or default_numbers()
    if not nums:
        sys.exit('no numbers given and bench/primes.txt not readable')
    build_and_run(nums, a.precision, a.seed)


if __name__ == '__main__':
    main()
