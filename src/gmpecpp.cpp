// Author:  Pate Williams (c) 1997 & James Wanless (c) 2008-19 & Chris
// Studholme (c) 2004 & Elisavet Konstantinou & Yiannis Stamatiu &
// Christos Zaroliagis (c) 2003 & Scott Contini (c) 1996 & Paul
// Zimmermann (c) 2005 & Timo Poikola (c) 2010
//
// Multiple precision Atkin primality test.
// See "A Course in Computational Algebraic Number Theory" by Henri
// Cohen Algorithm 9.2.4 page 474.
//
// This file is part of the GMP-ECPP Math C++ Program.
//
// The GMP-ECPP program is free software; you can redistribute it
// and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3 of
// the License, or (at your option) any later version.
//
// The GMP-ECPP program is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// aint64_t with this program.  If not, see
// <http://www.gnu.org/licenses/>.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include <getopt.h>
#include <sys/time.h>

#include <gmp.h>
#include <gmpxx.h>

#define SIEVE_LIMIT 1073741824

#define LOG2 0.301029995f

#define POLY_SIZE 8192

#define PRECISION 1000
#define ERROR_SHIFT 1000

#define BMAX 2000
#define DMAX 20

#define HAVE_GETTIMEOFDAY 1

// ===================================================================
// libgmpecpp: library scaffolding.
//
// The prover below is the original program with main() removed and all
// direct output redirected.  A library must not write to stdout, must
// not call exit(), and must be able to hand its results back as data
// rather than as printed text -- these three facts drive every change.
// ===================================================================

#include <pthread.h>

#include <atomic>

#include <sstream>
#include <string>
#include <vector>

#include "gmpecpp.h"

namespace gmpecpp_detail {

// Thrown where the original called exit(2): a non-invertible element mod
// N is a proof of compositeness, not an error, but a library has to
// unwind rather than terminate the host process.
struct CompositeFound {
  std::string factor;
  explicit CompositeFound(const std::string &f) : factor(f) {}
};

// Why Atkin() returned 2.  The original conflated these.
enum GiveUp { GU_NONE = 0, GU_COMPOSITE, GU_DISCRIMINANTS, GU_BOUNDS };

struct Recorded {
  std::string n, a, b, m, q, px, py, p1x, p1y, p2x, p2y;
  long d;
};

struct State {
  gmpecpp_log_fn log_fn;
  void *log_user;
  int verbosity;
  std::string pending;             // partial line awaiting its newline
  std::vector<Recorded> steps;
  std::string factor;
  GiveUp give_up;
  double ecm_seconds;
};

State &state();

// Accumulates like stdout did: emits to the callback one whole line at a
// time, so the original's mix of streamed fragments still comes out as
// coherent lines.
struct LogSink {
  // Several sweep workers may log at once; serialise so lines interleave
  // rather than corrupt each other.
  static pthread_mutex_t &lock() {
    static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    return m;
  }
  void put(const std::string &s) {
    State &st = state();
    if (!st.log_fn)
      return;
    pthread_mutex_lock(&lock());
    st.pending += s;
    size_t nl;
    while ((nl = st.pending.find('\n')) != std::string::npos) {
      std::string line = st.pending.substr(0, nl);
      st.pending.erase(0, nl + 1);
      st.log_fn(line.c_str(), st.log_user);
    }
    pthread_mutex_unlock(&lock());
  }
  template <class T> LogSink &operator<<(const T &v) {
    if (state().log_fn) {
      std::ostringstream os;
      os << v;
      put(os.str());
    }
    return *this;
  }
};

LogSink &log();

double wall_seconds();

// Selected ECM backend and its parameters for the proof in progress.
extern gmpecpp_backend_t g_backend;
extern unsigned int g_cuda_curves;
extern unsigned int g_cuda_min_bits;
extern int g_threads;
extern int g_sweep_threads;
extern unsigned long g_seed;

// Implemented in ecm_cuda.cu, or stubbed out in ecm_stub.cpp.
bool cuda_available();
const char *cuda_device_name();
bool cuda_ecm_factor(mpz_srcptr n, unsigned long b1, unsigned int curves,
                     unsigned long seed, mpz_ptr factor);
int cuda_ecm_run(mpz_srcptr n, unsigned long b1, unsigned long b2,
                 unsigned int curves, unsigned long seed, int (*stop)(void *),
                 void *stop_arg, mpz_ptr factor);
unsigned int cuda_ecm_default_curves(size_t bits);

} // namespace gmpecpp_detail


gmp_randstate_t rstate;

int32_t quiet = 0;
int32_t Quiet = 0;
int32_t verbose = 0;
int32_t one = 0;
int32_t weber = 0;
int32_t hilbert = 0;
int64_t seed = 0;
int64_t precision = 0;
int64_t error_shift = 0;
int64_t Bmax = 0;
int64_t Dmax = 0;

bool staticBmax = false;
bool staticDmax = false;

struct complex {
  mpf_class x, y;
};

struct point {
  mpz_class x, y;
};

mpf_class PIf;
mpf_class Ef;
mpf_class NATLOGONEPOINTNINEf;

mpf_class HALF;
mpf_class ONEPOINTNINE;

mpf_class ZERO;
mpf_class ONE;
mpf_class TWO;
mpf_class THREE;
mpf_class FOUR;
mpf_class FIVE;
mpf_class SIX;
mpf_class EIGHT;
mpf_class TEN;
mpf_class TWELVE;
mpf_class EIGHTEEN;
mpf_class NINETEEN;
mpf_class TWENTYFOUR;
mpf_class FIFTYSEVEN;
mpf_class TWOHUNDREDTHIRTYNINE;
mpf_class TWOHUNDREDFIFTYSIX;

// returns GMP pseudo-random number of 32-bits
/*
 * The prover's random source.
 *
 * rand2() is called from deep inside LenstrasECM and the point search,
 * far from anywhere a random state could be threaded through by hand.
 * When a worker thread installs its own state here, every call beneath
 * it picks that up; otherwise the file-scope rstate is used and
 * behaviour is exactly what it always was.
 *
 * Without this, sweep workers race on the shared rstate -- which is both
 * a data race and the reason a first attempt at a parallel sweep was
 * not reproducible.
 */
thread_local __gmp_randstate_struct *tls_rand = 0;

mpz_class rand2() {
  mpz_t temp;
  mpz_init(temp);
  mpz_urandomb(temp, tls_rand ? tls_rand : rstate, 32);
  mpz_class temp_class(temp);
  mpz_clear(temp);
  return temp_class;
}

// Reentrant form.  gmp_randstate_t is the one piece of GMP state the
// prover shares and mutates, so every worker thread carries its own.
mpz_class rand2_r(gmp_randstate_t st) {
  mpz_t temp;
  mpz_init(temp);
  mpz_urandomb(temp, st, 32);
  mpz_class temp_class(temp);
  mpz_clear(temp);
  return temp_class;
}

// returns greatest common divisor of a and b
mpz_class gcd(mpz_class a, mpz_class b) {
  mpz_t temp;
  mpz_init(temp);
  mpz_gcd(temp, a.get_mpz_t(), b.get_mpz_t());
  mpz_class temp_class(temp);
  mpz_clear(temp);
  return temp_class;
}

// returns inverse of a modulo b or 0 if it does not exist
mpz_class inverse(mpz_class a, mpz_class b) {
  mpz_t temp;
  mpz_init(temp);
  if (!mpz_invert(temp, a.get_mpz_t(), b.get_mpz_t()))
    mpz_set_si(temp, 0);
  mpz_class temp_class(temp);
  mpz_clear(temp);
  return temp_class;
}

// returns a modulo b, strictly non-negative
mpz_class modpos(mpz_class a, mpz_class b) {
  mpz_class temp_class;
  temp_class = a % b;
  if (temp_class < 0)
    temp_class += b;
  return temp_class;
}

// returns x ^ b mod n
mpz_class exp_mod(mpz_class x, mpz_class b, mpz_class n) {
  mpz_t temp;
  mpz_init(temp);
  mpz_powm(temp, x.get_mpz_t(), b.get_mpz_t(), n.get_mpz_t());
  mpz_class temp_class(temp);
  mpz_clear(temp);
  return temp_class;
}

// returns a ^ e
mpz_class exp_pow(mpz_class a, int64_t e) {
  mpz_t temp;
  mpz_init(temp);
  mpz_pow_ui(temp, a.get_mpz_t(), e);
  mpz_class temp_class(temp);
  mpz_clear(temp);
  return temp_class;
}

int32_t JACOBI(mpz_class a, mpz_class b) {
  return mpz_jacobi(a.get_mpz_t(), b.get_mpz_t());
}

// given an integer n >= 3 returns 0 composite, 1 probably prime
int32_t Rabin_Miller(mpz_class n) {
  return mpz_probab_prime_p(n.get_mpz_t(), 10);
}

// Given a positive integer n, this algorithm
// determines whether n is a square or not, and if it is
// outputs the square root of n.
int32_t square_test(mpz_class n, mpz_class *root) {
  int32_t issquare = 0;

  *root = sqrt(n);
  if ((*root * *root) == n)
    issquare = 1;
  return issquare;
}

// Given a positive integer n, this algorithm
// determines whether n is a cube or not, and if it is
// outputs the cube root of n.
int32_t cube_test(mpz_class n, mpz_class *root) {
  int32_t iscube = 0;

  mpz_t temp;
  mpz_init(temp);
  iscube = mpz_root(temp, n.get_mpz_t(), 3l);
  mpz_class temp_class(temp);
  mpz_clear(temp);

  *root = temp_class;

  return iscube;
}

// Given an integer n, this algorithm
// determines whether n is a power of 2 (or minus).
bool poweroftwo_test(mpz_class n) {
  bool ispoweroftwo = false;
  mpz_class temp_class;

  temp_class = n;
  while (temp_class % 2 == 0)
    temp_class /= 2;

  if (temp_class == 1 || temp_class == -1)
    ispoweroftwo = true;

  return ispoweroftwo;
}

// returns next prime after n
mpz_class nextp(mpz_class n) {
  mpz_t temp;
  mpz_init(temp);
  mpz_nextprime(temp, n.get_mpz_t());
  mpz_class temp_class(temp);
  mpz_clear(temp);
  return temp_class;
}

// affine coords
// returns 1 if P1 = -P2 therefore P1 + P2 = O, 0 otherwise
// P1 != P2
int32_t addition_1(mpz_class n, struct point P1, struct point P2,
                   struct point *P3) {
  mpz_class delta_x;
  mpz_class delta_y;
  mpz_class m;

  delta_x = modpos(P2.x - P1.x, n);
  delta_y = modpos(P2.y - P1.y, n);

  if (P1.x == P2.x && modpos(P1.y + P2.y, n) == 0) {
    P3->x = 0, P3->y = 1;
    return 1;
  }

  // calculate m = (y2 - y1)(x2 - x1) ^ -1 mod n
  m = modpos(delta_y * inverse(delta_x, n), n);

  // calculate x3 = m ^ 2 - (x1 + x2) mod n
  P3->x = modpos(m * m - (P1.x + P2.x), n);

  // calculate y3 = m(x1 - x3) - y1 mod n
  P3->y = modpos(m * (P1.x - P3->x) - P1.y, n);

  return 0;
}

// affine coords
// P1 == P2
void addition_2(mpz_class a, mpz_class n, struct point P1,
                struct point *P3) {
  mpz_class m;

  // calculate m = (3x1 ^ 2 + a)(2y1) ^ -1 mod n
  m = modpos((3 * P1.x * P1.x + a) * inverse(2 * P1.y, n), n);

  // calculate x3 = m ^ 2 - 2x1 mod n
  P3->x = modpos(m * m - 2 * P1.x, n);

  // calculate y3 = m(x1 - x3) - y1 mod n
  P3->y = modpos(m * (P1.x - P3->x) - P1.y, n);
}

// binary ladder
// returns -1 if O encountered, 0 if divisor not found, 1 otherwise
int32_t multiply(mpz_class a, mpz_class k, mpz_class n, struct point P,
                 struct point *R, mpz_class *d) {
  int32_t value = 1;
  struct point A, B, C;

  // A = P
  A = P;
  // B = O = (0, 1) the point at infinity
  B.x = 0, B.y = 1;

  while (value && k > 0) {
    if (k % 2 != 0) {
      *d = gcd(modpos(B.x - A.x, n), n);

      k--;
      value = (*d == 1 || *d == n);

      if (A.x == 0 && A.y == 1)
        ;
      else if (B.x == 0 && B.y == 1)
        B = A;
      else if (value) {
        addition_1(n, A, B, &C);
        B = C;
      }
    } else {
      *d = gcd(modpos(2 * A.y, n), n);

      k >>= 1;
      value = (*d == 1 || *d == n);

      if (value) {
        addition_2(a, n, A, &C);
        A = C;
      }
    }
  }

  *R = B;
  R->x = modpos(R->x, n);
  R->y = modpos(R->y, n);

  if (R->x == 0 && R->y == 1)
    return -1;

  return !value;
}

// Algorithm 1.5.1 (Square Root Mod p). See "A Course
// in Computational Algebraic Number Theory" by Henri
// Cohen page 33. This algorithm either outputs an
// x such that x^2 == a mod p, or says that such a
// solution does not exist.
mpz_class square_root_mod1(mpz_class a, mpz_class p) {
  mpz_class b, e = 0, m, n, q = p - 1, r, t, x, y, z;
  mpz_class TWO = 2;
  mpz_class A;

  A = modpos(a, p);

  // p - 1 = 2 ^ e * q with q odd
  while (q % 2 == 0)
    q >>= 1, e++;

  // find generator
  do
    do
      n = modpos(rand2(), p);
    while (n == 0);
  while (JACOBI(n, p) != -1);

  z = exp_mod(n, q, p);
  y = z, r = e;
  x = exp_mod(a, (q - 1) / 2, p);

  b = modpos((A * modpos(x, p)) * x, p);

  x = modpos((A * x), p);

  do {
    if (b == 1)
      return x;

    m = 1;
    while (exp_mod(b, TWO ^ m, p) != 1)
      m++;
    if (m == r)
      return 0;

    t = exp_mod(y, TWO ^ (r - m - 1), p);
    y = modpos(t * t, p);

    r = m;
    x = modpos(x * t, p);

    b = modpos(b * y, p);

  } while (true);
}

// Function: sqrtmod
//
// Description: Calculates the squareroot of a mod prime p.
// Uses the RESSOL algorithm of Shanks.  For description of algorithm,
// see:  An Introduction to the Theory of Numbers, 5th edition.
// Authors: Niven, Zuckerman, and Montgomery.  Pages 110-114.
//
// Always return the smallest squareroot
// *****************************************************************
// computes x so that x^2 == a mod p for prime p, and puts x in s.
// if no such x exists or if p is not prime, then sets s=0.
// ****************************************************************
void mpz_sqrtmod(mpz_t s, mpz_t a, mpz_t p) {

  mpz_t q, nn, x, y, t, b, z;
  int64_t n, i, m, r;

  mpz_init(q);
  mpz_init(y);

  if (mpz_fdiv_ui(p, 4) == 3) {
    mpz_add_ui(q, p, 1);
    mpz_div_2exp(q, q, 2);
    mpz_powm(s, a, q, p);
    mpz_mul(y, s, s);
    mpz_mod(y, y, p);
    if (mpz_cmp(y, a))
      mpz_set_ui(s, 0);
    mpz_clear(q);
    mpz_clear(y);
    return;
  }

  mpz_init(nn);
  mpz_init(x);
  mpz_init(t);
  mpz_init(b);
  mpz_init(z);

  mpz_sub_ui(q, p, 1);
  if (mpz_cmp_ui(q, 0) == 0)
    r = -1;
  else {
    r = mpz_scan1(q, 0);
    mpz_div_2exp(q, q, r);
  }
  n = 3;
  if ((mpz_fdiv_ui(p, 4) & n) == 3)
    m = -mpz_kronecker_si(p, n);
  else
    m = mpz_kronecker_si(p, n);

  while (m != -1) {
  loop:
    n += 2;
    if (!(n % 3))
      n += 2;
    for (i = 5; i * i <= n;) {
      if (!(n % i))
        goto loop;
      i += 2;
      if (!(n % i))
        goto loop;
      i += 4;
    }
    if ((mpz_fdiv_ui(p, 4) & n) == 3)
      m = -mpz_kronecker_si(p, n);
    else
      m = mpz_kronecker_si(p, n);
  }

  mpz_set_si(nn, n);

  mpz_powm(z, nn, q, p);

  mpz_set(y, z);
  mpz_sub_ui(t, q, 1);
  mpz_div_2exp(t, t, 1); // t = (q-1)/2
  mpz_powm(x, a, t, p);
  mpz_mul(t, x, x);
  mpz_mod(t, t, p);
  mpz_mul(b, t, a);
  mpz_mod(b, b, p); // b = a*x^2 mod p
  mpz_mul(x, a, x);
  mpz_mod(x, x, p);

  while (1) {
    if (mpz_cmp_ui(b, 1) == 0) {
      mpz_set(s, x);
      goto end;
    }

    mpz_mul(t, b, b);
    mpz_mod(t, t, p);
    m = 1;
    while (mpz_cmp_ui(t, 1) != 0) {
      if (m == r) {
        mpz_set_ui(s, 0);
        goto end;
      }
      mpz_mul(t, t, t);
      mpz_mod(t, t, p);
      m++;
    }

    mpz_set(t, y);
    if (r <= m) {
      mpz_set_ui(s, 0);
      goto end;
    }
    for (i = r - m - 1; i; i--) {
      mpz_mul(t, t, t);
      mpz_mod(t, t, p);
    }
    mpz_mul(y, t, t);
    mpz_mod(y, y, p);
    r = m;
    mpz_mul(x, x, t);
    mpz_mod(x, x, p);
    mpz_mul(b, b, y);
    mpz_mod(b, b, p);
  }
end:
  mpz_clear(q);
  mpz_clear(nn);
  mpz_clear(x);
  mpz_clear(y);
  mpz_clear(t);
  mpz_clear(b);
  mpz_clear(z);
}

// returns square root of a modulo p in the easy cases, 0 otherwise
mpz_class square_root_mod3(mpz_class a, mpz_class p) {
  if (modpos(p, 4) == 3)
    return exp_mod(a, (p + 1) / 4, p);

  else if (modpos(p, 8) == 5) {
    if (modpos(exp_mod(a, (p - 1) / 4, p), p) == 1)
      return exp_mod(a, (p + 3) / 8, p);
    else
      return modpos(2 * a * exp_mod(4 * a, (p - 5) / 8, p), p);
  }

  else
    return 0;
}

// returns square root of a modulo p if it exists, 0 otherwise
// version 3
mpz_class square_root_mod(mpz_class a, mpz_class p) {
  mpz_class temp_class3;
  if ((temp_class3 = square_root_mod3(a, p)) > 0)
    return temp_class3;

  mpz_class temp_class1;
  if ((temp_class1 = square_root_mod1(a, p)) > 0)
    return temp_class1;

  mpz_t temp;
  mpz_init(temp);
  mpz_sqrtmod(temp, a.get_mpz_t(), p.get_mpz_t());
  mpz_class temp_class2(temp);
  mpz_clear(temp);
  return temp_class2;
}

int64_t max(int64_t x, int64_t y) {
  if (x > y)
    return x;
  else
    return y;
}

void print_mpf(mpf_class x) {
  mpf_set_default_prec(precision);

  char *s = NULL;
  if (gmp_asprintf(&s, "%.*Ff", (int)(LOG2 * (x.get_prec())),
                   x.get_mpf_t()) >= 0 && s) {
    gmpecpp_detail::log() << s;
    free(s);
  }

  return;
}

mpf_class sinf2(mpf_class x) {
  mpz_class i, fact_2n_1 = 1;
  mpf_class x_2n_1(x, precision), sin_x(0, precision), old_sin_x(1, precision);

  bool sign = true;

  if (x < 0)
    return -sinf2(-x);

  if (x > 2 * PIf)
    return sinf2(x - 2 * PIf * floor(x / (2 * PIf)));

  if (x > PIf)
    return -sinf2(2 * PIf - x);

  if (x > PIf / 2)
    return sinf2(PIf - x);

  for (i = 1; old_sin_x != sin_x; i += 2) {
    old_sin_x = sin_x;

    if (sign)
      sin_x += x_2n_1 / fact_2n_1;
    else
      sin_x -= x_2n_1 / fact_2n_1;
    x_2n_1 *= x * x;
    fact_2n_1 *= (i + 1) * (i + 2);
    sign = !sign;
  }

  return sin_x;
}

mpf_class cosf2(mpf_class x) {
  mpz_class i, fact_2n = 1;
  mpf_class x_2n(1, precision), cos_x(0, precision), old_cos_x(1, precision);

  bool sign = true;

  if (x < 0)
    return cosf2(-x);

  if (x > 2 * PIf)
    return cosf2(x - 2 * PIf * floor(x / (2 * PIf)));

  if (x > PIf)
    return cosf2(2 * PIf - x);

  if (x > PIf / 2)
    return -cosf2(PIf - x);

  for (i = 0; old_cos_x != cos_x; i += 2) {
    old_cos_x = cos_x;

    if (sign)
      cos_x += x_2n / fact_2n;
    else
      cos_x -= x_2n / fact_2n;
    x_2n *= x * x;
    fact_2n *= (i + 1) * (i + 2);
    sign = !sign;
  }

  return cos_x;
}

mpf_class atanf2(mpf_class x) {
  mpz_class i, twon_1 = 1;
  mpf_class x_2n_1(x, precision), atan_x(0, precision),
      old_atan_x(1, precision);

  bool sign = true;

  if (x == 1)
    return (PIf / FOUR);

  if (x == -1)
    return (-PIf / FOUR);

  if (x > 1)
    return (PIf / TWO - atanf2(ONE / x));

  if (x < -1)
    return (-PIf / TWO - atanf2(ONE / x));

  for (i = 1; old_atan_x != atan_x; i += 2) {

    if (i > 1000000)
      break;

    old_atan_x = atan_x;

    if (sign)
      atan_x += x_2n_1 / twon_1;
    else
      atan_x -= x_2n_1 / twon_1;
    x_2n_1 *= x * x;
    twon_1 += 2;
    sign = !sign;
  }

  return atan_x;
}

mpf_class atan2f2(mpf_class x, mpf_class y) {

  mpf_class r(0, precision);

  if (x == 0 && y == 0)
    return 0;
  if (abs(y) >= abs(x)) {
    r = atanf2(x / y);
    if (y < 0) {
      if (x >= 0)
        return (r += PIf);
      return (r -= PIf);
    }
  } else {
    r = -atanf2(y / x);
    if (x < 0)
      return (r -= PIf / TWO);
    return (r += PIf / TWO);
  }

  return r;
}

mpf_class logf2(mpf_class x) {
  mpz_class i, n = 1;

  bool sign = true;

  if (x <= 0)
    return (ZERO);

  if (x == 1)
    return (ZERO);

  if (x > ONEPOINTNINE)
    return (logf2(x / ONEPOINTNINE) + NATLOGONEPOINTNINEf);

  if (x < 1)
    return (logf2(x * Ef) - ONE);

  x--;

  mpf_class x_n(x, precision), log_x(0, precision), old_log_x(1, precision);

  for (i = 1; old_log_x != log_x; i += 1) {
    old_log_x = log_x;

    if (sign)
      log_x += x_n / n;
    else
      log_x -= x_n / n;
    x_n *= x;
    n++;
    sign = !sign;
  }

  return log_x;
}

mpf_class expf2(mpf_class x) {
  mpz_class i, n = 1;
  mpf_class x_n(x, precision), exp_x(0, precision), old_exp_x(1, precision);

  if (x > 1)
    return (expf2(x - ONE) * Ef);

  if (x < 0)
    return (expf2(x + ONE) / Ef);

  for (i = 1; old_exp_x != exp_x; i += 1) {
    old_exp_x = exp_x;

    exp_x += x_n / n;
    x_n *= x;
    n *= (i + 1);
  }

  return (ONE + exp_x);
}

mpf_class sinhf2(mpf_class x) { return ((expf2(x) - expf2(-x)) / TWO); }

mpf_class coshf2(mpf_class x) { return ((expf2(x) + expf2(-x)) / TWO); }

mpf_class cabs2(struct complex u) { return sqrt(u.x * u.x + u.y * u.y); }

struct complex cadd(struct complex u, struct complex v) {
  struct complex w;

  w.x = u.x + v.x;
  w.y = u.y + v.y;
  return w;
}

struct complex csub(struct complex u, struct complex v) {
  struct complex w;

  w.x = u.x - v.x;
  w.y = u.y - v.y;
  return w;
}

struct complex cdiv(struct complex u, struct complex v) {
  mpf_class temp1, temp2;
  struct complex w;

  if (abs(v.x) <= abs(v.y)) {
    temp1 = v.x / v.y;
    temp2 = v.y + temp1 * v.x;
    w.x = (temp1 * u.x + u.y) / temp2;
    w.y = (temp1 * u.y - u.x) / temp2;
  } else {
    temp1 = v.y / v.x;
    temp2 = v.x + temp1 * v.y;
    w.x = (u.x + temp1 * u.y) / temp2;
    w.y = (u.y - temp1 * u.x) / temp2;
  }
  return w;
}

struct complex cmul(struct complex u, struct complex v) {
  struct complex w;

  w.x = u.x * v.x - u.y * v.y;
  w.y = u.x * v.y + u.y * v.x;
  return w;
}

struct complex cconj(struct complex u) {
  struct complex w;

  w.x = u.x;
  w.y = -u.y;
  return w;
}

struct complex csqrt2(struct complex u) {
  mpf_class r = sqrt(cabs2(u));
  mpf_class theta = atan2f2(u.y, u.x), phi = theta / TWO;
  struct complex w;

  w.x = r * cosf2(phi);
  w.y = r * sinf2(phi);

  return w;
}

struct complex cexp2(struct complex u) {
  mpf_class e = expf2(u.x);
  struct complex w;

  w.x = e * cosf2(u.y);
  w.y = e * sinf2(u.y);
  return w;
}

struct complex csin2(struct complex u) {
  struct complex w;

  w.x = sinf2(u.x) * coshf2(u.y);
  w.y = cosf2(u.x) * sinhf2(u.y);
  return w;
}

struct complex ccos2(struct complex u) {
  struct complex w;

  w.x = cosf2(u.x) * coshf2(u.y);
  w.y = -sinf2(u.x) * sinhf2(u.y);
  return w;
}

struct complex clog2(struct complex u) {
  mpf_class r = cabs2(u);
  struct complex w;

  w.x = logf2(r);
  w.y = atan2f2(u.y, u.x);
  return w;
}

struct complex cpow2(struct complex u, struct complex v) {
  return cexp2(cmul(v, clog2(u)));
}

mpz_class round2(mpf_class x) {
  mpz_class temp_class;
  if (x >= 0)
    temp_class = (mpz_class)trunc(x + HALF);
  else
    temp_class = (mpz_class)trunc(x - HALF);
  return temp_class;
}

void cpoly_mul(int64_t m, int64_t n, struct complex *a, struct complex *b,
               struct complex *c, int64_t *p) {
  int64_t i, j, k;
  struct complex ai, bj, sum, term;

  *p = m + n;
  for (k = 0; k <= *p; k++) {
    sum.x = sum.y = 0;
    for (i = 0; i <= k; i++) {
      j = k - i;
      if (i > m)
        ai.x = ai.y = 0;
      else
        ai = a[i];
      if (j > n)
        bj.x = bj.y = 0;
      else
        bj = b[j];
      term = cmul(ai, bj);
      sum = cadd(sum, term);
    }
    c[k] = sum;
  }
}

struct complex Sum(struct complex q) {
  mpf_class sign = -1;
  mpf_class error_term = 1;
  int64_t e1, e2, n = 1;
  struct complex expon1 = {ZERO, ZERO}, expon2 = {ZERO, ZERO};
  struct complex sum = {ZERO, ZERO}, term1, term2;

  for (int64_t i = 0; i < error_shift; i++)
    error_term /= 2;

  do {
    e1 = n * (3 * n - 1) / 2;
    e2 = n * (3 * n + 1) / 2;
    expon1.x = e1;
    expon2.x = e2;
    term1 = cpow2(q, expon1);
    term2 = cpow2(q, expon2);
    term1 = cadd(term1, term2);
    term1.x *= sign;
    term1.y *= sign;
    sum = cadd(sum, term1);
    sign = -sign;
    n++;
  } while (cabs2(term1) > error_term);

  sum.x += 1;

  return sum;
}

struct complex Delta(struct complex q) {
  struct complex s = {TWENTYFOUR, ZERO};
  struct complex Delta;

  Delta = cmul(q, cpow2(Sum(q), s));

  return Delta;
}

struct complex Theta(int64_t D, int64_t A, int64_t B) {
  struct complex s, Theta;

  s.x = -sqrt(mpf_class(D)) * PIf / A;
  s.y = -B * PIf / A;

  Theta = cexp2(s);

  return Theta;
}

struct complex j(struct complex tau) {
  struct complex c = {ZERO, TWO * PIf}, f, f1, q, q2,
                 twofiftysix = {TWOHUNDREDFIFTYSIX, ZERO}, one = {ONE, ZERO},
                 three = {THREE, ZERO};

  q = cexp2(cmul(c, tau));
  q2 = cmul(q, q);
  f = cdiv(Delta(q2), Delta(q));

  f1 = cmul(twofiftysix, f);
  f1 = cadd(one, f1);
  f1 = cpow2(f1, three);

  return cdiv(f1, f);
}

// Weber functions as listed "Explicit Construction of the Hilbert
// Class Fields of Imaginary Quadratic Fields by Integer Lattice
// Reduction" by Kaltofen-Yui pp. 180-1 and "Elliptic Curves and
// Primality Proving" by Atkin-Morain p. 7
struct complex F(int64_t j, int64_t D, int64_t A, int64_t B) {
  struct complex theta, minustwentyfourth = {-ONE / TWENTYFOUR, 0},
                        twelth = {ONE / TWELVE, 0};
  struct complex theta12, theta24, minustheta, theta2;
  struct complex result1, result2, result;
  struct complex root_two;
  struct complex two = {TWO, 0};

  root_two = csqrt2(two);

  theta = Theta(D, A, B);

  theta24 = cpow2(theta, minustwentyfourth);
  theta12 = cpow2(theta, twelth);
  theta2 = cmul(theta, theta);

  minustheta.x = -theta.x;
  minustheta.y = -theta.y;

  if (j == 0)
    result1 = Sum(minustheta);
  if (j == 1)
    result1 = Sum(theta);
  if (j == 2)
    result1 = Sum(cmul(cmul(theta, theta), cmul(theta, theta)));

  result2 = cdiv(result1, Sum(theta2));

  if (j == 0 || j == 1)
    result = cmul(theta24, result2);
  if (j == 2)
    result = cmul(root_two, cmul(theta12, result2));

  return result;
}

// the following functions implement the routines described in
// IEEEP1363 Standard

int64_t
G(int64_t D) {
  if (D % 3 == 0)
    return 3;
  else
    return 1;
}

int64_t I(int64_t D) {
  int64_t t1, t2;
  int64_t result = 0;

  t1 = D % 8;
  t2 = D % 3;

  if (t1 == 1 || t1 == 2 || t1 == 6 || t1 == 7)
    result = 3;

  if (t1 == 3 && t2 != 0)
    result = 0;

  if (t1 == 3 && t2 == 0)
    result = 2;

  if (t1 == 5)
    result = 6;

  return result;
}

int64_t J1(int64_t A, int64_t C) {

  int64_t t1, t2;
  int64_t result = 0;

  t1 = A * C;

  t2 = t1 % 2;

  if (t2 == 1)
    result = 0;

  t2 = C % 2;

  if (t2 == 0)
    result = 1;

  t2 = A % 2;

  if (t2 == 0)
    result = 2;

  return result;
}

int64_t K(int64_t D) {

  int64_t t1;
  int64_t result = 0;

  t1 = D % 8;

  if (t1 == 1 || t1 == 2 || t1 == 6)
    result = 2;
  if (t1 == 3 || t1 == 7)
    result = 1;
  if (t1 == 5)
    result = 4;

  return result;
}

int64_t L(int64_t D, int64_t A, int64_t C) {

  int64_t t1, t2, t3;
  int64_t result = 0;

  t1 = D % 8;
  t2 = (A * C) % 2;

  t3 = C % 2;

  if (t2 == 1 || (t1 == 5 && t3 == 0))
    result = A * A * C + A - C;
  if ((t1 == 1 || t1 == 2 || t1 == 3 || t1 == 6 || t1 == 7) && t3 == 0)
    result = A + C + C - A * C * C;

  t3 = A % 2;

  if (t1 == 3 && t3 == 0)
    result = A - C + 5 * A * C * C;

  if ((t1 == 1 || t1 == 2 || t1 == 5 || t1 == 6 || t1 == 7) && t3 == 0)
    result = A - C - A * C * C;

  return result;
}

int64_t M(int64_t A, int64_t C) {

  int64_t t1, t2, t3;
  int64_t result = 0;

  t1 = A % 2;

  if (t1 == 1) {
    t2 = (A * A - 1) / 8;
    t3 = t2 % 2;

    if (t3 == 0)
      result = 1;

    if (t3 == 1)
      result = -1;
  }

  if (t1 == 0) {
    t2 = (C * C - 1) / 8;
    t3 = t2 % 2;

    if (t3 == 0)
      result = 1;

    if (t3 == 1)
      result = -1;
  }

  return result;
}

int64_t N(int64_t D, int64_t A, int64_t C) {

  int64_t t1, t2;
  int64_t result = 0;

  t1 = D % 8;
  t2 = (A * C) % 2;

  if (t1 == 5 || (t1 == 3 && t2 == 1) || (t1 == 7 && t2 == 0))
    result = 1;
  if ((t1 == 1 || t1 == 2 || t1 == 6) || (t1 == 7 && t2 == 1))
    result = M(A, C);
  if (t1 == 3 && t2 == 0)
    result = -M(A, C);

  return result;
}

// Weber switch as listed "Explicit Construction of the Hilbert Class
// Fields of Imaginary Quadratic Fields by Integer Lattice Reduction"
// by Kaltofen-Yui p. 178 and "Elliptic Curves and Primality Proving"
// by Atkin-Morain p. 17  and Weber "Lehrbuch der Algebra III" Section
// 127
struct complex u(int64_t d, int64_t a, int64_t b, int64_t c) {
  struct complex result;
  struct complex f, temp1, temp2, temp3, temp4, temp5;
  int64_t t1;

  int64_t n, l, i, k, j, g;

  n = N(d, a, c);
  l = L(d, a, c);
  i = I(d);
  k = K(d);
  j = J1(a, c);
  g = G(d);

  f = F(j, d, a, b);

  temp1.x = k;
  temp1.y = 0;
  temp2 = cpow2(f, temp1);

  temp3.x = 0;
  temp3.y = -PIf / TWENTYFOUR;
  t1 = k * b * l;
  t1 = t1 % 48;
  if (t1 < 0)
    t1 += 48;
  temp3.y = temp3.y * t1;

  temp1 = cexp2(temp3);

  temp1.x *= n;
  temp1.y *= n;

  temp3.x = 2;
  temp3.y = 0;
  temp4.x = -i / SIX;
  temp4.y = 0;
  temp5 = cpow2(temp3, temp4);

  temp3 = cmul(temp1, temp5);
  temp4 = cmul(temp3, temp2);

  temp5.x = g;
  temp5.y = 0;

  result = cpow2(temp4, temp5);

  // if (d % 8 == -3)
  // result = j(tau);
  return result;
}

// Algorithm 7.6.1 (Hilbert Class Polynomial). See "A Course
// in Computational Algebraic Number Theory" by Henri
// Cohen page 415. Given a negative discriminant D, this
// algorithm computes the monic polynomial of degree h(D)
// in Z[X] of which j((D+sqrt(D))/2) is a root.
void Hilbert(int64_t D, mpz_class *P, int64_t *dP) {
  int64_t B, i, t, a, b, d, dQ, dR;
  struct complex J, aa = {ZERO, ZERO}, bb = {ZERO, ZERO};
  struct complex one = {ONE, ZERO}, two = {TWO, ZERO};
  struct complex DD = {ZERO, ZERO}, sqrtD, tau;
  struct complex PP[POLY_SIZE], Q[3], R[POLY_SIZE];

  d = D;

  DD.x = d;
  DD.y = 0;

  sqrtD = csqrt2(DD);

  PP[0].x = 1;
  PP[0].y = 0;
  *dP = 0;

  b = D % 2;
  if (b < 0)
    b += 2;
  B = (long)sqrt(labs(D) / 3.0);
L2:
  t = (b * b - D) / 4;
  a = max(b, 1);
L3:
  if (t % a != 0)
    goto L4;

  aa.x = 2 * a;
  aa.y = 0;
  bb.x = -b;
  bb.y = 0;

  tau = cdiv(cadd(bb, sqrtD), aa);

  J = j(tau);

  if (a == b || a * a == t || b == 0) {
    Q[0].x = -J.x;
    Q[0].y = -J.y;
    Q[1].x = 1;
    Q[1].y = 0;
    dQ = 1;
  } else {
    Q[0].x = J.x * J.x + J.y * J.y;
    Q[0].y = 0;
    Q[1].x = -2 * J.x;
    Q[1].y = 0;
    Q[2].x = 1;
    Q[2].y = 0;
    dQ = 2;
  }

  cpoly_mul(*dP, dQ, PP, Q, R, &dR);

  *dP = dR;
  for (i = 0; i <= dR; i++)
    PP[i] = R[i];
L4:
  a++;
  if (a * a <= t)
    goto L3;

  b += 2;
  if (b <= B)
    goto L2;

  for (i = 0; i <= *dP; i++)
    P[i] = round2(PP[i].x);
}

void Weber(int64_t D, mpz_class *P, int64_t *dP) {
  int64_t t1, t2, t3;
  mpz_class temp2;
  int64_t i, a, b, A, B1, C, h1, help1, x2, temp1, dQ, dR;
  struct complex J;
  struct complex PP[POLY_SIZE], Q[3], R[POLY_SIZE];

  if (D % 4 == 0)
    D /= 4;

  help1 = (long)sqrt(labs(D) * 4.0 / 3.0);

  PP[0].x = 1;
  PP[0].y = 0;
  *dP = 0;

  b = 0;

  while (b <= help1) {
    t1 = b * b;
    t2 = -D * 4;
    t3 = t1 + t2;
    t1 = 4;

    t2 = t3 % t1;

    if (t2 == 0) {
      t1 = t3 / 4;

      if (b > 1)
        a = b;
      else
        a = 1;

      x2 = (long)sqrt((double)t1);

      while (a <= x2) {
        h1 = t1 % a;
        if (h1 == 0) {
          C = t1 / a;
          B1 = b;
          A = a;

          temp2 = gcd(gcd(A, B1), C);

          if (temp2 == 1) {
            temp1 = B1 / 2;

            J = u(-D, A, temp1, C);

            if (!((B1 > 0) && (C > A) && (A > B1))) {
              Q[0].x = -J.x;
              Q[0].y = -J.y;
              Q[1].x = 1;
              Q[1].y = 0;
              dQ = 1;
            } else {
              Q[0].x = J.x * J.x + J.y * J.y;
              Q[0].y = 0;
              Q[1].x = -2 * J.x;
              Q[1].y = 0;
              Q[2].x = 1;
              Q[2].y = 0;
              dQ = 2;
            }

            cpoly_mul(*dP, dQ, PP, Q, R, &dR);

            *dP = dR;
            for (i = 0; i <= dR; i++)
              PP[i] = R[i];
          }
        }
        a++;
      }
    }
    b++;
  }
  for (i = 0; i <= *dP; i++)
    P[i] = round2(PP[i].x);
}

// Hilbert root from Watson root as listed "An Improved Las Vegas
// Primality Test" by Kaltofen-Valente 1989, p.4
mpz_class Vegas(mpz_class u, mpz_class N, int64_t d) {
  mpz_class j;
  mpz_class two = 2, three = 3, four = 4, eight = 8;
  mpz_class A = 0;
  bool passthrough = false;

  switch (d % 8) {
  case -1:
    A = 4 * exp_mod(inverse(u, N), four, N);
    break;
  case -2:
  case -6:
    A = -4 * exp_mod(u, four, N);
    break;
  case -3:
    A = 16 * exp_mod(inverse(u, N), eight, N);
    break;
  case -4:
    A = -8 * exp_mod(u, two, N);
    break;
  case -5:
    A = 4 * exp_mod(inverse(u, N), two, N);
    break;
  case -7:
    A = exp_mod(inverse(u, N), eight, N);
    break;
  default:
    // D == 0 mod 32
    passthrough = true;
    break;
  }

  if (d % 3 != 0)
    A = exp_mod(A, three, N);

  j = inverse(A, N) * exp_mod(A - 16, three, N);
  j = modpos(j, N);

  // if (d % 8 == -3)
  // j = u;
  if (passthrough)
    j = u;

  return j;
}

// Algorithm 1.5.3 (Modified Cornacchia). See "A Course
// in Computational Algebraic Number Theory" by Henri
// Cohen page 36. Let p be a prime number and D be a
// negative number such that D = 0 or 1 modulo 4 and
// | D | < 4 * p. This algorithm either outputs an
// integer solution (x, y) to the Diophantine equation
// x * x + | D | * y * y = 4 * p, or says that such a
// solution does not exist.
int32_t modified_Cornacchia(mpz_class D, mpz_class p, mpz_class *x,
                            mpz_class *y) {
  int32_t value = 0;
  mpz_class dd, xx;
  mpz_class a = 0, b = 0, c = 0, d = 0, e = 0;
  mpz_class l = 0, r = 0, x0 = 0;

  if (JACOBI(D, p) != -1) {
    x0 = square_root_mod(D, p);

    dd = modpos(D, 2);
    xx = modpos(x0, 2);

    if (dd != xx)
      x0 = p - x0;

    a = p * 2;
    b = x0;
    c = sqrt(p);

    l = c * 2;

    while (b > l) {
      r = modpos(a, b);
      a = b;
      b = r;
    }

    c = p * 4;
    a = b * b;
    e = c - a;
    d = abs(D);

    c = e / d;
    r = modpos(e, d);

    if ((r == 0) && square_test(c, y)) {
      *x = b;
      value = 1;
    }
  }

  return value;
}

void zpoly_print(int64_t da, mpz_class *za) {
  int64_t i;

  if (!Quiet) {
    for (i = da; i >= 0; i--) {
      gmpecpp_detail::log() << za[i] << " ";
    }
    gmpecpp_detail::log() << "\n";
  }
}

// multiplication (xy) modulo m
void myzmulmod(mpz_t *res, mpz_t *x, mpz_t *y, mpz_t *m) {
  mpz_t temp;

  mpz_init(temp);

  mpz_mul(temp, *x, *y);
  mpz_mod(*res, temp, *m);

  mpz_clear(temp);
}

// division (x/y) modulo m
void myzdivmod(mpz_t *res, mpz_t *x, mpz_t *y, mpz_t *m) {
  int32_t i;
  mpz_t h1;

  mpz_init(h1);

  i = mpz_invert(h1, *y, *m);

  if (i == 0) {
    // A missing inverse exhibits gcd(y, m) as a nontrivial factor.
    mpz_t g;
    mpz_init(g);
    mpz_gcd(g, *y, *m);
    char *s = mpz_get_str(NULL, 10, g);
    std::string f(s ? s : "");
    void (*fr)(void *, size_t);
    mp_get_memory_functions(NULL, NULL, &fr);
    if (s)
      fr(s, f.size() + 1);
    mpz_clear(g);
    mpz_clear(h1);
    throw gmpecpp_detail::CompositeFound(f);
  }

  else
    myzmulmod(res, x, &h1, m);

  mpz_clear(h1);
}

void zpoly_copy(int64_t da, mpz_t *za, mpz_t *zb, int64_t *db) {

  int64_t i;

  *db = da;

  for (i = 0; i <= da; i++)
    mpz_set(zb[i], za[i]);
}

void zpoly_mul(int64_t m, int64_t n, mpz_t *za, mpz_t *zb, mpz_t *zc,
               int64_t *p) {

  int64_t i, j, k;

  mpz_t zd, zai, zbk, zsum, zterm;

  mpz_init(zd);
  mpz_init(zai);
  mpz_init(zbk);
  mpz_init(zsum);
  mpz_init(zterm);

  *p = m + n;

  for (k = 0; k <= *p; k++) {

    mpz_set_ui(zsum, 0);

    for (i = 0; i <= k; i++) {

      j = k - i;

      if (i > m)
        mpz_set_ui(zai, 0);

      else
        mpz_set(zai, za[i]);

      if (j > n)
        mpz_set_ui(zbk, 0);

      else
        mpz_set(zbk, zb[j]);

      mpz_mul(zterm, zai, zbk);

      mpz_set(zd, zsum);

      mpz_add(zsum, zterm, zd);
    }

    mpz_set(zc[k], zsum);
  }

  mpz_clear(zd);
  mpz_clear(zai);
  mpz_clear(zbk);
  mpz_clear(zsum);
  mpz_clear(zterm);
}

void zpoly_div(int64_t m, int64_t n, mpz_t *zu, mpz_t *zv, mpz_t *zq, mpz_t *zr,
               int64_t *p, int64_t *s) {

  int64_t j, jk, k, nk;

  mpz_t za, zb, zvn;

  mpz_init(za);
  mpz_init(zb);
  mpz_init(zvn);

  mpz_set(zvn, zv[n]);

  for (j = 0; j <= m; j++)

    mpz_set(zr[j], zu[j]);

  if (m < n) {

    *p = 0, *s = m;

    mpz_set_ui(zq[0], 0);

  }

  else {

    *p = m - n, *s = n - 1;

    for (k = *p; k >= 0; k--) {

      nk = n + k;

      mpz_pow_ui(za, zvn, k);

      mpz_mul(zq[k], zr[nk], za);

      for (j = nk - 1; j >= 0; j--) {

        jk = j - k;

        if (jk >= 0) {

          mpz_mul(za, zvn, zr[j]);

          mpz_mul(zb, zr[nk], zv[jk]);

          mpz_sub(zr[j], za, zb);

        }

        else {

          mpz_set(za, zr[j]);

          mpz_mul(zr[j], zvn, za);
        }
      }
    }

    while (*p > 0 && mpz_cmp_ui(zq[*p], 0l) == 0)
      *p = *p - 1;

    while (*s > 0 && mpz_cmp_ui(zr[*s], 0l) == 0)
      *s = *s - 1;
  }

  mpz_clear(za);
  mpz_clear(zb);
  mpz_clear(zvn);
}

void zpoly_mod(mpz_t *zp, mpz_t *za, int64_t *da) {

  int64_t i;

  mpz_t zb;

  mpz_init(zb);

  for (i = 0; i <= *da; i++) {

    mpz_mod(zb, za[i], *zp);

    mpz_set(za[i], zb);
  }

  while (*da > 0 && mpz_cmp_ui(za[*da], 0l) == 0)
    *da = *da - 1;

  mpz_clear(zb);
}

void zpoly_pow(int64_t degreeA, int64_t degreem, mpz_t *zn, mpz_t *zp,
               mpz_t *zA, mpz_t *zm, mpz_t *zs, int64_t *ds) {

  int64_t dP, dq, dx = degreeA, i;

  mpz_t za, zb, zP[POLY_SIZE], zq[POLY_SIZE], zx[POLY_SIZE], zy[POLY_SIZE];

  mpz_init(za);
  mpz_init(zb);

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_init(zP[i]);
    mpz_init(zq[i]);
    mpz_init(zx[i]);
    mpz_init(zy[i]);
  }

  *ds = 0;

  mpz_set(za, *zn);

  mpz_set_ui(zs[0], 1l);

  for (i = 0; i <= dx; i++)
    mpz_set(zx[i], zA[i]);

  while (mpz_cmp_ui(za, 0l) > 0) {

    if (mpz_tdiv_ui(za, 2l) == 1) {

      zpoly_mul(*ds, dx, zs, zx, zP, &dP);

      zpoly_div(dP, degreem, zP, zm, zq, zs, &dq, ds);

      zpoly_mod(zp, zs, ds);
    }

    mpz_set(zb, za);

    mpz_tdiv_q_2exp(za, zb, 1l);

    if (mpz_cmp_ui(za, 0l) > 0) {

      for (i = 0; i <= dx; i++)
        mpz_set(zy[i], zx[i]);

      zpoly_mul(dx, dx, zx, zy, zP, &dP);

      zpoly_div(dP, degreem, zP, zm, zq, zx, &dq, &dx);

      zpoly_mod(zp, zx, &dx);
    }
  }

  mpz_clear(za);
  mpz_clear(zb);

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_clear(zP[i]);
    mpz_clear(zq[i]);
    mpz_clear(zx[i]);
    mpz_clear(zy[i]);
  }
}

void zpoly_sub(int64_t da, int64_t db, mpz_t *za, mpz_t *zb, mpz_t *zc,
               int64_t *dc) {

  int64_t i;

  mpz_t zz;

  mpz_init(zz);

  mpz_set_ui(zz, 0);

  if (da >= db) {

    for (i = 0; i <= db; i++)

      mpz_sub(zc[i], za[i], zb[i]);

    for (i = db + 1; i <= da; i++)

      mpz_set(zc[i], za[i]);

    *dc = da;

  }

  else {

    for (i = 0; i <= da; i++)

      mpz_sub(zc[i], za[i], zb[i]);

    for (i = da + 1; i <= db; i++)

      mpz_sub(zc[i], zz, zb[i]);

    *dc = db;
  }

  mpz_clear(zz);
}

void zpoly_gcd(int64_t degreeA, int64_t degreeB, mpz_t *zp, mpz_t *zA,
               mpz_t *zB, mpz_t *za, int64_t *da) {

  int32_t nonzero = 0, zero;

  int64_t db, dq, dr, i;

  mpz_t zc;

  mpz_t zb[POLY_SIZE], zq[POLY_SIZE], zr[POLY_SIZE];

  mpz_init(zc);

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_init(zb[i]);
    mpz_init(zq[i]);
    mpz_init(zr[i]);
  }

  if (degreeA > degreeB) {

    *da = degreeA;

    db = degreeB;

    for (i = 0; i <= *da; i++)
      mpz_set(za[i], zA[i]);

    for (i = 0; i <= db; i++)
      mpz_set(zb[i], zB[i]);

  }

  else {

    *da = degreeB;

    db = degreeA;

    for (i = 0; i <= *da; i++)
      mpz_set(za[i], zB[i]);

    for (i = 0; i <= db; i++)
      mpz_set(zb[i], zA[i]);
  }

  for (i = 0; i <= db && !nonzero; i++)

    nonzero = mpz_cmp_ui(zb[i], 0l) != 0;

  while (nonzero) {

    zpoly_div(*da, db, za, zb, zq, zr, &dq, &dr);

    for (i = 0; i <= dr; i++) {

      mpz_set(zc, zr[i]);

      mpz_mod(zr[i], zc, *zp);
    }

    zero = 1;

    for (i = dr; i >= 0 && zero; i--) {

      zero = mpz_cmp_ui(zr[i], 0l) == 0;

      if (zero && dr > 0)
        dr--;
    }

    for (i = 0; i <= db; i++)
      mpz_set(za[i], zb[i]);

    *da = db;

    for (i = 0; i <= dr; i++)
      mpz_set(zb[i], zr[i]);

    db = dr;

    nonzero = 0;

    for (i = 0; i <= db && !nonzero; i++)

      nonzero = mpz_cmp_ui(zb[i], 0l) != 0;
  }

  mpz_clear(zc);

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_clear(zb[i]);
    mpz_clear(zq[i]);
    mpz_clear(zr[i]);
  }
}

// find the roots of the polynomial zA modulo zp
void Recurse(int64_t degreeA, mpz_t *zp, mpz_t *zA, mpz_t *zroot,
             int64_t *rootSize) {

  int64_t dd, degreeB, dq, dr, du = 1, i, flag = 0;

  mpz_t zD, za, zb, zc, ze;

  mpz_t zn, x0;

  mpz_t zB[POLY_SIZE], zd[POLY_SIZE];

  mpz_t zq[POLY_SIZE], zr[POLY_SIZE];

  mpz_t zu[2];

  mpz_init(zD);
  mpz_init(za);
  mpz_init(zb);
  mpz_init(zc);
  mpz_init(ze);
  mpz_init(zn);
  mpz_init(x0);

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_init(zB[i]);
    mpz_init(zd[i]);
    mpz_init(zq[i]);
    mpz_init(zr[i]);
  }

  mpz_init(zu[0]);
  mpz_init(zu[1]);

  mpz_set_ui(x0, (long)102);

  if (mpz_cmp_ui(zA[degreeA], (long)1) != 0) {
    for (i = 0; i < (degreeA + 1); i++) {
      myzdivmod(&zA[i], &zA[i], &zA[degreeA], zp);
    }
  }

  while (degreeA != 1) {
    i = 0;
    do {

      mpz_sub_ui(za, *zp, 1l);
      mpz_tdiv_q_2exp(zn, za, 1l);

      mpz_add_ui(x0, x0, (long)1);

      mpz_set(zu[0], x0);
      mpz_set_ui(zu[1], 1l);

      zpoly_mod(zp, zu, &du);

      zpoly_pow(du, degreeA, &zn, zp, zu, zA, zd, &dd);
      zpoly_mod(zp, zd, &dd);

      mpz_sub_ui(zd[0], zd[0], 1l);

      zpoly_gcd(dd, degreeA, zp, zd, zA, zB, &degreeB);

      zpoly_mod(zp, zB, &degreeB);

      if ((mpz_cmp_ui(x0, (long)200) == 1) || *rootSize > (degreeA - 1)) {
        flag = 1;

        goto L1;
      }

    } while ((degreeB == 0 || degreeB == degreeA));

    if (degreeB >= 1 && flag != 1) {

      Recurse(degreeB, zp, zB, zroot, rootSize);

      zpoly_div(degreeA, degreeB, zA, zB, zq, zr, &dq, &dr);
      zpoly_mod(zp, zq, &dq);
      zpoly_copy(dq, zq, zA, &degreeA);

      Recurse(degreeA, zp, zA, zroot, rootSize);
    }
  }

  if (degreeA == 1) {
    mpz_invert(za, zA[1], *zp);
    mpz_mul(zb, zA[0], za);
    mpz_neg(zb, zb);
    mpz_mod(zroot[*rootSize], zb, *zp);

    *rootSize = *rootSize + 1;
  }

L1:

  mpz_clear(zD);
  mpz_clear(za);
  mpz_clear(zb);
  mpz_clear(zc);
  mpz_clear(ze);
  mpz_clear(zn);
  mpz_clear(x0);

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_clear(zB[i]);
    mpz_clear(zd[i]);
    mpz_clear(zq[i]);
    mpz_clear(zr[i]);
  }

  mpz_clear(zu[0]);
  mpz_clear(zu[1]);
}

// Algorithm 1.6.1 (Roots Mod p). See "A Course
// in Computational Algebraic Number Theory" by
// Henri Cohen page 37.
void FindRootsModuloAPrime(int64_t degreeP, mpz_class p, mpz_class *P,
                           mpz_class *root, int64_t *rootSize) {
  int64_t i, j;
  mpz_class t;
  mpz_t zp, zA[POLY_SIZE], zroot[POLY_SIZE];

  mpz_init(zp);

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_init(zA[i]);
    mpz_init(zroot[i]);
  }

  mpz_set(zp, p.get_mpz_t());

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_set(zA[i], P[i].get_mpz_t());
    mpz_set(zroot[i], root[i].get_mpz_t());
  }

  *rootSize = 0;

  Recurse(degreeP, &zp, zA, zroot, rootSize);

  for (i = 0; i < POLY_SIZE; i++) {
    mpz_class temp_class(zA[i]);
    P[i] = temp_class;
    mpz_class temp_class2(zroot[i]);
    root[i] = temp_class2;
  }

  // sort the roots using selection sort
  for (i = 0; i < *rootSize - 1; i++) {
    for (j = i + 1; j < *rootSize; j++) {
      if (root[i] > root[j]) {
        t = root[i];
        root[i] = root[j];
        root[j] = t;
      }
    }
  }

  mpz_clear(zp);
  for (i = 0; i < POLY_SIZE; i++) {
    mpz_clear(zA[i]);
    mpz_clear(zroot[i]);
  }
}

// Algorithm 10.3.3 (Lenstra's ECM). See "A Course
// in Computational Algebraic Number Theory" by
// Henri Cohen page 488.
int32_t LenstrasECM(mpz_class *N, mpz_class *g, mpz_class Bmax);

/*
 * Parallel Lenstra ECM (POSIX threads).
 *
 * Curves are independent trials and one success ends the search, which
 * makes this the most parallel thing in the prover -- and, at the default
 * precision, the thing it spends 52 to 99.5% of its time on.
 *
 * Each worker owns a gmp_randstate_t; that is the only GMP state the
 * prover mutates rather than merely reads.  Everything else the inner
 * loop touches (multiply, gcd, modpos, nextp) works on locals and was
 * verified thread safe on disjoint objects.
 *
 * Unlike the GPU backend this has no per-launch cost, so it is useful at
 * every input size rather than only large ones.  Like the GPU backend it
 * changes *which* factor is found, so certificates differ from the
 * sequential run while remaining valid -- check them with
 * gmpecpp-verify, not by byte comparison.
 */
struct EcmShared {
  const mpz_class *N;
  mpz_class bound;
  unsigned long seed;
  int nthreads;
  int curves_per_bound;

  pthread_mutex_t mtx;
  // Read on every inner iteration and written once. volatile would not
  // be enough: it orders nothing and the read/write pair is a data race
  // that ThreadSanitizer correctly flags.
  std::atomic<int> found;
  mpz_class factor;
};

struct EcmWorker {
  EcmShared *sh;
  int id;
};

static void *ecm_thread(void *arg) {
  EcmWorker *w = (EcmWorker *)arg;
  EcmShared *sh = w->sh;

  gmp_randstate_t st;
  gmp_randinit_default(st);
  // Distinct, reproducible stream per worker.
  gmp_randseed_ui(st, sh->seed + 0x9e3779b9u * (unsigned long)(w->id + 1));
  tls_rand = st;

  const mpz_class &N = *sh->N;
  mpz_class B = 1000l;

  while (mpz_cmp(B.get_mpz_t(), sh->bound.get_mpz_t()) < 0 &&
         !sh->found.load(std::memory_order_relaxed)) {
    for (int c = w->id; c < sh->curves_per_bound && !sh->found;
         c += sh->nthreads) {
      mpz_class a = modpos(rand2_r(st), N);
      mpz_class d = 0, l, q, q1, newq;
      struct point x, y;
      x.x = 0;
      x.y = 1;
      int32_t hit = 0;

      newq = 2;
      for (; newq < B && hit != 1 && !sh->found.load(std::memory_order_relaxed);) {
        q = newq;
        q1 = q;
        l = B / q;
        while (q1 <= l)
          q1 *= q;
        hit = multiply(a, q1, N, x, &y, &d);
        x.x = y.x;
        x.y = y.y;
        newq = nextp(q);
      }

      mpz_class g = gcd(d, N);
      if (g != N && g != 1) {
        pthread_mutex_lock(&sh->mtx);
        if (!sh->found.load(std::memory_order_relaxed)) {
          sh->factor = g;
          sh->found.store(1, std::memory_order_relaxed);
        }
        pthread_mutex_unlock(&sh->mtx);
        break;
      }
    }
    B = B * 5;
  }

  tls_rand = 0;
  gmp_randclear(st);
  return NULL;
}

int32_t LenstrasECM_mt(mpz_class *N, mpz_class *g, mpz_class bound,
                       int nthreads, unsigned long seed) {
  if (nthreads < 2)
    return LenstrasECM(N, g, bound);

  EcmShared sh;
  sh.N = N;
  sh.bound = bound;
  sh.seed = seed;
  sh.nthreads = nthreads;
  // The sequential version tries 20 curves per bound; never do fewer, and
  // never leave a worker idle.
  sh.curves_per_bound = nthreads > 20 ? nthreads : 20;
  sh.found.store(0, std::memory_order_relaxed);
  pthread_mutex_init(&sh.mtx, NULL);

  std::vector<pthread_t> th(nthreads);
  std::vector<EcmWorker> w(nthreads);
  int started = 0;
  for (int i = 0; i < nthreads; i++) {
    w[i].sh = &sh;
    w[i].id = i;
    if (pthread_create(&th[i], NULL, ecm_thread, &w[i]) == 0)
      started++;
    else
      break;
  }
  for (int i = 0; i < started; i++)
    pthread_join(th[i], NULL);
  pthread_mutex_destroy(&sh.mtx);

  if (!started)               // could not create any thread; fall back
    return LenstrasECM(N, g, bound);

  if (!sh.found.load(std::memory_order_relaxed))
    return 0;
  *g = sh.factor;
  *N /= *g;
  return 1;
}

int32_t LenstrasECM(mpz_class *N, mpz_class *g, mpz_class Bmax) {
  int32_t found = 0;
  mpz_class B = 1000l;
  mpz_class l, q, q1, newq;
  struct point x, y;
  mpz_class a, d = 0;
  int64_t curve = 0;

  do {

    do {
      a = modpos(rand2(), *N);
      x.x = 0;
      x.y = 1;

      curve++;

      newq = 2;
      for (; newq < B && found != 1;) {
        q = newq;
        q1 = q;
        l = B / q;
        while (q1 <= l)
          q1 *= q;

        found = multiply(a, q1, *N, x, &y, &d);

        x.x = y.x;
        x.y = y.y;

        newq = nextp(q);
      }

      *g = gcd(d, *N);

    } while (curve < 20 && (*g == *N || *g == 1));

    B = B * 5;
    curve = 0;

  } while (B < Bmax && (*g == *N || *g == 1));

  if (*g == *N || *g == 1)
    return 0;
  else {
    *N /= *g;
    return 1;
  }
}

// return true if t < *q < m and *q prime
/*
 * ECM dispatch.  This is the only stage of the prover that parallelises
 * usefully -- every curve is an independent trial and one success is
 * enough -- so it is where the CUDA/CGBN backend is applied.
 *
 * The GPU runs many curves in one launch at a fixed bound.  If none of
 * them splits n we fall through to the CPU implementation, which keeps
 * escalating its own bound, so the CUDA path can only add attempts, never
 * remove them: a proof that succeeds on the CPU still succeeds here.
 *
 * Mirrors LenstrasECM's contract: returns 1 having set *g to a factor and
 * divided it out of *N, or 0.
 */
int32_t ecm_find_factor(mpz_class *N, mpz_class *g, mpz_class bound) {
  double t0 = gmpecpp_detail::wall_seconds();

  // The CPU attempt runs first even when CUDA is selected.  Measurement
  // says the GPU kernel costs a fixed ~0.25s at B1=10000 and ~1.7s at
  // B1=50000 -- it is latency bound on the serial chain of prime powers,
  // not throughput bound -- while the CPU splits the small cofactors this
  // prover usually meets in tens of milliseconds.  Going to the GPU first
  // would therefore make almost every proof slower.  The GPU earns its
  // place only once the CPU has failed, where its thousands of concurrent
  // curves find factors the CPU misses entirely.
  //
  // Threads, when enabled, apply to that host attempt: curves are
  // independent, so the twenty trials per bound become twenty parallel
  // trials, with no launch cost to amortise.
  int32_t found = (gmpecpp_detail::g_threads > 1)
                      ? LenstrasECM_mt(N, g, bound, gmpecpp_detail::g_threads,
                                       gmpecpp_detail::g_seed)
                      : LenstrasECM(N, g, bound);

  // Only escalate to the GPU when the cofactor is big enough that its
  // fixed kernel cost can pay for itself; see cuda_min_bits.
  bool worth_gpu =
      gmpecpp_detail::g_cuda_min_bits == 0 ||
      mpz_sizeinbase(N->get_mpz_t(), 2) >= gmpecpp_detail::g_cuda_min_bits;

  if (!found && worth_gpu &&
      gmpecpp_detail::g_backend == GMPECPP_BACKEND_CUDA) {
    unsigned int curves =
        gmpecpp_detail::g_cuda_curves ? gmpecpp_detail::g_cuda_curves : 4096;
    unsigned long b1 = (unsigned long)mpz_get_ui(bound.get_mpz_t());
    if (b1 < 1000)
      b1 = 1000;
    if (b1 > 100000)
      b1 = 100000;
    // Stage 1 and stage 2 on the GPU (the resumable kernels behind
    // gmpecpp_cuda_ecm()). B2 follows what measured best per curve-second:
    // 100*B1 below 5*10^4, 50*B1 from there up. Stage 2 makes each curve
    // about ten times likelier to split the cofactor for under twice the time.
    const unsigned long b2 = (b1 >= 50000UL ? 50UL : 100UL) * b1;
    mpz_t f;
    mpz_init(f);
    if (gmpecpp_detail::cuda_ecm_run(N->get_mpz_t(), b1, b2, curves,
                                     gmpecpp_detail::g_seed, nullptr,
                                     nullptr, f) == 1) {
      *g = mpz_class(f);
      *N /= *g;
      found = 1;
    }
    mpz_clear(f);
  }

  {
    static pthread_mutex_t acc = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&acc);
    gmpecpp_detail::state().ecm_seconds += gmpecpp_detail::wall_seconds() - t0;
    pthread_mutex_unlock(&acc);
  }
  return found;
}

bool check_for_factor(mpz_class *q, mpz_class m, mpz_class t) {
  *q = m;

  if (Rabin_Miller(*q))
    return false;

  // trial division
  while (*q % 2 == 0)
    *q /= 2;
  while (*q % 3 == 0)
    *q /= 3;

  if (*q < t)
    return false;
  if (*q < m && Rabin_Miller(*q))
    return true;

  mpz_class d = 1;
  mpz_class oldq;

  do {
    oldq = *q;
    ecm_find_factor(q, &d, Bmax);
    if (*q < t)
      return false;
    if (*q < m && Rabin_Miller(*q))
      return true;
  } while (*q < oldq);

  return false;
}

bool find_curve(int32_t type, mpz_class *a, mpz_class *b, int64_t D,
                mpz_class N, mpz_class *root, int64_t *rootSize) {

  mpz_class T[POLY_SIZE];
  int64_t dT;

  mpz_t zp, zA[POLY_SIZE];
  int64_t i;

  *rootSize = 0;

  mpz_init(zp);
  for (i = 0; i < POLY_SIZE; i++) {
    mpz_init(zA[i]);
  }

  if (type == 0) {
    if (D == -3) {
      *a = 0;
      *b = -1;
      *rootSize = 1;
      goto clear_variables;
    } else if (D == -4) {
      *a = -1;
      *b = 0;
      *rootSize = 1;
      goto clear_variables;
    }
  }

  if (hilbert && type == 1) {
    Hilbert(D, T, &dT);

    if (!Quiet) {
      gmpecpp_detail::log() << "D = " << D << ", dT = " << dT << ", T = ";
      zpoly_print(dT, T);
    }

    mpz_class SmallestCoeff, Root3;
    SmallestCoeff = T[0];
    if (cube_test(SmallestCoeff, &Root3) == 0) {
      if (!Quiet) {
        gmpecpp_detail::log() << "HCP loss of precision\n";
      }
      *rootSize = 0;
      goto clear_variables;
    }
  }

  if (weber && type == 2 && D % 32 != 0) {
    Weber(D, T, &dT);

    if (!Quiet) {
      gmpecpp_detail::log() << "D = " << D << ", dW = " << dT << ", W = ";
      zpoly_print(dT, T);
    }

    mpz_class SmallestCoeff;
    SmallestCoeff = T[0];
    if (poweroftwo_test(SmallestCoeff) == false) {
      if (!Quiet) {
        gmpecpp_detail::log() << "WCP loss of precision\n";
      }
      *rootSize = 0;
      goto clear_variables;
    }
  }

  if ((hilbert && type == 1) || (weber && type == 2 && D % 32 != 0)) {
    mpz_set(zp, N.get_mpz_t());
    for (i = 0; i < POLY_SIZE; i++) {
      mpz_set(zA[i], T[i].get_mpz_t());
    }
    zpoly_mod(&zp, zA, &dT);
    for (i = 0; i < POLY_SIZE; i++) {
      mpz_class temp_class(zA[i]);
      T[i] = temp_class;
    }

    FindRootsModuloAPrime(dT, N, T, root, rootSize);
  }

clear_variables:
  mpz_clear(zp);
  for (i = 0; i < POLY_SIZE; i++) {
    mpz_clear(zA[i]);
  }

  if (*rootSize > 0)
    return true;
  else
    return false;
}

// returns 2 if N is composite, 1 if probably prime, 0 if proven prime

// ------------------------------------------------------------------ //
// The discriminant sweep, factored out so it can run in parallel      //
// ------------------------------------------------------------------ //

namespace gmpecpp_detail {

enum SweepOutcome { SWEEP_REJECTED = 0, SWEEP_FOUND = 1, SWEEP_COMPOSITE = 2 };

struct SweepResult {
  int outcome;
  int tag;              // which "N factor =" message a composite came from
  int64_t D;
  mpz_class a, b, m, q;
  struct point P, P1, P2;
  std::string factor;
  SweepResult() : outcome(SWEEP_REJECTED), tag(0), D(0) {}
};

/*
 * Try one discriminant against Ni.  This is the body of the sweep loop,
 * lifted verbatim so both the sequential and parallel drivers run the
 * same code -- the alternative was two copies of 150 lines of transcribed
 * number theory, which would drift.
 *
 * The random stream is a parameter rather than the file-scope rstate.
 * That is what makes the function callable from several threads at once:
 * gmp_randstate_t is the only GMP state the prover mutates rather than
 * merely reads.  Passing the global rstate reproduces the historical
 * behaviour exactly, which is what the sequential driver does.
 */
static SweepOutcome try_discriminant(int64_t D, const mpz_class &Ni,
                                     gmp_randstate_t st, mpz_class *root,
                                     SweepResult *r) {
  mpz_class a, b, d, m, q, t, u, v, x, y, g, j, w;
  struct point P, P1, P2;
  bool found = false, found2 = false;
  int32_t value1, value2;
  int64_t k = 0, points_tried = 0, rootSize = 0;

  r->D = D;

        found = false;
        found2 = false;

        if (verbose) {
          gmpecpp_detail::log() << D << '\r';
        }

        //  gmpecpp_detail::log() << "Step 3\n";

        if (JACOBI(D + Ni, Ni) != 1)
          return SWEEP_REJECTED;
        if (!modified_Cornacchia(D, Ni, &u, &v))
          return SWEEP_REJECTED;

        //  gmpecpp_detail::log() << "Step 4\n";

        t = sqrt(sqrt(Ni)) + 1;
        t = t * t;

        if (check_for_factor(&q, m = Ni + 1 + u, t))
          found = true;
        else if (check_for_factor(&q, m = Ni + 1 - u, t))
          found = true;
        else if (D == -4) {
          if (check_for_factor(&q, m = Ni + 1 + 2 * v, t))
            found = true;
          else if (check_for_factor(&q, m = Ni + 1 - 2 * v, t))
            found = true;
        } else if (D == -3) {
          if (check_for_factor(&q, m = Ni + 1 + (u + 3 * v) / 2, t))
            found = true;
          else if (check_for_factor(&q, m = Ni + 1 - (u + 3 * v) / 2, t))
            found = true;
          else if (check_for_factor(&q, m = Ni + 1 + (u - 3 * v) / 2, t))
            found = true;
          else if (check_for_factor(&q, m = Ni + 1 - (u - 3 * v) / 2, t))
            found = true;
        }

        if (!found)
          return SWEEP_REJECTED;

        //  gmpecpp_detail::log() << "Step 6\n";

        rootSize = 0;

        for (int32_t type = 0; type <= 2; type++) {

          if (!find_curve(type, &a, &b, D, Ni, root, &rootSize))
            continue;

          for (int64_t roots_tried = 0; roots_tried < rootSize; roots_tried++) {
            if (type == 0) {
              a = modpos(a, Ni);
              b = modpos(b, Ni);
            }

            else if (hilbert && type == 1) {
              j = modpos(root[roots_tried], Ni);

              if (!Quiet) {
                gmpecpp_detail::log() << "j = " << j << "\n";
              }

              mpz_class c;
              c = modpos(j * inverse(j - 1728, Ni), Ni);
              a = modpos(-3 * c, Ni);
              b = modpos(2 * c, Ni);
            }

            else if (weber && type == 2) {
              w = modpos(root[roots_tried], Ni);

              if (!Quiet) {
                gmpecpp_detail::log() << "u = " << w << "\n";
              }

              if (D % 4 == 0)
                j = Vegas(w, Ni, D / 4);
              else
                j = Vegas(w, Ni, D);

              if (!Quiet) {
                gmpecpp_detail::log() << "j = " << j << "\n";
              }

              mpz_class c;
              c = modpos(j * inverse(j - 1728, Ni), Ni);
              a = modpos(-3 * c, Ni);
              b = modpos(2 * c, Ni);
            }

            //  gmpecpp_detail::log() << "Step 7\n";

            do {
              do
                g = modpos(rand2_r(st), Ni);
              while (g == 0);
              if (JACOBI(g, Ni) != -1)
                continue;
              if (D == -3) // Cohen
                           //							if
                           //(Ni % 3 == 1) // Studholme
                if (exp_mod(g, (Ni - 1) / 3, Ni) == 1)
                  continue;
              break;
            } while (true);

            //  gmpecpp_detail::log() << "Step 8\n";

            points_tried = 0;

            do {

              do {
                do {
                  do
                    x = modpos(rand2_r(st), Ni);
                  while (x == 0);
                  y = modpos(((x * x) % Ni * x) % Ni + a * x + b, Ni);
                } while (JACOBI(y, Ni) == -1);
                y = square_root_mod(y, Ni);
              } while (y == 0);

              //  gmpecpp_detail::log() << "Step 9\n";

              P.x = x, P.y = y;
              points_tried++;
              k = 0;

              do {
                //  gmpecpp_detail::log() << "Step 12\n";

                value2 = multiply(a, m / q, Ni, P, &P2, &d);

                if (value2 == 1) {
                  r->factor = mpz_class(d).get_str();
                  r->tag = 3;
                  r->outcome = SWEEP_COMPOSITE;
                  return SWEEP_COMPOSITE;
                }

                value1 = multiply(a, q, Ni, P2, &P1, &d);

                if (value1 == 1) {
                  r->factor = mpz_class(d).get_str();
                  r->tag = 2;
                  r->outcome = SWEEP_COMPOSITE;
                  return SWEEP_COMPOSITE;
                }

                if ((value1 == -1) && (value2 == 0)) {
                  //  gmpecpp_detail::log() << "Step 13\n";

                  found2 = true;
                  break;
                }

                //  gmpecpp_detail::log() << "Step 10\n";

                ++k;

                if (D == -3) {
                  if (k >= 6)
                    break;
                  b *= g;
                } else if (D == -4) {
                  if (k >= 4)
                    break;
                  a *= g;
                } else {
                  if (k >= 2)
                    break;
                  a *= (g * g);
                  b *= (g * g * g);
                }

                a = modpos(a, Ni);
                b = modpos(b, Ni);

              } while (true);

            } while (!found2 && points_tried < 100);

            if (found2)
              break;
          }

          if (found2)
            break;
        }


  if (!found2)
    return SWEEP_REJECTED;

  r->outcome = SWEEP_FOUND;
  r->a = a; r->b = b; r->m = m; r->q = q;
  r->P = P; r->P1 = P1; r->P2 = P2;
  return SWEEP_FOUND;
}


/*
 * Parallel discriminant sweep (POSIX threads).
 *
 * Each D is an independent trial and the sweep wants the first that
 * works, so this is speculative parallelism: run several D at once and
 * keep the best.  It covers essentially the whole proof -- the factoring
 * stage and the class polynomial construction both sit under it.
 *
 * Two design points make it worth having rather than merely faster.
 *
 * Determinism.  Each attempt seeds its own random stream from the
 * discriminant, so try_discriminant(D) is a pure function of (D, Ni,
 * seed).  Results therefore do not depend on how the work was scheduled,
 * and a run at 16 threads reproduces a run at 2.
 *
 * The smallest D still wins.  Workers draw monotonically increasing
 * indices, and a success at index k stops new work beyond k while
 * in-flight attempts below k are allowed to finish.  Taking the smallest
 * successful index means the sweep chooses the same discriminant the
 * sequential version would -- which matters, because a larger |D| means a
 * larger class number and a more expensive proof step.  Racing to "any
 * success" would often pick a worse curve and give back the speedup.
 */
struct SweepShared {
  const mpz_class *Ni;
  int64_t limit;          // stop when index reaches this
  unsigned long seed;
  int nthreads;

  std::atomic<int64_t> next_n;
  std::atomic<int64_t> best_n;   // smallest index that succeeded so far
  pthread_mutex_t mtx;
  SweepResult best;
  int have;
};

struct SweepWorker {
  SweepShared *sh;
  int id;
};

static void *sweep_thread(void *arg) {
  SweepWorker *w = (SweepWorker *)arg;
  SweepShared *sh = w->sh;

  gmp_randstate_t st;
  gmp_randinit_default(st);
  tls_rand = st;   // everything below this point draws from it

  // FindRootsModuloAPrime reads the incoming contents of this array, so
  // it is state that persists between attempts rather than scratch. The
  // sequential path keeps one for the whole proof; each worker keeps its
  // own, which is what lets attempts run concurrently.
  std::vector<mpz_class> root(POLY_SIZE);

  for (;;) {
    int64_t n = sh->next_n.fetch_add(1, std::memory_order_relaxed);
    if (n >= sh->limit)
      break;
    if (n >= sh->best_n.load(std::memory_order_relaxed)) // smaller one won
      break;

    int64_t D = -n;
    if (D % 4 != 0 && D % 4 != -3 && D % 4 != 1)
      continue;

    // Per-discriminant seeding: this is what makes the sweep
    // reproducible no matter how attempts are distributed.
    gmp_randseed_ui(st, sh->seed + 0x9e3779b97f4a7c15ULL * (unsigned long)n);

    SweepResult r;
    SweepOutcome o = try_discriminant(D, *sh->Ni, st, &root[0], &r);
    if (o == SWEEP_REJECTED)
      continue;

    pthread_mutex_lock(&sh->mtx);
    if (!sh->have || n < sh->best_n.load(std::memory_order_relaxed)) {
      sh->best = r;
      sh->best_n.store(n, std::memory_order_relaxed);
      sh->have = 1;
    }
    pthread_mutex_unlock(&sh->mtx);
  }

  tls_rand = 0;
  gmp_randclear(st);
  return NULL;
}

/*
 * Returns 1 if some discriminant produced a result (success or a
 * compositeness proof), 0 if the whole range was exhausted.
 */
/*
 * Most inputs succeed on one of the first few discriminants, and for
 * those the sweep is over before thread creation would have paid for
 * itself -- measured at 0.66s sequential against 2.29s with sixteen
 * workers, a 3.5x loss. So probe the smallest discriminants on this
 * thread first and only fan out if they all fail.
 *
 * This costs nothing in the parallel case (those attempts had to happen
 * anyway, and they are the ones whose results would be preferred) and it
 * keeps -S from ever being much worse than -S 1.
 */
static const int64_t SWEEP_PROBE = 8;

static int sweep_parallel(const mpz_class &Ni, int64_t limit, int nthreads,
                          unsigned long seed, SweepResult *out) {
  {
    gmp_randstate_t st;
    gmp_randinit_default(st);
    std::vector<mpz_class> root(POLY_SIZE);
    tls_rand = st;
    for (int64_t n = 1; n < SWEEP_PROBE && n < limit; n++) {
      int64_t D = -n;
      if (D % 4 != 0 && D % 4 != -3 && D % 4 != 1)
        continue;
      gmp_randseed_ui(st, seed + 0x9e3779b97f4a7c15ULL * (unsigned long)n);
      SweepResult r;
      if (try_discriminant(D, Ni, st, &root[0], &r) != SWEEP_REJECTED) {
        tls_rand = 0;
        gmp_randclear(st);
        *out = r;
        return 1;
      }
    }
    tls_rand = 0;
    gmp_randclear(st);
  }

  SweepShared sh;
  sh.Ni = &Ni;
  sh.limit = limit;
  sh.seed = seed;
  sh.nthreads = nthreads;
  sh.next_n.store(SWEEP_PROBE, std::memory_order_relaxed); // probe did 1..7
  sh.best_n.store(limit, std::memory_order_relaxed);
  sh.have = 0;
  pthread_mutex_init(&sh.mtx, NULL);

  // find_curve, Hilbert and Recurse together put roughly a megabyte of
  // POLY_SIZE arrays on the stack, and Recurse is recursive; the default
  // thread stack is not enough headroom.
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 64u * 1024u * 1024u);

  std::vector<pthread_t> th(nthreads);
  std::vector<SweepWorker> w(nthreads);
  int started = 0;
  for (int i = 0; i < nthreads; i++) {
    w[i].sh = &sh;
    w[i].id = i;
    if (pthread_create(&th[i], &attr, sweep_thread, &w[i]) == 0)
      started++;
    else
      break;
  }
  for (int i = 0; i < started; i++)
    pthread_join(th[i], NULL);
  pthread_attr_destroy(&attr);
  pthread_mutex_destroy(&sh.mtx);

  if (sh.have)
    *out = sh.best;
  return sh.have;
}

} // namespace gmpecpp_detail

int32_t Atkin(mpz_class N) {
  int32_t value1, value2;
  int64_t p;
  mpz_class Ni = N, a, b, d, i = 0, m, q, t, x, y;
  struct point P, P1, P2;
  int64_t n = 0, k = 0;
  int64_t D;
  bool found = false, found2 = false;
  mpz_class u, v;
  mpz_class g;
  int64_t points_tried = 0;
  mpz_class root[POLY_SIZE];
  int64_t rootSize = 0;
  mpz_class j, w;

  do {

    //  gmpecpp_detail::log() << "Step 2\n";

    if (Ni <= SIEVE_LIMIT) {
      p = 2;
      while (p <= sqrt(Ni)) {
        if (Ni % p == 0) {
          gmpecpp_detail::state().factor = mpz_class(p).get_str();
          gmpecpp_detail::log() << "1 factor = " << p << "\n";
          return 2;
        }
        p++;
      }
      return 0;
    }

    if (!Rabin_Miller(Ni))
      return 2;

    while (Bmax <= 2000000000 && Dmax <= 312500) {

      // The original cleared these at the top of every discriminant
      // attempt, inside the loop body. That body is now a function with
      // its own locals, so Atkin's copies have to be cleared here --
      // otherwise a sweep that finds nothing inherits the previous
      // downrun step's success and proceeds with its stale curve.
      found = false;
      found2 = false;


      if (!Quiet) {
        gmpecpp_detail::log() << "Bmax = " << Bmax << "\n";
        gmpecpp_detail::log() << "Dmax = " << Dmax << "\n";
      }

      n = 1;

      if (gmpecpp_detail::g_sweep_threads > 1) {
        // Speculative parallel sweep. Deterministic, and it settles on the
        // same discriminant the sequential walk below would reach.
        gmpecpp_detail::SweepResult sr;
        if (gmpecpp_detail::sweep_parallel(Ni, Dmax, gmpecpp_detail::g_sweep_threads,
                                           gmpecpp_detail::g_seed, &sr)) {
          if (sr.outcome == gmpecpp_detail::SWEEP_COMPOSITE) {
            gmpecpp_detail::state().factor = sr.factor;
            gmpecpp_detail::log()
                << sr.tag << " factor = " << sr.factor << "\n";
            return 2;
          }
          D = sr.D;
          a = sr.a;
          b = sr.b;
          m = sr.m;
          q = sr.q;
          P = sr.P;
          P1 = sr.P1;
          P2 = sr.P2;
          found2 = true;
        }
      } else
      while ((D = -n++) > -Dmax) {

        if (D % 4 != 0 && D % 4 != -3 && D % 4 != 1)
          continue;

        gmpecpp_detail::SweepResult sr;
        gmpecpp_detail::SweepOutcome o =
            gmpecpp_detail::try_discriminant(D, Ni, rstate, root, &sr);

        if (o == gmpecpp_detail::SWEEP_COMPOSITE) {
          gmpecpp_detail::state().factor = sr.factor;
          gmpecpp_detail::log() << sr.tag << " factor = " << sr.factor << "\n";
          return 2;
        }
        if (o == gmpecpp_detail::SWEEP_FOUND) {
          a = sr.a;
          b = sr.b;
          m = sr.m;
          q = sr.q;
          P = sr.P;
          P1 = sr.P1;
          P2 = sr.P2;
          found2 = true;
          break;
        }
      }

      if (found2) {
        if (!staticBmax)
          Bmax = BMAX;
        if (!staticDmax)
          Dmax = DMAX;
        break;
      } else {
        if (!staticBmax)
          Bmax *= 10;
        if (!staticDmax)
          Dmax *= 5;
      }
    }

    if (Dmax > 312500) {
      if (!Quiet) {
        gmpecpp_detail::log() << "ProvePrime: ran out of discriminants\n";
      }
      gmpecpp_detail::state().give_up = gmpecpp_detail::GU_DISCRIMINANTS;
      return 2;
    }
    if (Bmax > 2000000000) {
      if (!Quiet) {
        gmpecpp_detail::log() << "ProvePrime: exceeded maximum factoring bounds\n";
      }
      gmpecpp_detail::state().give_up = gmpecpp_detail::GU_BOUNDS;
      return 2;
    }

    {
      gmpecpp_detail::Recorded rec;
      rec.n = Ni.get_str();
      rec.a = a.get_str();
      rec.b = b.get_str();
      rec.m = m.get_str();
      rec.q = q.get_str();
      rec.px = P.x.get_str();
      rec.py = P.y.get_str();
      rec.p1x = P1.x.get_str();
      rec.p1y = P1.y.get_str();
      rec.p2x = P2.x.get_str();
      rec.p2y = P2.y.get_str();
      rec.d = (long)D;
      gmpecpp_detail::state().steps.push_back(rec);
    }

    gmpecpp_detail::log() << "N[" << i << "] = " << Ni << "\n";
    gmpecpp_detail::log() << "a = " << a << "\n";
    gmpecpp_detail::log() << "b = " << b << "\n";
    gmpecpp_detail::log() << "m = " << m << "\n";
    gmpecpp_detail::log() << "q = " << q << "\n";
    gmpecpp_detail::log() << "P = (" << P.x << ", " << P.y << ")\n";
    gmpecpp_detail::log() << "P1 = (" << P1.x << ", " << P1.y << ")\n";
    gmpecpp_detail::log() << "P2 = (" << P2.x << ", " << P2.y << ")\n";


    i++;
    Ni = q;

  } while (!one);

  return one ? 1 : 0;
}

// The name of this program.


// Extracted from the original main(): the file scope mpf_class
// constants are only meaningful once their precision is set and their
// values computed.  PI comes from Hutton's formula via atanf2 itself.
void init_constants_uncached() {
  PIf.set_prec(precision);
  PIf = 3;
  Ef.set_prec(precision);
  Ef = 2;
  NATLOGONEPOINTNINEf.set_prec(precision);
  NATLOGONEPOINTNINEf = 1;

  ZERO.set_prec(precision);
  ZERO = 0;
  ONE.set_prec(precision);
  ONE = 1;
  TWO.set_prec(precision);
  TWO = 2;
  THREE.set_prec(precision);
  THREE = 3;
  FOUR.set_prec(precision);
  FOUR = 4;
  FIVE.set_prec(precision);
  FIVE = 5;
  SIX.set_prec(precision);
  SIX = 6;
  EIGHT.set_prec(precision);
  EIGHT = 8;
  TEN.set_prec(precision);
  TEN = 10;
  TWELVE.set_prec(precision);
  TWELVE = 12;
  EIGHTEEN.set_prec(precision);
  EIGHTEEN = 18;
  NINETEEN.set_prec(precision);
  NINETEEN = 19;
  TWENTYFOUR.set_prec(precision);
  TWENTYFOUR = 24;
  FIFTYSEVEN.set_prec(precision);
  FIFTYSEVEN = 57;
  TWOHUNDREDTHIRTYNINE.set_prec(precision);
  TWOHUNDREDTHIRTYNINE = 239;
  TWOHUNDREDFIFTYSIX.set_prec(precision);
  TWOHUNDREDFIFTYSIX = 256;

  HALF.set_prec(precision);
  HALF = ONE / TWO;
  ONEPOINTNINE.set_prec(precision);
  ONEPOINTNINE = NINETEEN / TEN;

  PIf = FOUR * (atanf2(ONE / TWO) + atanf2(ONE / THREE)); // Hutton
  // PIf = FOUR *
  // (TWELVE * atanf2(ONE/EIGHTEEN)
  // + EIGHT * atanf2(ONE/FIFTYSEVEN)
  // - FIVE * atanf2(ONE/TWOHUNDREDTHIRTYNINE));	// Gauss

  Ef = expf2(ONE);

  NATLOGONEPOINTNINEf = logf2(ONEPOINTNINE);
}

// ===================================================================
// Public API implementation
// ===================================================================

#include <cstring>
#include <mutex>
#include <unistd.h>
#include <new>

namespace gmpecpp_detail {

State &state() {
  static State st;
  return st;
}

LogSink &log() {
  static LogSink sink;
  return sink;
}

double wall_seconds() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

// The prover keeps its configuration in file scope globals, so calls have
// to be serialised.  Documented in the header; the alternative is
// threading a context through 2600 lines of transcribed algorithms.
std::mutex &big_lock() {
  static std::mutex m;
  return m;
}

// ECM backend hooks.  The CUDA implementation lives in ecm_cuda.cu and is
// linked in only when the library is built with CUDA support.
bool cuda_available();
const char *cuda_device_name();
bool cuda_ecm_factor(mpz_srcptr n, unsigned long b1, unsigned int curves,
                     unsigned long seed, mpz_ptr factor);
int cuda_ecm_run(mpz_srcptr n, unsigned long b1, unsigned long b2,
                 unsigned int curves, unsigned long seed, int (*stop)(void *),
                 void *stop_arg, mpz_ptr factor);
unsigned int cuda_ecm_default_curves(size_t bits);

} // namespace gmpecpp_detail

struct gmpecpp_config {
  unsigned long precision;
  unsigned long error_shift;
  unsigned long seed;
  unsigned long bmax;
  unsigned long dmax;
  int have_bmax, have_dmax;
  int hilbert, weber, one_step;
  int verbosity;
  gmpecpp_backend_t backend;
  unsigned int cuda_curves;
  unsigned int cuda_min_bits;
  unsigned int threads;
  unsigned int sweep_threads;
  gmpecpp_log_fn log_fn;
  void *log_user;

  gmpecpp_config()
      : precision(0), error_shift(0), seed(0), bmax(0), dmax(0),
        have_bmax(0), have_dmax(0), hilbert(0), weber(0), one_step(0),
        verbosity(0), backend(GMPECPP_BACKEND_CPU), cuda_curves(0),
        cuda_min_bits(192), threads(1), sweep_threads(1),
        log_fn(NULL), log_user(NULL) {}
};

struct gmpecpp_certificate {
  std::vector<gmpecpp_detail::Recorded> steps;
  std::string factor;
  gmpecpp_backend_t backend;
  unsigned long seed;
  double elapsed;
  double ecm_elapsed;
};

extern "C" {

gmpecpp_config_t *gmpecpp_config_new(void) {
  return new (std::nothrow) gmpecpp_config();
}
void gmpecpp_config_free(gmpecpp_config_t *c) { delete c; }

void gmpecpp_config_set_precision(gmpecpp_config_t *c, unsigned long v) {
  if (c) c->precision = v;
}
void gmpecpp_config_set_error_shift(gmpecpp_config_t *c, unsigned long v) {
  if (c) c->error_shift = v;
}
void gmpecpp_config_set_seed(gmpecpp_config_t *c, unsigned long v) {
  if (c) c->seed = v;
}
void gmpecpp_config_set_bmax(gmpecpp_config_t *c, unsigned long v) {
  if (c) { c->bmax = v; c->have_bmax = 1; }
}
void gmpecpp_config_set_dmax(gmpecpp_config_t *c, unsigned long v) {
  if (c) { c->dmax = v; c->have_dmax = 1; }
}
void gmpecpp_config_set_hilbert(gmpecpp_config_t *c, int v) {
  if (c) c->hilbert = v ? 1 : 0;
}
void gmpecpp_config_set_weber(gmpecpp_config_t *c, int v) {
  if (c) c->weber = v ? 1 : 0;
}
void gmpecpp_config_set_one_step(gmpecpp_config_t *c, int v) {
  if (c) c->one_step = v ? 1 : 0;
}
void gmpecpp_config_set_backend(gmpecpp_config_t *c, gmpecpp_backend_t b) {
  if (c) c->backend = b;
}
void gmpecpp_config_set_cuda_curves(gmpecpp_config_t *c, unsigned int n) {
  if (c) c->cuda_curves = n;
}
void gmpecpp_config_set_cuda_min_bits(gmpecpp_config_t *c, unsigned int n) {
  if (c) c->cuda_min_bits = n;
}
void gmpecpp_config_set_threads(gmpecpp_config_t *c, unsigned int n) {
  if (c) c->threads = n;
}
void gmpecpp_config_set_sweep_threads(gmpecpp_config_t *c, unsigned int n) {
  if (c) c->sweep_threads = n;
}
void gmpecpp_config_set_log(gmpecpp_config_t *c, gmpecpp_log_fn f, void *u) {
  if (c) { c->log_fn = f; c->log_user = u; }
}
void gmpecpp_config_set_verbosity(gmpecpp_config_t *c, int v) {
  if (c) c->verbosity = v;
}

} // extern "C"

namespace gmpecpp_detail {

// Selected backend for the current proof, read by check_for_factor().
gmpecpp_backend_t g_backend = GMPECPP_BACKEND_CPU;
unsigned int g_cuda_curves = 0;
unsigned int g_cuda_min_bits = 192;
int g_threads = 1;
int g_sweep_threads = 1;
unsigned long g_seed = 0;


// Constants are expensive at high precision (a few tenths of a second at
// the default 10000 bits) and depend only on `precision`, so cache them.
int64_t g_constants_precision = -1;
void init_constants_cached() {
  if (g_constants_precision == ::precision)
    return;
  mpf_set_default_prec((mp_bitcnt_t)::precision);
  init_constants_uncached();
  g_constants_precision = ::precision;
}

// True once the preamble has been emitted for the pending proof, so that
// gmpecpp_prove() does not repeat what gmpecpp_preamble() already said.
bool g_preamble_done = false;

void emit_preamble(unsigned long seed) {
  if (g_preamble_done)
    return;
  g_preamble_done = true;
  if (!::Quiet) {
    log() << "random seed = " << seed << "\n";
    log() << "error_shift = " << ::error_shift << "\n";
    log() << "precision = " << ::precision << "\n";
  }

  init_constants_cached();

  if (!::Quiet && !::quiet) {
    log() << "PI = ";
    print_mpf(PIf);
    log() << "\n";
    log() << "E = ";
    print_mpf(Ef);
    log() << "\n";
    log() << "NATLOGONEPOINTNINE = ";
    print_mpf(NATLOGONEPOINTNINEf);
    log() << "\n";
  }

  // -v used to dump the first 20 class polynomials before reading input,
  // as a way to exercise the floating point layer on its own.
  if (::verbose) {
    int64_t Dv, dCP;
    std::vector<mpz_class> CP(POLY_SIZE);
    if (::hilbert) {
      log() << "\nFirst few Hilbert Class Polynomials :\n";
      for (int32_t k = 1; k <= 20; k++) {
        Dv = -k;
        if (Dv % 4 != 0 && Dv % 4 != -3 && Dv % 4 != 1)
          continue;
        Hilbert(Dv, &CP[0], &dCP);
        log() << "D = " << Dv << ", dT = " << dCP << ", T = ";
        zpoly_print(dCP, &CP[0]);
      }
    }
    if (::weber) {
      log() << "\nFirst few Weber Class Polynomials :\n";
      for (int32_t k = 1; k <= 20; k++) {
        Dv = -k;
        if (Dv % 4 != 0 && Dv % 4 != -3 && Dv % 4 != 1)
          continue;
        if (Dv % 32 != 0) {
          Weber(Dv, &CP[0], &dCP);
          log() << "D = " << Dv << ", dW = " << dCP << ", W = ";
        }
        zpoly_print(dCP, &CP[0]);
      }
    }
    log() << "\n";
  }

}

// Copies the configuration onto the prover's file scope globals and
// returns the seed that was chosen.
unsigned long apply_config(const gmpecpp_config &c) {
  State &st = state();
  st.log_fn = c.log_fn;
  st.log_user = c.log_user;
  st.verbosity = c.verbosity;

  ::precision = (int64_t)(c.precision ? c.precision : PRECISION);
  ::error_shift = (int64_t)(c.error_shift ? c.error_shift : ERROR_SHIFT);
  ::one = c.one_step;
  ::hilbert = c.hilbert;
  ::weber = c.weber;
  if (!::hilbert && !::weber) {
    ::hilbert = 1;
    ::weber = 1;
  }
  ::staticBmax = c.have_bmax != 0;
  ::staticDmax = c.have_dmax != 0;
  ::Bmax = (int64_t)(c.have_bmax ? c.bmax : BMAX);
  ::Dmax = (int64_t)(c.have_dmax ? c.dmax : DMAX);
  {
    int t = (int)c.threads;
    if (t == 0) {
      long n = sysconf(_SC_NPROCESSORS_ONLN);
      t = (n > 0) ? (int)n : 1;
    }
    g_threads = t < 1 ? 1 : t;

    int s = (int)c.sweep_threads;
    if (s == 0) {
      long n = sysconf(_SC_NPROCESSORS_ONLN);
      s = (n > 0) ? (int)n : 1;
    }
    g_sweep_threads = s < 1 ? 1 : s;
    // Sweep workers each run their own ECM; letting both fan out would
    // oversubscribe by the product of the two.
    if (g_sweep_threads > 1)
      g_threads = 1;
  }
  ::Quiet = (c.verbosity <= 0) ? 1 : 0;
  ::quiet = (c.verbosity <= 1) ? 1 : 0;
  ::verbose = (c.verbosity >= 3) ? 1 : 0;

  unsigned long seed = c.seed;
  if (!seed) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    seed = (unsigned long)(tv.tv_sec + tv.tv_usec);
  }
  return seed;
}

static gmpecpp_status_t run(const gmpecpp_config *cfg, mpz_srcptr n,
                            gmpecpp_certificate_t **out) {
  static const gmpecpp_config defaults;
  const gmpecpp_config &c = cfg ? *cfg : defaults;

  std::lock_guard<std::mutex> guard(big_lock());

  State &st = state();
  st.log_fn = c.log_fn;
  st.log_user = c.log_user;
  st.verbosity = c.verbosity;
  st.pending.clear();
  st.steps.clear();
  st.factor.clear();
  st.give_up = GU_NONE;
  st.ecm_seconds = 0.0;

  // Map the config onto the prover's globals.
  unsigned long seed = apply_config(c);
  g_seed = seed;
  g_backend = c.backend;
  g_cuda_curves = c.cuda_curves;
  g_cuda_min_bits = c.cuda_min_bits;
  if (g_backend == GMPECPP_BACKEND_CUDA && !cuda_available())
    g_backend = GMPECPP_BACKEND_CPU;

  gmp_randinit_default(rstate);
  gmp_randseed_ui(rstate, seed);

  emit_preamble(seed);

  double t0 = wall_seconds();
  gmpecpp_status_t status;
  int32_t rc = 2;
  try {
    rc = Atkin(mpz_class(mpz_srcptr(n)));
    if (rc == 0)
      status = GMPECPP_PROVEN_PRIME;
    else if (rc == 1)
      status = GMPECPP_PROBABLY_PRIME;
    else if (st.give_up == GU_DISCRIMINANTS)
      status = GMPECPP_OUT_OF_DISCRIMINANTS;
    else if (st.give_up == GU_BOUNDS)
      status = GMPECPP_FACTORING_BOUND;
    else
      status = GMPECPP_COMPOSITE;
  } catch (const CompositeFound &cf) {
    st.factor = cf.factor;
    status = GMPECPP_COMPOSITE;
  } catch (const std::bad_alloc &) {
    status = GMPECPP_ERR_NO_MEMORY;
  } catch (...) {
    status = GMPECPP_ERR_INTERNAL;
  }
  double elapsed = wall_seconds() - t0;

  // Flush any fragment the prover left without a trailing newline.
  if (st.log_fn && !st.pending.empty()) {
    st.log_fn(st.pending.c_str(), st.log_user);
    st.pending.clear();
  }
  gmp_randclear(rstate);

  if (out) {
    gmpecpp_certificate_t *cert = new (std::nothrow) gmpecpp_certificate();
    if (!cert)
      return GMPECPP_ERR_NO_MEMORY;
    cert->steps = st.steps;
    cert->factor = st.factor;
    cert->backend = g_backend;
    cert->seed = seed;
    cert->elapsed = elapsed;
    cert->ecm_elapsed = st.ecm_seconds;
    *out = cert;
  }
  st.log_fn = NULL;
  st.log_user = NULL;
  g_preamble_done = false;
  return status;
}

} // namespace gmpecpp_detail

extern "C" {

void gmpecpp_preamble(const gmpecpp_config_t *cfg) {
  static const gmpecpp_config defaults;
  const gmpecpp_config &c = cfg ? *cfg : defaults;
  std::lock_guard<std::mutex> guard(gmpecpp_detail::big_lock());
  gmpecpp_detail::State &st = gmpecpp_detail::state();
  st.pending.clear();
  unsigned long seed = gmpecpp_detail::apply_config(c);
  gmpecpp_detail::g_seed = seed;
  gmpecpp_detail::emit_preamble(seed);
  if (st.log_fn && !st.pending.empty()) {
    st.log_fn(st.pending.c_str(), st.log_user);
    st.pending.clear();
  }
}

gmpecpp_status_t gmpecpp_prove(const gmpecpp_config_t *cfg, const char *n,
                               gmpecpp_certificate_t **cert) {
  if (cert)
    *cert = NULL;
  if (!n)
    return GMPECPP_ERR_INVALID_INPUT;
  mpz_t v;
  mpz_init(v);
  if (mpz_set_str(v, n, 10) != 0 || mpz_cmp_ui(v, 2) < 0) {
    mpz_clear(v);
    return GMPECPP_ERR_INVALID_INPUT;
  }
  gmpecpp_status_t s = gmpecpp_detail::run(cfg, v, cert);
  mpz_clear(v);
  return s;
}

gmpecpp_status_t gmpecpp_prove_mpz(const gmpecpp_config_t *cfg, mpz_srcptr n,
                                   gmpecpp_certificate_t **cert) {
  if (cert)
    *cert = NULL;
  if (!n || mpz_cmp_ui(n, 2) < 0)
    return GMPECPP_ERR_INVALID_INPUT;
  return gmpecpp_detail::run(cfg, n, cert);
}

gmpecpp_status_t gmpecpp_ecm_factor(const gmpecpp_config_t *cfg,
                                    const char *n, unsigned long b1,
                                    char **factor) {
  if (factor)
    *factor = NULL;
  if (!n)
    return GMPECPP_ERR_INVALID_INPUT;

  static const gmpecpp_config defaults;
  const gmpecpp_config &c = cfg ? *cfg : defaults;
  std::lock_guard<std::mutex> guard(gmpecpp_detail::big_lock());

  mpz_t v;
  mpz_init(v);
  if (mpz_set_str(v, n, 10) != 0 || mpz_cmp_ui(v, 2) < 0) {
    mpz_clear(v);
    return GMPECPP_ERR_INVALID_INPUT;
  }

  unsigned long seed = gmpecpp_detail::apply_config(c);
  gmpecpp_detail::g_seed = seed;
  gmpecpp_detail::g_backend = c.backend;
  gmpecpp_detail::g_cuda_curves = c.cuda_curves;
  gmpecpp_detail::g_cuda_min_bits = c.cuda_min_bits;
  if (gmpecpp_detail::g_backend == GMPECPP_BACKEND_CUDA &&
      !gmpecpp_detail::cuda_available())
    gmpecpp_detail::g_backend = GMPECPP_BACKEND_CPU;

  gmp_randinit_default(rstate);
  gmp_randseed_ui(rstate, seed);
  ::Bmax = (int64_t)b1;

  gmpecpp_status_t status = GMPECPP_PROBABLY_PRIME;
  try {
    mpz_class N = mpz_class(mpz_srcptr(v));
    mpz_class g;
    if (ecm_find_factor(&N, &g, mpz_class((unsigned long)b1))) {
      std::string s = g.get_str();
      if (factor) {
        *factor = (char *)malloc(s.size() + 1);
        if (*factor)
          memcpy(*factor, s.c_str(), s.size() + 1);
      }
      status = GMPECPP_COMPOSITE;
    }
  } catch (const gmpecpp_detail::CompositeFound &cf) {
    if (factor) {
      *factor = (char *)malloc(cf.factor.size() + 1);
      if (*factor)
        memcpy(*factor, cf.factor.c_str(), cf.factor.size() + 1);
    }
    status = GMPECPP_COMPOSITE;
  } catch (...) {
    status = GMPECPP_ERR_INTERNAL;
  }
  gmp_randclear(rstate);
  mpz_clear(v);
  return status;
}

gmpecpp_status_t gmpecpp_cuda_ecm(const char *n, unsigned long b1,
                                  unsigned long b2, unsigned int curves,
                                  unsigned long seed, int (*stop)(void *),
                                  void *stop_arg, char **factor) {
  if (factor)
    *factor = NULL;
  if (!n)
    return GMPECPP_ERR_INVALID_INPUT;
  mpz_t v, f;
  mpz_init(v);
  if (mpz_set_str(v, n, 10) != 0 || mpz_cmp_ui(v, 2) < 0) {
    mpz_clear(v);
    return GMPECPP_ERR_INVALID_INPUT;
  }
  mpz_init(f);
  int r = gmpecpp_detail::cuda_ecm_run(v, b1, b2, curves, seed ? seed : 1,
                                       stop, stop_arg, f);
  gmpecpp_status_t status;
  if (r == 1) {
    status = GMPECPP_COMPOSITE;
    if (factor) {
      char *str = mpz_get_str(NULL, 10, f);
      size_t len = strlen(str) + 1;
      *factor = (char *)malloc(len);
      if (*factor)
        memcpy(*factor, str, len);
      void (*freefunc)(void *, size_t);
      mp_get_memory_functions(NULL, NULL, &freefunc);
      freefunc(str, len);
    }
  } else if (r == 0) {
    status = GMPECPP_PROBABLY_PRIME;
  } else if (r == -2) {
    status = GMPECPP_ERR_INVALID_INPUT;
  } else {
    status = GMPECPP_ERR_INTERNAL;
  }
  mpz_clear(f);
  mpz_clear(v);
  return status;
}

unsigned int gmpecpp_cuda_ecm_curves(const char *n) {
  if (!n)
    return 0;
  mpz_t v;
  mpz_init(v);
  unsigned int c = 0;
  if (mpz_set_str(v, n, 10) == 0 && mpz_cmp_ui(v, 2) >= 0)
    c = gmpecpp_detail::cuda_ecm_default_curves(mpz_sizeinbase(v, 2));
  mpz_clear(v);
  return c;
}

void gmpecpp_string_free(char *s) { free(s); }

void gmpecpp_certificate_free(gmpecpp_certificate_t *c) { delete c; }

size_t gmpecpp_certificate_num_steps(const gmpecpp_certificate_t *c) {
  return c ? c->steps.size() : 0;
}

int gmpecpp_certificate_get_step(const gmpecpp_certificate_t *c, size_t i,
                                 gmpecpp_step_t *out) {
  if (!c || !out || i >= c->steps.size())
    return -1;
  const gmpecpp_detail::Recorded &r = c->steps[i];
  out->n = r.n.c_str();
  out->a = r.a.c_str();
  out->b = r.b.c_str();
  out->m = r.m.c_str();
  out->q = r.q.c_str();
  out->px = r.px.c_str();
  out->py = r.py.c_str();
  out->p1x = r.p1x.c_str();
  out->p1y = r.p1y.c_str();
  out->p2x = r.p2x.c_str();
  out->p2y = r.p2y.c_str();
  out->discriminant = r.d;
  return 0;
}

const char *gmpecpp_certificate_factor(const gmpecpp_certificate_t *c) {
  return (c && !c->factor.empty()) ? c->factor.c_str() : NULL;
}
gmpecpp_backend_t gmpecpp_certificate_backend(const gmpecpp_certificate_t *c) {
  return c ? c->backend : GMPECPP_BACKEND_CPU;
}
unsigned long gmpecpp_certificate_seed(const gmpecpp_certificate_t *c) {
  return c ? c->seed : 0;
}
double gmpecpp_certificate_elapsed(const gmpecpp_certificate_t *c) {
  return c ? c->elapsed : 0.0;
}
double gmpecpp_certificate_ecm_elapsed(const gmpecpp_certificate_t *c) {
  return c ? c->ecm_elapsed : 0.0;
}

int gmpecpp_have_cuda(void) {
  return gmpecpp_detail::cuda_available() ? 1 : 0;
}
const char *gmpecpp_cuda_device_name(void) {
  return gmpecpp_detail::cuda_device_name();
}

const char *gmpecpp_version_string(void) {
  return "libgmpecpp " "1.0.0"
#ifdef GMPECPP_WITH_CUDA
         " (CUDA/CGBN)"
#else
         " (CPU only)"
#endif
      ;
}

const char *gmpecpp_status_string(gmpecpp_status_t s) {
  switch (s) {
  case GMPECPP_PROVEN_PRIME: return "proven prime";
  case GMPECPP_PROBABLY_PRIME: return "probably prime";
  case GMPECPP_COMPOSITE: return "composite";
  case GMPECPP_OUT_OF_DISCRIMINANTS: return "gave up: ran out of discriminants";
  case GMPECPP_FACTORING_BOUND: return "gave up: exceeded factoring bounds";
  case GMPECPP_ERR_INVALID_INPUT: return "invalid input";
  case GMPECPP_ERR_NO_MEMORY: return "out of memory";
  case GMPECPP_ERR_INTERNAL: return "internal error";
  }
  return "unknown";
}

} // extern "C"
