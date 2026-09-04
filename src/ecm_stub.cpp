/*
 * ecm_stub.cpp -- the ECM backend hooks for builds without CUDA.
 *
 * Linked in place of ecm_cuda.o when the library is built CPU-only, so
 * that gmpecpp_have_cuda() answers honestly rather than failing to link.
 */

#include <gmp.h>

namespace gmpecpp_detail {

bool cuda_available() { return false; }

const char *cuda_device_name() { return 0; }

bool cuda_ecm_factor(mpz_srcptr, unsigned long, unsigned int, unsigned long,
                     mpz_ptr) {
  return false;
}

} // namespace gmpecpp_detail
