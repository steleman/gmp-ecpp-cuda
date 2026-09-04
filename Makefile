# libgmpecpp -- elliptic curve primality prover (ECPP, Atkin-Morain).
# See README.md for usage.

# ---------------------------------------------------------------- #
# Toolchain                                                         #
# ---------------------------------------------------------------- #

CXX      = /usr/bin/g++
CC       = /usr/bin/gcc
CPPFLAGS = -D_GNU_SOURCE -D_XOPEN_SOURCE=700
CXXFLAGS = -O3 -Wall -Wextra -std=c++17
CFLAGS   = -O3 -Wall -std=c17

# Conventional order, C++ wrapper before the C library. Irrelevant for
# the shared libs (libgmpxx.so has a NEEDED entry on libgmp.so), but
# required if these are ever linked as static archives.
LDLIBS := -lgmpxx -lgmp

# ---------------------------------------------------------------- #
# CUDA / CGBN                                                       #
# ---------------------------------------------------------------- #
#
# CGBN accelerates the ECM factoring stage only; everything else in the
# prover is either sequential or arbitrary precision floating point,
# which CGBN (fixed width integer) cannot express. Set CUDA=0 to force a
# CPU-only build.

NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_ARCH ?= sm_89
# Change to this for Blackwell:
# CUDA_ARCH ?= sm_120
CGBN_INC  ?= /usr/local/include
CUDA_LIB  ?= /usr/local/cuda/lib64

CUDA ?= $(if $(wildcard $(NVCC)),1,0)

ifeq ($(CUDA),1)
  CUDA_CPPFLAGS := -DGMPECPP_WITH_CUDA
  CUDA_OBJ      := build/ecm_cuda.o
  CUDA_LDFLAGS  := -L$(CUDA_LIB) -lcudart
  CUDA_RPATH    := -Wl,-rpath,$(CUDA_LIB)
else
  CUDA_CPPFLAGS :=
  CUDA_OBJ      := build/ecm_stub.o
  CUDA_LDFLAGS  :=
  CUDA_RPATH    :=
endif

NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH)

# ---------------------------------------------------------------- #
# Targets                                                           #
# ---------------------------------------------------------------- #

# Versioned shared object with the usual two symlinks: the SONAME that
# gets recorded in dependent binaries, and the bare name the linker uses
# at build time.
LIB_VER := 1.0.0
LIB_SO  := libgmpecpp.so
SONAME  := $(LIB_SO).1
LIB     := $(LIB_SO).$(LIB_VER)
BIN     := gmp-ecpp
BENCH   := gmpecpp-bench
REACH   := ecm-reach
KERNEL  := kernel-latency
VERIFY  := gmpecpp-verify

SEED ?= 1

.PHONY: all clean check bench bench-all bench-reach bench-kernel \
        bench-precision bench-cuda bench-profile parity verify-all \
        doc format help cuda-info

all: $(LIB) $(BIN) $(BENCH) $(REACH) $(KERNEL) $(VERIFY)

build:
	@mkdir -p build

build/gmpecpp.o: src/gmpecpp.cpp include/gmpecpp.h | build
	$(CXX) $(CPPFLAGS) $(CUDA_CPPFLAGS) $(CXXFLAGS) -fPIC \
	  -Iinclude -c $< -o $@

build/ecm_stub.o: src/ecm_stub.cpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC \
	  -Iinclude -c $< -o $@

build/ecm_cuda.o: src/ecm_cuda.cu | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -I$(CGBN_INC) -Iinclude -c $< -o $@

$(LIB): build/gmpecpp.o $(CUDA_OBJ)
	$(CXX) -shared -Wl,-soname,$(SONAME) -o $@ $^ $(CUDA_LDFLAGS) $(LDLIBS)
	ln -sf $(LIB) $(SONAME)
	ln -sf $(SONAME) $(LIB_SO)

$(BIN): tools/gmp-ecpp.cpp $(LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Iinclude -o $@ $< \
	  -L. -lgmpecpp -lgmp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

# Independent certificate checker. Shares no code with the prover: the
# point of a proof is that it can be checked without trusting the prover.
$(VERIFY): tools/gmpecpp-verify.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Iinclude -o $@ $< \
	  -L. -lgmpecpp -lgmp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

$(BENCH): bench/gmpecpp-bench.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Iinclude -o $@ $< \
	  -L. -lgmpecpp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

$(REACH): bench/ecm-reach.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Iinclude -o $@ $< \
	  -L. -lgmpecpp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

$(KERNEL): bench/kernel-latency.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -Iinclude -o $@ $< \
	  -L. -lgmpecpp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

clean:
	rm -rf build $(LIB) $(SONAME) $(LIB_SO) $(BIN) $(BENCH) $(REACH) \
	  $(KERNEL) $(VERIFY)
	rm -f doc/*.aux doc/*.log doc/*.out doc/*.toc

# Reformats in place per .clang-format. Note that clang-format 19 skips
# the construct following any function whose return type is an
# elaborated `struct X`, of which this file has 11 -- so a bare run can
# leave a few spots untouched.
format:
	clang-format -style=file -i src/gmpecpp.cpp include/gmpecpp.h \
	  tools/gmp-ecpp.cpp bench/gmpecpp-bench.c

# The article. Needs pdflatex; three passes for the table of contents.
doc: gmpecpp-design.pdf

gmpecpp-design.pdf: gmpecpp-design.tex
	pdflatex -interaction=nonstopmode -halt-on-error gmpecpp-design.tex \
	  && pdflatex -interaction=nonstopmode -halt-on-error gmpecpp-design.tex \
	  && pdflatex -interaction=nonstopmode -halt-on-error gmpecpp-design.tex

cuda-info:
	@echo "CUDA        : $(CUDA)"
	@echo "NVCC        : $(NVCC)"
	@echo "CUDA_ARCH   : $(CUDA_ARCH)"
	@echo "CGBN_INC    : $(CGBN_INC)"
	@echo "linking     : $(CUDA_OBJ)"

# The exit code is the oracle: 0 proven, 1 probably prime (-o), 2 both
# composite and gave-up. The first case stays under SIEVE_LIMIT (2^30)
# and so only exercises trial division; the second is the one that
# matters, a full recursive ECPP downrun.
check: $(BIN) $(VERIFY)
	@echo "== trial-division path (< 2^30), expect proven prime =="
	@echo 1000003 | ./$(BIN) -Q -s $(SEED); \
	  test $$? -eq 0 || { echo "FAIL: expected exit 0"; exit 1; }
	@echo "== full ECPP downrun (31 digits), expect proven prime =="
	@echo 1000000000000000000000000000057 | ./$(BIN) -Q -s $(SEED); \
	  test $$? -eq 0 || { echo "FAIL: expected exit 0"; exit 1; }
	@echo "== composite (product of two 19-digit primes), expect composite =="
	@echo 1000000000000000012000000000000000027 | ./$(BIN) -Q -s $(SEED); \
	  test $$? -eq 2 || { echo "FAIL: expected exit 2"; exit 1; }
ifeq ($(CUDA),1)
	@echo "== same three via the CUDA/CGBN backend =="
	@echo 1000003 | ./$(BIN) -Q -C -s $(SEED) >/dev/null; \
	  test $$? -eq 0 || { echo "FAIL: expected exit 0"; exit 1; }
	@echo 1000000000000000000000000000057 | ./$(BIN) -Q -C -s $(SEED) >/dev/null; \
	  test $$? -eq 0 || { echo "FAIL: expected exit 0"; exit 1; }
	@echo 1000000000000000012000000000000000027 | ./$(BIN) -Q -C -s $(SEED) >/dev/null; \
	  test $$? -eq 2 || { echo "FAIL: expected exit 2"; exit 1; }
endif
	@echo "== certificate verified independently (and verifier self-tested) =="
	@./$(VERIFY) --selftest 1000000000000000000000000000057 >/dev/null || \
	  { echo "FAIL: verifier accepted a bad certificate"; exit 1; }
	@./$(VERIFY) 100000000000000000000000000000000000000000000000000000000019 \
	  | grep -q "verified independently" || \
	  { echo "FAIL: certificate did not verify"; exit 1; }
ifeq ($(CUDA),1)
	@./$(VERIFY) -C 100000000000000000000000000000000000000000000000000000000019 \
	  | grep -q "verified independently" || \
	  { echo "FAIL: CUDA certificate did not verify"; exit 1; }
endif
	@echo "== all smoke tests passed =="

# Each of these regenerates one of the tables in README.md. They are
# separate targets because the sweeps take minutes, not seconds.
#
# No -p: benchmark at the library's default precision, whatever it is.
bench: $(BENCH)
	./$(BENCH) -c 4096 -b 50000

# How large a factor can each backend find at all?
bench-reach: $(REACH)
	./$(REACH) -c 4096 -b 50000 bench/semiprimes.txt

# Is the GPU kernel latency bound or throughput bound?
bench-kernel: $(KERNEL)
	./$(KERNEL) bench/semiprimes.txt

# Why the default precision is 1000 bits and not 10000. Several minutes.
bench-precision: $(BIN)
	bash bench/precision-sweep.sh

# Whole proofs, CPU vs CUDA, across sizes and seeds. Many minutes.
bench-cuda: $(BIN)
	bash bench/cuda-sweep.sh

# Where the time actually goes. Pass -p to profile at another precision.
bench-profile: $(BIN)
	python3 bench/profile-phases.py -p 1000
	python3 bench/profile-phases.py -p 10000

# Byte-identity against a reference build of the original program.
# EXTRA=-C additionally checks the CUDA backend stays inert below the gate.
parity: $(BIN)
	bash bench/parity.sh

# Prove with both backends and check every certificate. The right test
# above the size gate, where the certificates legitimately differ.
verify-all: $(BIN) $(VERIFY)
	bash bench/verify-both.sh

bench-all: bench bench-reach bench-kernel bench-profile bench-precision bench-cuda

help:
	@echo "make            build $(LIB), ./$(BIN) and ./$(BENCH)"
	@echo "make check      smoke tests (override seed with SEED=n)"
	@echo "make bench           CPU vs CUDA, whole proofs"
	@echo "make bench-reach     largest factor each backend can find"
	@echo "make bench-kernel    GPU kernel cost vs B1 and curve count"
	@echo "make bench-profile   per-phase time at 1000 and 10000 bits"
	@echo "make bench-precision precision sweep (minutes)"
	@echo "make bench-cuda      CPU vs CUDA sweep (many minutes)"
	@echo "make bench-all       all of the above"
	@echo "make parity          byte-identity vs the original program"
	@echo "make verify-all      certificates from both backends"
	@echo "./$(VERIFY) [-C] [-v] N   prove N and check the certificate"
	@echo "make clean      remove build artefacts"
	@echo "make doc        build doc/gmpecpp.pdf (needs pdflatex)"
	@echo "make format     clang-format the sources"
	@echo "make cuda-info  show the detected CUDA configuration"
	@echo
	@echo "CUDA=0 forces a CPU-only build; it is auto-detected otherwise."
	@echo "The number under test is read from stdin, not argv:"
	@echo "    echo 1000000000000000000000000000057 | ./$(BIN) -Q"

