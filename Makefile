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
LDLIBS := -lgmpxx -lgmp -lpthread

# ---------------------------------------------------------------- #
# CUDA / CGBN                                                       #
# ---------------------------------------------------------------- #
#
# CGBN accelerates the ECM factoring stage only; everything else in the
# prover is either sequential or arbitrary precision floating point,
# which CGBN (fixed width integer) cannot express. Set CUDA=0 to force a
# CPU-only build.

# ---------------------------------------------------------------- #
# MPI (optional)                                                    #
# ---------------------------------------------------------------- #
#
# Used by gmpecpp-mpi, which distributes a batch of numbers across
# ranks (a single ECPP downrun is sequential and offers nothing to
# distribute; see README.md), and by gmpecmfft-mpi, which distributes
# one number's ECM curves across GPUs (needs CUDA too).

MPICC ?= $(shell command -v mpicc 2>/dev/null || \
           ls /usr/lib64/openmpi/bin/mpicc 2>/dev/null || echo mpicc)
MPI   ?= $(if $(wildcard $(MPICC)),1,$(shell command -v $(MPICC) >/dev/null 2>&1 && echo 1 || echo 0))

ifeq ($(MPI),1)
  MPI_TARGET := gmpecpp-mpi
else
  MPI_TARGET :=
endif

HERE       = $(shell pwd)
NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_ARCH ?= sm_120
# sm_120 is the RTX PRO 4500 Blackwell (CUDA 13.2, which /usr/local/cuda
# points to). For the RTX 4080 (Ada) use:
# CUDA_ARCH ?= sm_89
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

# cuFFT, for libgmpecmfft: ECM with FFT based modular arithmetic above
# the widest CGBN instance (16352 bits). libcufft, not libcufftw: the
# FFTW-style interface works on host arrays and would copy every operand
# across PCIe on every call.
CUFFT_HOME ?= /usr/local/cuda-13.2
CUFFT_INC  ?= $(CUFFT_HOME)/include
CUFFT_LIB  ?= $(CUFFT_HOME)/lib64

FFT_LIB_SO := libgmpecmfft.so
FFT_SONAME := $(FFT_LIB_SO).1
FFT_LIB    := $(FFT_LIB_SO).1.0.0
FFT_BIN    := gmpecmfft

ifeq ($(CUDA),1)
  FFT_TARGETS := $(FFT_LIB) $(FFT_BIN)
  ifeq ($(MPI),1)
    FFT_TARGETS += gmpecmfft-mpi
  endif
else
  FFT_TARGETS :=
endif

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

.PHONY: all clean check bench bench-all bench-reach bench-kernel bench-threads bench-sweep \
        bench-precision bench-cuda bench-profile parity verify-all \
        bench-mpi-cgbn bench-cgbn-latency bench-gmp-profile \
        doc format help cuda-info check-fft

all: $(LIB) $(BIN) $(BENCH) $(REACH) $(KERNEL) $(VERIFY) $(MPI_TARGET) \
     $(FFT_TARGETS)

build:
	@mkdir -p build

build/gmpecpp.o: src/gmpecpp.cpp include/gmpecpp.h | build
	$(CXX) $(CPPFLAGS) $(CUDA_CPPFLAGS) $(CXXFLAGS) -fPIC \
	  -fvisibility=hidden -fvisibility-inlines-hidden \
	  -I$(HERE)/include -c $< -o $@

build/ecm_stub.o: src/ecm_stub.cpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -fvisibility=hidden \
	  -I$(HERE)/include -c $< -o $@

build/ecm_cuda.o: src/ecm_cuda.cu | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -I$(HERE)/include -I$(CGBN_INC) -c $< -o $@

build/fftmod.o: src/fftmod.cu src/fftmod.cuh | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC,-fvisibility=hidden \
	  -I$(HERE)/include -I$(CUFFT_INC) -c $< -o $@

build/fftpoly.o: src/fftpoly.cu src/fftpoly.cuh src/fftmod.cuh | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC,-fvisibility=hidden \
	  -I$(HERE)/include -I$(CUFFT_INC) -c $< -o $@

build/ecmfft.o: src/ecmfft.cu src/fftmod.cuh src/fftpoly.cuh \
                include/gmpecmfft.h include/gmpecpp.h | build
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC,-fvisibility=hidden \
	  -I$(HERE)/include -I$(CUFFT_INC) -c $< -o $@

# Links against libgmpecpp, whose CGBN kernels it uses up to 16352 bits.
$(FFT_LIB): build/fftmod.o build/fftpoly.o build/ecmfft.o $(LIB)
	$(CXX) -shared -Wl,-soname,$(FFT_SONAME) -o $@ build/fftmod.o \
	  build/fftpoly.o build/ecmfft.o -L. -lgmpecpp -L$(CUFFT_LIB) -lcufft \
	  -L$(CUDA_LIB) -lcudart -lgmp -Wl,-rpath,'$$ORIGIN' \
	  -Wl,-rpath,$(CUFFT_LIB) $(CUDA_RPATH)
	ln -sf $(FFT_LIB) $(FFT_SONAME)
	ln -sf $(FFT_SONAME) $(FFT_LIB_SO)

$(FFT_BIN): tools/gmpecmfft.c $(FFT_LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(HERE)/include -o $@ $< \
	  -L. -lgmpecmfft -lgmp -Wl,-rpath,'$$ORIGIN'

# One number's ECM curves across MPI ranks, one GPU each (cudart for
# choosing the device).
gmpecmfft-mpi: tools/gmpecmfft-mpi.c $(FFT_LIB)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -I$(HERE)/include -I$(CUFFT_INC) -o $@ $< \
	  -L. -lgmpecmfft -L$(CUDA_LIB) -lcudart -Wl,-rpath,'$$ORIGIN' \
	  $(CUDA_RPATH)

$(LIB): build/gmpecpp.o $(CUDA_OBJ)
	$(CXX) -shared -Wl,-soname,$(SONAME) -o $@ $^ $(CUDA_LDFLAGS) $(LDLIBS)
	ln -sf $(LIB) $(SONAME)
	ln -sf $(SONAME) $(LIB_SO)

$(BIN): tools/gmp-ecpp.cpp $(LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(HERE)/include -o $@ $< \
	  -L. -lgmpecpp -lgmp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

# Independent certificate checker. Shares no code with the prover: the
# point of a proof is that it can be checked without trusting the prover.
$(VERIFY): tools/gmpecpp-verify.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(HERE)/include -o $@ $< \
	  -L. -lgmpecpp -lgmp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

$(BENCH): bench/gmpecpp-bench.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(HERE)/include -o $@ $< \
	  -L. -lgmpecpp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

# MPI + POSIX threads: ranks take whole numbers, threads split the ECM
# curves inside each. Built only when an MPI compiler is present.
gmpecpp-mpi: tools/gmpecpp-mpi.c $(LIB)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -I$(HERE)/include -o $@ $< \
	  -L. -lgmpecpp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

$(REACH): bench/ecm-reach.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(HERE)/include -o $@ $< \
	  -L. -lgmpecpp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

$(KERNEL): bench/kernel-latency.c $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I$(HERE)/include -o $@ $< \
	  -L. -lgmpecpp -Wl,-rpath,'$$ORIGIN' $(CUDA_RPATH)

clean:
	rm -rf build $(LIB) $(SONAME) $(LIB_SO) $(BIN) $(BENCH) $(REACH) \
	  $(KERNEL) $(VERIFY) gmpecpp-mpi $(FFT_LIB) $(FFT_SONAME) \
	  $(FFT_LIB_SO) $(FFT_BIN) gmpecmfft-mpi
	rm -f gmpecpp-design.aux gmpecpp-design.log gmpecpp-design.out \
	  gmpecpp-design.toc gmp-or-cgbn.aux gmp-or-cgbn.log gmp-or-cgbn.out \
	  gmp-or-cgbn.toc

# Reformats in place per .clang-format. Note that clang-format 19 skips
# the construct following any function whose return type is an
# elaborated `struct X`, of which this file has 11 -- so a bare run can
# leave a few spots untouched.
format:
	clang-format -style=file -i src/gmpecpp.cpp include/gmpecpp.h \
	  tools/gmp-ecpp.cpp bench/gmpecpp-bench.c

# The design document. Needs pdflatex; three passes for the table of
# contents and cross-references.
#
# pdflatex stamps /CreationDate and /ModDate into the PDF, so an
# unchanged document rebuilt a minute later differs in 64 bytes and shows
# up as modified in git. SOURCE_DATE_EPOCH (TeX Live 2016 and later)
# pins those, making the build byte-reproducible; FORCE_SOURCE_DATE makes
# \today follow it too, which matters if the explicit \date is ever
# removed. The default is the document's own last commit, so the stamp
# means something; override it, or set it to 0, if you prefer.
SOURCE_DATE_EPOCH ?= $(shell git log -1 --format=%ct -- gmpecpp-design.tex \
                       2>/dev/null || echo 0)

doc: gmpecpp-design.pdf gmp-or-cgbn.pdf

gmpecpp-design.pdf: gmpecpp-design.tex
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 \
	  pdflatex -interaction=nonstopmode -halt-on-error gmpecpp-design.tex \
	  && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 \
	  pdflatex -interaction=nonstopmode -halt-on-error gmpecpp-design.tex \
	  && SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 \
	  pdflatex -interaction=nonstopmode -halt-on-error gmpecpp-design.tex

# The GMP-or-CGBN report (CGBN-VS-GMP.md as LaTeX), stamped the same way
# with its own last commit.
GOC_SOURCE_DATE_EPOCH ?= $(or $(shell git log -1 --format=%ct -- gmp-or-cgbn.tex \
                           2>/dev/null),0)

gmp-or-cgbn.pdf: gmp-or-cgbn.tex
	SOURCE_DATE_EPOCH=$(GOC_SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 \
	  pdflatex -interaction=nonstopmode -halt-on-error gmp-or-cgbn.tex \
	  && SOURCE_DATE_EPOCH=$(GOC_SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 \
	  pdflatex -interaction=nonstopmode -halt-on-error gmp-or-cgbn.tex \
	  && SOURCE_DATE_EPOCH=$(GOC_SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 \
	  pdflatex -interaction=nonstopmode -halt-on-error gmp-or-cgbn.tex

cuda-info:
	@echo "MPI         : $(MPI) ($(MPICC))"
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

# libgmpecmfft: the FFT arithmetic against GMP at sizes from 200 bits to
# a million, then ECM on identical curves -- FFT path vs CGBN at 2000
# bits (stage 1 and 2 hit counts must match exactly) and, if gmp-ecm is
# installed, FFT path vs GMP-ECM at 20000 bits, beyond CGBN's reach.
check-fft: $(FFT_BIN)
	bash bench/fft-check.sh

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

# How the ECM stage scales with POSIX threads. Minutes.
bench-threads: $(BIN)
	bash bench/thread-sweep.sh

# How the discriminant sweep scales. Minutes.
bench-sweep: $(BIN)
	bash bench/sweep-threads.sh

# Where the time actually goes. Pass -p to profile at another precision.
bench-profile: $(BIN)
	python3 bench/profile-phases.py -p 1000
	python3 bench/profile-phases.py -p 10000

# CPU (GMP) against CGBN. bench-cgbn-latency: one GMP operation vs one
# CGBN operation vs a batch of 4096, 512 to 2048 bits. bench-gmp-profile:
# where CPU time goes, by library, in the prover and gmpecmfft (perf).
# bench-mpi-cgbn: gmpecpp-mpi with and without -C over 90-180 digit
# primes, several layouts and seeds (~40 min).
build/cgbn-latency: bench/cgbn-latency.cu | build
	$(NVCC) $(NVCCFLAGS) -I$(CGBN_INC) -o $@ $< -lgmp

bench-cgbn-latency: build/cgbn-latency
	./build/cgbn-latency

bench-gmp-profile: $(BIN) $(FFT_BIN)
	bash bench/gmp-profile.sh

bench-mpi-cgbn: gmpecpp-mpi
	bash bench/mpi-cgbn.sh build/bench-mpi-cgbn

# Byte-identity against a reference build of the original program.
# EXTRA=-C additionally checks the CUDA backend stays inert below the gate.
parity: $(BIN)
	bash bench/parity.sh

# Prove with both backends and check every certificate. The right test
# above the size gate, where the certificates legitimately differ.
verify-all: $(BIN) $(VERIFY)
	bash bench/verify-both.sh

bench-all: bench bench-reach bench-kernel bench-profile bench-precision \
           bench-cuda bench-threads bench-sweep

help:
	@echo "make            build $(LIB), ./$(BIN) and ./$(BENCH)"
	@echo "make check      smoke tests (override seed with SEED=n)"
	@echo "make bench           CPU vs CUDA, whole proofs"
	@echo "make bench-reach     largest factor each backend can find"
	@echo "make bench-kernel    GPU kernel cost vs B1 and curve count"
	@echo "make bench-profile   per-phase time at 1000 and 10000 bits"
	@echo "make bench-precision precision sweep (minutes)"
	@echo "make bench-cuda      CPU vs CUDA sweep (many minutes)"
	@echo "make bench-threads   ECM scaling with POSIX threads"
	@echo "make bench-sweep     discriminant sweep scaling"
	@echo "make bench-all       all of the above"
	@echo "make parity          byte-identity vs the original program"
	@echo "make verify-all      certificates from both backends"
	@echo "./$(VERIFY) [-C] [-v] N   prove N and check the certificate"
	@echo "make clean      remove build artefacts"
	@echo "make doc        build gmpecpp-design.pdf (byte-reproducible)"
	@echo "make format     clang-format the sources"
	@echo "make cuda-info  show the detected CUDA configuration"
	@echo "make check-fft  libgmpecmfft: FFT arithmetic vs GMP, and FFT vs"
	@echo "                CGBN on identical curves (minutes)"
	@echo "echo N | ./$(FFT_BIN) [-1 B1] [-2 B2] [-c C] [-v]   GPU ECM, any size"
	@echo
	@echo "CUDA=0 forces a CPU-only build; MPI=0 skips gmpecpp-mpi and"
	@echo "gmpecmfft-mpi."
	@echo "Both are auto-detected otherwise."
	@echo
	@echo "Parallelism:  ./gmp-ecpp -T N   ECM worker threads (0 = one per CPU)"
	@echo "              ./gmp-ecpp -S N   discriminant sweep threads"
	@echo "              mpirun -n R ./gmpecpp-mpi -T N file   batch across ranks"
	@echo "echo N | mpirun -n G+1 ./gmpecmfft-mpi [-1 B1] [-v]   ECM on G GPUs"
	@echo "echo N | tools/gmpecmfft-slurm submit -d DIR -n G ...  ... as a Slurm array"
	@echo "The number under test is read from stdin, not argv:"
	@echo "    echo 1000000000000000000000000000057 | ./$(BIN) -Q"

