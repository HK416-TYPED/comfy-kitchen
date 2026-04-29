# Build all phaseN_*.cu kernels into shared libraries for ctypes loading.
#
# Each phase has one .cu producing one .so with extern "C" launch entry point.
# Standalone — no dependency on kitchen's CMake / nanobind.

NVCC ?= nvcc
ARCH ?= sm_120
NVCC_FLAGS = -O3 -arch=$(ARCH) -Xcompiler -fPIC -shared --use_fast_math -lineinfo

PHASE_SOURCES := $(wildcard phase*.cu)
PHASE_LIBS    := $(PHASE_SOURCES:.cu=.so)

all: $(PHASE_LIBS)

%.so: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(PHASE_LIBS)

.PHONY: all clean
