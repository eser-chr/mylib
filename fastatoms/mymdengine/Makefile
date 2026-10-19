# Standalone executables (the Python module is built by setup.py).
#   make cpu   -> ./md_cpu   (CPU only)
#   make gpu   -> ./md_gpu   (CPU + GPU comparison)
CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS  = -O3 -march=native -std=c++17
# nvcc 12.0 supports gcc <= 12; override with `make NVCC_CCBIN=` if not needed.
NVCC_CCBIN ?= g++-12
NVCCFLAGS = -O3 -std=c++17 -arch=native $(if $(NVCC_CCBIN),-ccbin $(NVCC_CCBIN))

all: cpu gpu

cpu: md_cpu
gpu: md_gpu

md_cpu: src/main.cpp src/atoms_cpu.hpp src/common.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

md_gpu: src/main.cpp src/atoms_gpu.cu src/atoms_cpu.hpp src/atoms_gpu.cuh src/common.hpp
	$(NVCC) $(NVCCFLAGS) -DWITH_CUDA -x cu src/main.cpp src/atoms_gpu.cu -o $@

clean:
	rm -f md_cpu md_gpu
	rm -rf build python/*.egg-info python/mymd/*.so

.PHONY: all cpu gpu clean
