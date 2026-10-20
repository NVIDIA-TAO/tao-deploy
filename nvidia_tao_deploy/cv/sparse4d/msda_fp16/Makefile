# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
TENSORRT_INCLUDE ?= /usr/include/$(shell $(CXX) -dumpmachine)
TENSORRT_LIB ?= /usr/lib/$(shell $(CXX) -dumpmachine)
BUILD_DIR ?= build
# Compile only architectures understood by this CUDA toolkit. Override to reduce
# build time, for example CUDA_ARCHS=120. No GPU is needed to compile.
supported_archs = $(patsubst sm_%,%,$(shell $(NVCC) --list-gpu-code 2>/dev/null))
CUDA_ARCHS ?= $(filter 75 80 86 87 89 90 100 103 110 120 121,$(supported_archs))
gencode = $(foreach arch,$(CUDA_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
includes = -I$(CUDA_HOME)/include -I$(TENSORRT_INCLUDE) -Isrc
compile_flags = -O3 -std=c++17 -fPIC
objects = $(BUILD_DIR)/deformable_aggregation_cuda.o $(BUILD_DIR)/deformable_aggregation_plugin.o
library = $(BUILD_DIR)/libmsda_fp16.so

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
.PHONY: all check-toolchain test-host clean
all: $(library)

check-toolchain:
	@test -x "$(NVCC)" || { echo "Set CUDA_HOME or NVCC to a CUDA toolkit with nvcc"; exit 1; }
	@test -f "$(TENSORRT_INCLUDE)/NvInfer.h" || { echo "Set TENSORRT_INCLUDE to the TensorRT header directory"; exit 1; }
	@test -n "$(CUDA_ARCHS)" || { echo "No supported CUDA architectures; set CUDA_ARCHS explicitly"; exit 1; }
	@test -z "$(filter-out $(supported_archs),$(CUDA_ARCHS))" || { echo "Unsupported CUDA_ARCHS: $(filter-out $(supported_archs),$(CUDA_ARCHS))"; exit 1; }

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/deformable_aggregation_cuda.o: src/deformable_aggregation_cuda.cu src/deformable_aggregation_cuda.h | $(BUILD_DIR) check-toolchain
	$(NVCC) $(includes) -O3 -std=c++17 -Xcompiler -fPIC $(gencode) -c $< -o $@

$(BUILD_DIR)/deformable_aggregation_plugin.o: src/deformable_aggregation_plugin.cpp src/deformable_aggregation_plugin.h src/deformable_aggregation_cuda.h src/check.hpp | $(BUILD_DIR) check-toolchain
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(compile_flags) $(includes) -c $< -o $@

$(library): $(objects)
	$(CXX) -shared $(LDFLAGS) $^ -L$(CUDA_HOME)/lib64 -L$(TENSORRT_LIB) -lcudart -lnvinfer -lnvinfer_plugin -o $@

$(BUILD_DIR)/test_plugin_host: tests/test_plugin_host.cpp $(library)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(compile_flags) $(includes) $< -L$(BUILD_DIR) -lmsda_fp16 -L$(TENSORRT_LIB) -lnvinfer -Wl,-rpath,'$$ORIGIN' -o $@

test-host: $(BUILD_DIR)/test_plugin_host
	$(BUILD_DIR)/test_plugin_host

clean:
	rm -f $(objects) $(library) $(BUILD_DIR)/test_plugin_host
