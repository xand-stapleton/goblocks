# ========================
# Compiler flags for CUDA
# ========================
NVCCFLAGS  := -O3 -Xcompiler -fPIC -arch=sm_70
CXXFLAGS   := -O3 -fPIC
INCLUDES   := -I$(CUDA_PATH)/include
LIBS       := -L$(CUDA_PATH)/lib64 -lcublas -lcudart

# ========================
# GPU sources
# ========================
GPU_ITER_SRC    := gpu_iterative/iterative_update.cu

GPU_ITER_OBJ    := $(patsubst %.cu,%.o,$(GPU_ITER_SRC))

GPU_OUTDIR      := ../lib
GPU_ITER_LIB    := $(GPU_OUTDIR)/libgpuiter.so

# ========================
# Go project settings
# ========================
BINARY_NAME   := goblocks
LIBRARY_NAME  := librecursive
GO_SRC        := $(wildcard *.go)

GPU ?= 0  # Set to 1 to enable GPU build

# ========================
# Default target
# ========================
.PHONY: all build clean fmt lint gpu test test-unit test-verbose test-knowngood test-regression
all: build

# ========================
# Build targets
# ========================
build: gpu_build go_build

gpu_build:
ifeq ($(GPU),1)
	@echo "Building CUDA libraries..."
	mkdir -p $(GPU_OUTDIR)

	# Build iterative update library
	nvcc $(NVCCFLAGS) -c $(GPU_ITER_SRC) -o $(GPU_ITER_OBJ)
	g++ $(CXXFLAGS) -shared -o $(GPU_ITER_LIB) $(GPU_ITER_OBJ) $(LIBS)
endif

go_build:
	@echo "Building Go project..."
	mkdir -p ../bin/ ../lib/
ifeq ($(GPU),1)
	go build -tags=gpu ./...
	go build -tags=gpu -o ../bin/$(BINARY_NAME) .
	go build -tags=gpu -o ../lib/$(LIBRARY_NAME).so -buildmode=c-shared
else
	go build -o ../bin/$(BINARY_NAME) .
	go build -o ../lib/$(LIBRARY_NAME).so -buildmode=c-shared
endif

# ========================
# Clean targets
# ========================
clean:
	@echo "Cleaning Go binaries..."
	rm -f ../bin/$(BINARY_NAME) ../lib/$(LIBRARY_NAME).so
ifeq ($(GPU),1)
	@echo "Cleaning GPU artifacts..."
	rm -f $(GPU_ITER_OBJ) $(GPU_ITER_LIB)
endif

# ========================
# Formatting and linting
# ========================
fmt:
	go fmt ./...

lint:
	golangci-lint run

# ========================
# Testing
# ========================
test: test-unit

test-verbose: test-unit

test-unit:
ifeq ($(GPU),1)
	go test -v -tags=gpu -count=1 -run 'Test' ./...
else
	go test -v -count=1 -run 'Test' ./...
endif

test-knowngood:
ifeq ($(GPU),1)
	go test -tags=gpu -run 'TestRecurseAndEvaluateDFDefaultsNmax8KnownGood(|NumericAndNonNumeric)$$' -count=1 ./...
else
	go test -run 'TestRecurseAndEvaluateDFDefaultsNmax8KnownGood(|NumericAndNonNumeric)$$' -count=1 ./...
endif

test-regression: test-knowngood

