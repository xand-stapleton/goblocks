NVCCFLAGS  := -O3 -Xcompiler -fPIC
CXXFLAGS   := -O3 -fPIC
INCLUDES   := -I$(CUDA_PATH)/include
LIBS       := -L$(CUDA_PATH)/lib64 -lcublas -lcudart

ITER_SRC       := iterative_update.cu
RMATRIX_SRC    := rmatrix_kernel.cu
ITER_OBJ       := iterative_update.o
RMATRIX_OBJ    := rmatrix_kernel.o
OUTDIR         := ../../lib
ITER_LIB       := $(OUTDIR)/libgpuiter.so
RMATRIX_LIB    := $(OUTDIR)/librmatrix.so

all: $(ITER_LIB) $(RMATRIX_LIB)

$(ITER_LIB): $(ITER_OBJ)
	mkdir -p $(OUTDIR)
	g++ $(CXXFLAGS) -shared -o $(ITER_LIB) $(ITER_OBJ) $(LIBS)

$(RMATRIX_LIB): $(RMATRIX_OBJ)
	mkdir -p $(OUTDIR)
	g++ $(CXXFLAGS) -shared -o $(RMATRIX_LIB) $(RMATRIX_OBJ) $(LIBS)

$(ITER_OBJ): $(ITER_SRC)
	nvcc $(NVCCFLAGS) -c $(ITER_SRC) -o $(ITER_OBJ)

$(RMATRIX_OBJ): $(RMATRIX_SRC)
	nvcc $(NVCCFLAGS) -c $(RMATRIX_SRC) -o $(RMATRIX_OBJ)

clean:
	rm -f $(ITER_OBJ) $(RMATRIX_OBJ) $(ITER_LIB) $(RMATRIX_LIB)

.PHONY: all clean

