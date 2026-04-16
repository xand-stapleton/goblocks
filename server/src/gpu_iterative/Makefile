NVCCFLAGS  := -O3 -Xcompiler -fPIC
CXXFLAGS   := -O3 -fPIC
INCLUDES   := -I$(CUDA_PATH)/include
LIBS       := -L$(CUDA_PATH)/lib64 -lcublas -lcudart

ITER_SRC       := iterative_update.cu
ITER_OBJ       := iterative_update.o
OUTDIR         := ../../lib
ITER_LIB       := $(OUTDIR)/libgpuiter.so

all: $(ITER_LIB)

$(ITER_LIB): $(ITER_OBJ)
	mkdir -p $(OUTDIR)
	g++ $(CXXFLAGS) -shared -o $(ITER_LIB) $(ITER_OBJ) $(LIBS)

$(ITER_OBJ): $(ITER_SRC)
	nvcc $(NVCCFLAGS) -c $(ITER_SRC) -o $(ITER_OBJ)

clean:
	rm -f $(ITER_OBJ) $(ITER_LIB)

.PHONY: all clean

