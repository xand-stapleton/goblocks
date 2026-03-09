NVCCFLAGS  := -O3 -Xcompiler -fPIC
CXXFLAGS   := -O3 -fPIC
INCLUDES   := -I$(CUDA_PATH)/include
LIBS       := -L$(CUDA_PATH)/lib64 -lcublas -lcudart

SRC        := iterative_update.cu
OBJ        := iterative_update.o
OUTDIR     := ../lib
OUTLIB     := $(OUTDIR)/libgpu.so

all: $(OUTLIB)

$(OUTLIB): $(OBJ)
	mkdir -p $(OUTDIR)
	g++ $(CXXFLAGS) -shared -o $(OUTLIB) $(OBJ) $(LIBS)

$(OBJ): $(SRC)
	nvcc $(NVCCFLAGS) -c $(SRC) -o $(OBJ)

clean:
	rm -f $(OBJ) $(OUTLIB)

.PHONY: all clean

