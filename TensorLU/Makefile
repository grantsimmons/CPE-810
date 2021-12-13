CXX = c++
NVCC = nvcc
CUDA_VER = cuda-11.4
CUDA_INC = /usr/local/$(CUDA_VER)/include
CUDA_LIB = /usr/local/$(CUDA_VER)/lib64

MAIN = main.cpp
OBJS = 
HEADERS = lu_entry.hpp
CXX_INCLUDE = -I$(CUDA_INC) -I.
CXX_LINK = -L $(CUDA_LIB)
CXX_LIBS = -lcudart -lcublas
CXX_FLAGS = --std=c++11
CXX_OUT = tensor_lu_no_tensor

.PHONY: default
default: all

.PHONY: all
all: $(CXX_OUT)

$(CXX_OUT): $(MAIN) $(HEADERS) convert.o lu.o
	$(CXX) $^ $(CXX_FLAGS) $(CXX_INCLUDE) $(CXX_LINK) $(CXX_LIBS) -o $(CXX_OUT)

convert.o: convert.cuh convert.cu
	$(NVCC) -c convert.cu

lu.o: lu.cuh lu.cu
	$(NVCC) -c lu.cu

