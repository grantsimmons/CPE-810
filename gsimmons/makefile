CFLAGS = --std=c++17 -fconcepts -g
OMPFLAGS = -fopenmp

.PHONY:block
block: matrix_blocking.cpp
	g++ $^ $(CFLAGS) $(OMPFLAGS) -o run

.PHONY:block_mp
block_mp: omp_test.cpp
	export NUM_THREADS=16
	g++ $^ $(CFLAGS) $(OMPFLAGS) -o run

.PHONY: default
default: matrix.cpp
	g++ $^ $(CFLAGS) -o run

.PHONY: test
test: matrix_pivoting.cpp
	g++ $^ $(CFLAGS) -o test.out
