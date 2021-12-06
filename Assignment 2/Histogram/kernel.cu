
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#define BIN_COUNT (int) 32

template <int BINS>
__global__ void histKernel(int* a, int arr_size, int* hist, int bin_width) {

    __shared__ unsigned int hist_local[BINS];
    
    if (blockDim.x < BINS) {
        //#pragma unroll
        for (int i = 0; i < BINS; i += blockDim.x) {
            hist_local[i + threadIdx.x] = 0;
        }
    }
    else {
        if (threadIdx.x < BINS) {
            hist_local[threadIdx.x] = 0;
        }
    }
    
    __syncthreads();

    int arr_index = blockIdx.x * blockDim.x + threadIdx.x;

    if(arr_index < arr_size) {
        int bin = a[arr_index] / bin_width;
        //printf("Thread: %d, Block: %d, Dim: %d; incrementing bin %d\n", threadIdx.x, blockIdx.x, blockDim.x, bin);
        atomicAdd(&hist_local[bin], 1);
        //atomicAdd(&hist[bin], 1);
    }

    __syncthreads();

    if (blockDim.x < BINS) {
        //#pragma unroll
        for (int i = 0; i < BINS; i += blockDim.x) {
            atomicAdd(&hist[i + threadIdx.x], hist_local[i + threadIdx.x]);
        }
    }
    else {
        if (threadIdx.x < BINS) {
            atomicAdd(&hist[threadIdx.x], hist_local[threadIdx.x]);
        }
    }
}


float calculateHist(int* a, int arr_size, int* hist, int bin_width, int block_width) {
    int* dev_a;
    int* dev_hist;

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    // Allocate GPU buffers for three vectors (two input, one output)    .
    checkCudaErrors(cudaMalloc((void**)&dev_a, arr_size * sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&dev_hist, BIN_COUNT * sizeof(int)));

    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(dev_a, a, arr_size * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dev_hist, hist, BIN_COUNT * sizeof(int), cudaMemcpyHostToDevice));

    dim3 threads(block_width, 1, 1);

    dim3 grid( ((arr_size - 1) / threads.x) + 1, 1, 1 );
    //dim3 grid(1, 1, 1);

    float time = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));
    
    histKernel<BIN_COUNT> << <grid, threads >> > (dev_a, arr_size, dev_hist, bin_width);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    printf("Time to generate: %3.5f ms \n", time);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(hist, dev_hist, BIN_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

    return time;
}

int main(int argc, char** argv)
{
    printf("[Histogram computation Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -bin_count=BinCount -vec_dim=VecDim ()\n");

        exit(EXIT_SUCCESS);
    }

    long int vec_dim_t = 100000000;
    int block_width = 256;

    if (checkCmdLineFlag(argc, (const char**)argv, "vec_dim"))
        vec_dim_t = getCmdLineArgumentInt(argc, (const char**)argv, "vec_dim");
    if (checkCmdLineFlag(argc, (const char**)argv, "block_width"))
        block_width = getCmdLineArgumentInt(argc, (const char**)argv, "block_width");
    //if (checkCmdLineFlag(argc, (const char**)argv, "vec_dim"))
    //    bin_count_t = getCmdLineArgumentInt(argc, (const char**)argv, "vec_dim");

    const long int vec_dim = vec_dim_t;

    srand(time(NULL));

    int* a = new int[vec_dim];

    const int max_value = RAND_MAX;
    //const int max_value = 2;

    const int bin_width = (int)((max_value - 1) / BIN_COUNT) + 1;
    printf("Bin width: %d\n", bin_width);

    printf("Array values: ");
    for (long int i = 0; i < vec_dim; i++) {
        int uval = (int)rand();
        //int uval = 20;
        int val = uval % max_value;
        a[i] = val;
        //printf("%d, ", val);
    }

    printf("\n");

    int* hist = new int[BIN_COUNT];

    for (int i = 0; i < BIN_COUNT; i++) {
        hist[i] = 0;
    }

    float time = calculateHist(a, vec_dim, hist, bin_width, block_width);

    printf("Array Size: %d, Bin Count: %d\n", vec_dim, BIN_COUNT);
    for (int i = 0; i < BIN_COUNT; i++) {
        printf("%d, ", hist[i]);
    }
    printf("\n");

    int block_count = (((vec_dim - 1) / block_width) + 1);
    int total_global_atomics = BIN_COUNT * block_count;
    int total_shared_atomics = vec_dim;

    printf("Global Memory Reads: %d\n", vec_dim);
    printf(" (Including atomic operations): %d\n", vec_dim + total_global_atomics);
    printf(" (Per Element): %f\n", (float)(vec_dim + total_global_atomics) / vec_dim);
    printf("Shared Memory Writes: %d\n", vec_dim);
    printf(" (Per Block): %.1f\n", (float) (vec_dim / block_width));
    printf("Global Memory Writes: %d\n", total_global_atomics);
    printf(" (Per Element): %f\n", (float)(total_global_atomics) / vec_dim);
    printf("GFLOPS: %f\n", (float) (vec_dim / (time / 1000)) / 1000000000);

    checkCudaErrors(cudaDeviceReset());

    return 0;
}
