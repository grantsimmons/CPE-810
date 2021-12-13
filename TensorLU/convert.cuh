#pragma once

// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

   
using half = __half;


void convert(cudaStream_t const stream, int m, int n, float *const a, int lda, half *const aout, int ldaout);