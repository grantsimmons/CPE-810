/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>

#include "convolutionTexture_common.h"

////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Use unrolled innermost convolution loop
#define UNROLL_INNER 0
//Kernel Radius is dynamic, so we can't use template unrolling

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel and input array storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[0x4000]; //Just max out

extern "C" void setConvolutionKernel(float *h_Kernel, int kernel_length)
{
    //printf("set kernel_length: %d\n", kernel_length);
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, kernel_length * sizeof(float));
}

#if(UNROLL_INNER)
////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float x, float y, cudaTextureObject_t texSrc)
{
    return
        tex2D<float>(texSrc, x + (float)(KERNEL_RADIUS - i), y) * c_Kernel[i]
        + convolutionRow<i - 1>(x, y, texSrc);
}

template<> __device__ float convolutionRow<-1>(float x, float y, cudaTextureObject_t texSrc)
{
    return 0;
}

template<int i> __device__ float convolutionColumn(float x, float y, cudaTextureObject_t texSrc)
{
    return
        tex2D<float>(texSrc, x, y + (float)(KERNEL_RADIUS - i)) * c_Kernel[i]
        + convolutionColumn<i - 1>(x, y, texSrc);
}

template<> __device__ float convolutionColumn<-1>(float x, float y, cudaTextureObject_t texSrc)
{
    return 0;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(
    float *d_Dst,
    int imageW,
    int imageH,
    int kernel_radius,
    cudaTextureObject_t texSrc
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionRow<2 *KERNEL_RADIUS>(x, y, texSrc);
#else

    //printf("Kernel Rad: %d\n", kernel_radius);
    for (int k = -kernel_radius; k <= kernel_radius; k++)
    {
        //printf("c_Kernel: %f\n", c_Kernel[kernel_radius - k]);
        sum += tex2D<float>(texSrc, x + (float)k, y) * c_Kernel[kernel_radius - k];
    }

#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}


extern "C" void convolutionRowsGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    int kernel_radius,
    int block_size,
    cudaTextureObject_t texSrc
)
{
    dim3 threads(block_size, block_size);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    convolutionRowsKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH,
        kernel_radius,
        texSrc
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(
    float *d_Dst,
    int imageW,
    int imageH,
    int kernel_radius,
    cudaTextureObject_t texSrc
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH)
    {
        return;
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionColumn<2 *KERNEL_RADIUS>(x, y, texSrc);
#else

    for (int k = -kernel_radius; k <= kernel_radius; k++)
    {
        sum += tex2D<float>(texSrc, x, y + (float)k) * c_Kernel[kernel_radius - k];
    }

#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    int kernel_radius,
    int block_size,
    cudaTextureObject_t texSrc
)
{
    dim3 threads(block_size, block_size);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    convolutionColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH,
        kernel_radius,
        texSrc
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
}
