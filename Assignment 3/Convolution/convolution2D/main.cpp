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


/*
 * This sample implements the same algorithm as the convolutionSeparable
 * CUDA Sample, but without using the shared memory at all.
 * Instead, it uses textures in exactly the same way an OpenGL-based
 * implementation would do.
 * Refer to the "Performance" section of convolutionSeparable whitepaper.
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "convolutionTexture_common.h"



////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    float
    *h_Kernel,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    cudaArray *a_Src;
    cudaTextureObject_t texSrc;
    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    int kernel_length;
    int kernel_radius;
    kernel_length = 17;
    int image_width = 3270;
    int image_height = 3270 / 2;
    int block_size_noconst = 16;

    if (checkCmdLineFlag(argc, (const char**)argv, "k"))
        kernel_length = getCmdLineArgumentInt(argc, (const char**)argv, "k"); //dimK
    if (checkCmdLineFlag(argc, (const char**)argv, "i"))
        image_width = getCmdLineArgumentInt(argc, (const char**)argv, "i"); //dimX
    if (checkCmdLineFlag(argc, (const char**)argv, "j"))
        image_height = getCmdLineArgumentInt(argc, (const char**)argv, "j"); //dimY
    if (checkCmdLineFlag(argc, (const char**)argv, "b"))
        block_size_noconst = getCmdLineArgumentInt(argc, (const char**)argv, "b"); //Block Size
    
    if (kernel_length % 2 == 0) {
        printf("ERROR: Convolution Kernel is not an odd integer");
        exit(EXIT_FAILURE);
    }
    if (kernel_length < 1) {
        printf("ERROR: Invalid Kernel Length (%d)", kernel_length);
        exit(EXIT_FAILURE);
    }
    if (image_width < 1 || image_height < 1) {
        printf("ERROR: Invalid image size (%d, %d)", image_width, image_height);
        exit(EXIT_FAILURE);
    }

    kernel_radius = (kernel_length - 1) / 2;
    float
    *d_Output;

    double
    gpuTime;

    StopWatchInterface *hTimer = NULL;

    const unsigned long int imageW = image_width;
    const unsigned long int imageH = image_height;
    const unsigned int iterations = 1;
    const int block_size = block_size_noconst;

    printf("[%s] - Starting...\n", argv[0]);    printf("imageW: %d, imageH: %d, Kern Len: %d, Kern Rad: %d\n", imageW, imageH, kernel_length, kernel_radius);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //findCudaDevice(argc, 0);
    cudaSetDevice(0);

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    h_Kernel    = (float *)malloc(kernel_length * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));

    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = a_Src;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texSrc, &texRes, &texDescr, NULL));

    srand(2009);

    for (unsigned int i = 0; i < kernel_length; i++)
    {
        //Populate convolution mask with random data
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned int i = 0; i < imageW * imageH; i++)
    {
        //Populate input data with random data
        h_Input[i] = (float)(rand() % 16);
    }

    setConvolutionKernel(h_Kernel, kernel_length);
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));


    printf("Running GPU rows convolution (%u identical iterations)...\n", iterations);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (unsigned int i = 0; i < iterations; i++)
    {
        convolutionRowsGPU(
            d_Output,
            a_Src,
            imageW,
            imageH,
            kernel_radius,
            block_size,
            texSrc
        );
    }

    checkCudaErrors(cudaDeviceSynchronize());
    double kernelTime = 0;
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    kernelTime += (double) gpuTime;
    double rGflops;
    rGflops = 2 * imageW * imageH * kernel_length * 1e-9 / (0.001 * gpuTime);
    printf("Average convolutionRowsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));
    printf("rows GFLOPS: %lf\n", rGflops);
    printf("rows GMEM_READS: %i\n", imageW * imageH);
    printf("rows TMEM_READS: %li\n", imageH * kernel_length * imageW);

    //While CUDA kernels can't write to textures directly, this copy is inevitable
    printf("Copying convolutionRowGPU() output back to the texture...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("cudaMemcpyToArray() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    printf("Running GPU columns convolution (%i iterations)\n", iterations);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    for (int i = 0; i < iterations; i++)
    {
        convolutionColumnsGPU(
            d_Output,
            a_Src,
            imageW,
            imageH,
            kernel_radius,
            block_size,
            texSrc
        );
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer) / (float)iterations;
    kernelTime += (double) gpuTime;
    double cGflops;
    cGflops = 2 * imageW * imageH * kernel_length * 1e-9 / (0.001 * gpuTime);
    printf("GPU_TIME: %lf", kernelTime);
    printf("Average convolutionColumnsGPU() time: %f msecs; //%f Mpix/s;\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));
    printf("columns GFLOPS: %lf\n", cGflops);
    printf("columns GMEM_READS: %i\n", imageW * imageH);
    printf("columns TMEM_READS: %li\n", imageH * kernel_length * imageW);

    printf("total GFLOPS: %lf\n", (cGflops + rGflops) / 2);
    printf("total GMEM_READS: %i\n", 2 * imageW * imageH);
    printf("total TMEM_READS: %llu\n", (__int64) (2 * imageH * kernel_length * imageW));

    printf("Reading back GPU results...\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Checking the results...\n");
    printf("...running convolutionRowsCPU()\n");

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    convolutionRowsCPU(
        h_Buffer,
        h_Input,
        h_Kernel,
        imageW,
        imageH,
        kernel_radius
    );
    sdkStopTimer(&hTimer);

    double cpuTime = sdkGetTimerValue(&hTimer);
    printf("...running convolutionColumnsCPU()\n");
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    convolutionColumnsCPU(
        h_OutputCPU,
        h_Buffer,
        h_Kernel,
        imageW,
        imageH,
        kernel_radius
    );
    sdkStopTimer(&hTimer);
    cpuTime += sdkGetTimerValue(&hTimer);
    printf("CPU_TIME: %lf", cpuTime);

    double delta = 0;
    double sum = 0;

    for (unsigned int i = 0; i < imageW * imageH; i++)
    {
        //printf("(i=%d): GPU: %f, CPU: %f\n", i, h_OutputGPU[i], h_OutputCPU[i]);
        sum += h_OutputCPU[i] * h_OutputCPU[i];
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
    }

    double L2norm = sqrt(delta / sum);
    printf("Relative L2 norm: %E\n", L2norm);
    printf("Shutting down...\n");

    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFreeArray(a_Src));
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    sdkDeleteTimer(&hTimer);

    if (L2norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
