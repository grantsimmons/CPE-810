#include "convert.cuh"
#include <cstdint>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#define BLK_X 8
#define BLK_Y BLK_X

const int max_blocks = INT16_MAX;

static __device__
   void convert_sp2hp_device(
         int64_t m, int64_t n,
         const float  *dA, int64_t ldda,
         __half *dB, int64_t lddb )
   {
      int64_t ind = blockIdx.x*BLK_X + threadIdx.x;
      int64_t iby = blockIdx.y*BLK_Y;
      /* check if full block-column */
      bool full = (iby + BLK_Y <= n);

      /* do only rows inside matrix */
      if ( ind < m ) {

         dA += ind + iby*ldda;
         dB += ind + iby*lddb;


         if ( full ) {
            // full block-column
#pragma unroll
            for( std::int64_t j=0; j < BLK_Y; ++j ) {
               dB[j*lddb] = dA[j*ldda] ;
            }
         }
         else {
            // partial block-column
            for( std::int64_t j=0; j < BLK_Y && iby+j < n; ++j ) {
               dB[j*lddb] = dA[j*ldda];
            }
         }
      }
   }

__global__
void convert_kernel(
        int m, int n,
        const float  *dA, int ldda,
        __half *dB, int lddb ) {
    convert_sp2hp_device(m, n, dA, ldda, dB, lddb);
}

void convert(
        cudaStream_t const stream,
        int m, int n,
        float *const a, int lda, 
        __half *const aout, int ldaout) {
            
    if(BLK_X != BLK_Y) {
        exit(1);
    }
    std::int64_t const super_NB = max_blocks*BLK_X;
    dim3 super_grid(
        (m + super_NB - 1) / super_NB, 
        (n + super_NB - 1) / super_NB);
    
    dim3 threads( BLK_X, 1 );
    dim3 grid;

    std::int64_t mm, nn;
    for( std::int64_t i=0; i < super_grid.x; ++i ) {
        mm = (i == super_grid.x-1 ? m % super_NB : super_NB);
        grid.x = (mm + BLK_X - 1) / BLK_X;
        for( std::int64_t j=0; j < super_grid.y; ++j ) {  // full row
        nn = (j == super_grid.y-1 ? n % super_NB : super_NB);
        grid.y = (nn + BLK_Y - 1) / BLK_Y;

        convert_kernel 
            <<< grid, threads, 0, stream >>>
            (mm, nn, &a[i*super_NB + j*super_NB*lda], lda, &aout[i*super_NB + j*super_NB*ldaout], ldaout);
        }
    }
}
   
   // FP32 to FP16
void convert(
         cudaStream_t const stream, int m, int n, float *const a, int lda, 
         half *const aout, int ldaout);