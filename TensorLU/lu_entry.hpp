#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstring>
#include <chrono>
#include <cstdio>
#include <random>
#include <iostream>

#include <cstdint>
#include <cuda_fp16.h>
#include <algorithm>

#include "convert.cuh"
#include "lu.cuh"

#include "helper_cuda.h"

#define BLOCK_SIZE 8

#define CUBLAS_ALGO CUBLAS_GEMM_ALGO4

struct Opts {
    size_t m = 8;
    size_t nb = 2;
};

void parse_opts(Opts* opts, int argc, char** argv) {
    for(int i = 0; i < argc; i++) {
        if(!strcmp("--m", argv[i]) && i + 1 < argc) {
            opts->m = atoi(argv[++i]);
        }
        if(!strcmp("--nb", argv[i]) && i + 1 < argc) {
            opts->nb = atoi(argv[++i]);
        }
    }
}

void cublas_check_error(cublasStatus_t custat) {
    if (custat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Cublas Error");
    }
}

template<
    typename T,
    int64_t ib=BLOCK_SIZE,
    typename CpyType = half
    >
void rl (
        const cublasHandle_t cuhandle, 
         int64_t m, // Number of rows 
         int64_t n, // Number of columns
         int64_t nb, // Outer block size
         T *const d_a, // Matrix pointer on device 
         int64_t ld_a, // Matrix leadind dim on device
         T *work, // Workspace
         int *d_info, // Info device
         CpyType *d_a_cpy=nullptr, // Matrix copy in CpyType
         int64_t ld_a_cpy=-1

) {
          cudaError_t cuerr; // CUDA error
      cublasStatus_t custat; // CuBLAS status
      cudaStream_t stream; // CUDA Stream
      // Retreive CUDA stream from cuBLAS handle
      custat = cublasGetStream(cuhandle, &stream);
      cublas_check_error(custat);

      cublasHandle_t queue = cuhandle;

            // Workspace
      T *d_d = nullptr;
      std::int64_t lddd = ib;

            d_d = work;

                  T alpha = -1.0, beta = 1.0;
      float alpha_fp32 = -1.0, beta_fp32 = 1.0;

      std::int64_t inc = (n-1) / ib + 1; 
      std::int64_t ofst = nb;

    for(int64_t k = 0; k < inc; ++k) {
         // Factor kth block column
         std::int64_t iofs = k*ib; // Number of eliminated columns
         std::int64_t cblkm = m-iofs; // Block column height
         std::int64_t cblkn = std::min(n-iofs, ib); // Block column width
         // Trailing submatrix info
         std::int64_t iofst = (k+1)*ib; // Offset to trailing submatrix in outer block
         std::int64_t tblkm = m-iofst; // Width of trailing submatrix in outer block
         std::int64_t tblkn = n-iofst; // Width of trailing submatrix in outer block
         cudaMemcpy2DAsync(
               d_d, lddd*sizeof(T),
               &d_a[iofs+iofs*ld_a], ld_a*sizeof(T),
               cblkn*sizeof(T), cblkn,
               cudaMemcpyDeviceToDevice,
               stream);


         lu_nopiv_panel(
               stream, cblkm, cblkn,
               d_d, lddd,
               &d_a[iofs+iofs*ld_a], ld_a,
               &d_a[iofs+iofs*ld_a], ld_a,
               d_info);

        // Copy of L and U factors into buffers
        if (cblkm>cblkn) {
            convert(stream, cblkn, cblkm-cblkn, &d_a[iofs+iofst*ld_a], ld_a, &d_a_cpy[iofs+iofst*ld_a_cpy], ld_a_cpy);
        }
        convert(stream, cblkm, cblkn, &d_a[iofs+iofs *ld_a], ld_a, &d_a_cpy[iofs+iofs *ld_a_cpy], ld_a_cpy);

        if(tblkn>0) {
            float alpha = -1.0;
            float beta = 1.0;

            //Update L
            cublasStatus_t custat = cublasGemmEx(
                queue, CUBLAS_OP_N, CUBLAS_OP_N,
                tblkm, tblkn, ib,
                &alpha,
                &d_a_cpy[iofst + iofs * ld_a_cpy], CUDA_R_16F, ld_a_cpy,
                &d_a_cpy[iofs + iofst * ld_a_cpy], CUDA_R_16F, ld_a_cpy,
                &beta,
                &d_a[iofst + iofst*ld_a], CUDA_R_32F, ld_a,
                CUDA_R_32F,
                CUBLAS_ALGO);
                //CUBLAS_GEMM_ALGO0_TENSOR_OP);

            if(tblkm>tblkn) {
                //Update U
                cublasStatus_t custat = cublasGemmEx(
                    queue, CUBLAS_OP_N, CUBLAS_OP_N,
                    tblkn, tblkm-tblkn, ib,
                    &alpha,
                    &d_a_cpy[iofst + iofs * ld_a_cpy], CUDA_R_16F, ld_a_cpy,
                    &d_a_cpy[iofs + ofst * ld_a_cpy], CUDA_R_16F, ld_a_cpy,
                    &beta,
                    &d_a[iofst + ofst*ld_a], CUDA_R_32F, ld_a,
                    CUDA_R_32F,
                    CUBLAS_ALGO);
                    //CUBLAS_GEMM_ALGO0_TENSOR_OP);
            }
        }

    }
}

template<
    typename T, // Working prec
    //remifa::compute_type I=FP32, // Inner update comupte type
    //remifa::compute_type O=FP32, // Outer update comupte type
    int ib=BLOCK_SIZE,
    // int nb=OUTER_BLOCK_SIZE,
    //bool use_cutlass=false,
    typename CpyType = half
    >
void decompose_right_looking(
    const cublasHandle_t cuhandle, 
    int64_t m, // Number of rows 
    int64_t n, // Number of columns
    int64_t nb,
    T *const d_a, // Matrix pointer on device 
    int64_t ldda, // Matrix leadind dim on device
    int *d_info, // Info device
    CpyType *d_a_cpy=nullptr, // Matrix copy in CpyType
    int64_t ld_a_cpy=-1
) {
    cudaError_t cuerr; // CUDA error
    cublasStatus_t custat; // CuBLAS status
    
    // Number of (outer) block columns
    std::cout << nb << std::endl;
    std::int64_t const nc = (n-1) / nb + 1; //Max of num blocks in columns

    // if (std::is_same<T, float>::value && (TC32==O) &&
    //     d_a_cpy == nullptr) {
    //     throw std::invalid_argument("A fp16 buffer should be provided");
    // }

          cudaStream_t stream; // CUDA Stream
      // Retreive CUDA stream from cuBLAS handle
      custat = cublasGetStream(cuhandle, &stream);
      cublas_check_error(custat);

    cublasHandle_t queue = cuhandle;

    T alpha = -1.0, beta = 1.0;
    float alpha_fp32 = -1.0, beta_fp32 = 1.0;

     // Workspace
      T *d_d = nullptr;
      std::int64_t lddd = ib;
      cuerr = cudaMalloc((void**)&d_d, lddd*ib*sizeof(T));
      checkCudaErrors(cuerr);

    // Allocate buffer for storing a FP16 copy of the panel
      half *d_l_tmp = nullptr;
      std::int64_t lddl = ldda;
    
      half *d_u_tmp = nullptr;
      std::int64_t lddu = ldda;

      for (int64_t kk = 0; kk < nc; ++kk) {
                   std::int64_t ofs = kk*nb;
         std::int64_t ofst = (kk+1)*nb;
         std::int64_t in = std::min(n-ofs, nb);
         std::int64_t inc = (in-1) / ib + 1; 
         std::int64_t updm = m-ofs;

        rl
             <T, ib>
             (cuhandle, 
              updm, in, nb, 
              &d_a[ofs + ofs*ldda], ldda,
              d_d, d_info,
              &d_a_cpy[ofs + ofs*ldda], ld_a_cpy);

        // Copy of L and U factors into buffers
        if (updm>in) {
            convert(stream, in, updm-in, &d_a[ofs+ofst*ldda], ldda, &d_a_cpy[ofs+ofst*ld_a_cpy], ld_a_cpy);
        }
        convert(stream, updm, in, &d_a[ofs+ofs *ldda], ldda, &d_a_cpy[ofs+ofs *ld_a_cpy], ld_a_cpy);

                 //
         // Perform trailing submatrix update w.r.t previous panel
         // factorization
         //
         int64_t tm = m-ofst; // Width of trailing submatrix
         int64_t tn = n-ofst; // Width of trailing submatrix

         float alpha = -1.0;
         float beta = 1.0;

         if (tn>0) {
            cublasStatus_t custat = cublasGemmEx(
                queue, CUBLAS_OP_N, CUBLAS_OP_N,
                tm, tn, nb,
                &alpha,
                &d_a_cpy[ofst + ofs * ld_a_cpy], CUDA_R_16F, ld_a_cpy,
                &d_a_cpy[ofs + ofst*ld_a_cpy], CUDA_R_16F, ld_a_cpy,
                &beta,
                &d_a[ofst + ofst*ldda], CUDA_R_32F, ldda,
                CUDA_R_32F,
                CUBLAS_ALGO);
                //CUBLAS_GEMM_ALGO0_TENSOR_OP);
            cublas_check_error(custat);      
         }

      }

            // Wait for completion
      cuerr = cudaStreamSynchronize(stream);
      checkCudaErrors(cuerr);

      // Cleanup memory
      // cublasDestroy(cuhandle);
      cuerr = cudaFree(d_d);
      checkCudaErrors(cuerr);
}


template <typename T>
void gen_random_matrix(int m, int n, T* a, int lda) {
    std::cout << "Generating matrix" << std::endl;
    std::default_random_engine generator(1u);
    std::uniform_real_distribution<T> distribution(-1.0, 1.0);

    for(int j = 0; j < n; ++j)
        for(int i = 0; i < m; ++i) {
            // a[j*lda+i] =  1.0;
            a[j * lda + i] =  distribution(generator);
        }
}

template <typename T>
void gen_diagdom_matrix(int m, int n, T* a, int lda) {
    std::cout << "Generating Diagonal Dominant Matrix" << std::endl;
    std::default_random_engine generator(1u);
    std::uniform_real_distribution<T> distribution(-1.0, 1.0);

   for(int i=0; i<m; ++i) {
      for(int j=0; j<n; ++j) {
         a[j*lda+i] = distribution(generator);
      }
      a[i*lda+i] = T(m);
   }
}

template <typename T>
void print_matrix(int m, T* a, int lda) {
    std::cout << "Printing Matrix: " << std::endl;
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < m; ++j) {
            printf("%8.3f ", a[j*lda+i]);
        }
        printf("\n");
    }
}

template <typename T>
int lu_test (
    int m, // Matrix dimensions
    Opts* opts
    ) {
        //Calculate the number of words required to properly memory-align the matrix
        const int alignment = 64;
        int element_alignment = 0;
        if(alignment % sizeof(T) == 0) {
            element_alignment = alignment / sizeof(T);
        }
        int lda = element_alignment*((m-1)/element_alignment + 1);

        T* a = nullptr;
        T* l = nullptr;
        T* b = nullptr;

        a = new T[(std::size_t) lda*(std::size_t) m]; //Allocate the number of words necessary

        //Generate the matrix
        gen_diagdom_matrix(m, m, a, lda);

        //print_matrix(m, a, lda);

        l = new T[(std::size_t) lda * (std::size_t) m]; //Allocate number of words necessary for the decompose too

        b = new T[m];

        //std::vector<T> x_zero(m, 0.0);

        //remifa::tests::unsym_gen_rhs(m, &a[0], lda, &b[0], &x_zero[0]);

        std::memcpy(l, a, (std::size_t) lda * (std::size_t) m * sizeof(T));

        // Error managment
        cudaError_t cuerr;
        //cublasStatus_t custat;

        // CUDA stream
        cudaStream_t stream;
        cuerr = cudaStreamCreate(&stream);
        checkCudaErrors(cuerr); //Include helper functions

        T* d_l = nullptr;
        half* d_l_f16 = nullptr;

        cublasStatus_t custat; // CuBLAS status
        cublasHandle_t cuhandle;

        int info;
        int *d_inform; // Device side status
        cuerr = cudaMalloc((void**)&d_inform, sizeof(int));
        checkCudaErrors(cuerr);


        //Factorization init
        custat = cublasCreate(&cuhandle);
        cublas_check_error(custat);

        custat = cublasSetStream(cuhandle, stream);
        cublas_check_error(custat);

        custat = cublasSetMathMode(cuhandle, CUBLAS_DEFAULT_MATH);            
        cublas_check_error(custat);

        //Allocated device memory
        cuerr = cudaMalloc((void**)&d_l, (std::size_t) m * lda * sizeof(T));
        checkCudaErrors(cuerr);

        //Copy matrix to device
        custat = cublasSetMatrix(m, m, sizeof(T), l, lda, d_l, lda);
        cublas_check_error(custat);

        cuerr = cudaMalloc((void**)&d_l_f16, (std::size_t)m * lda * sizeof(half));
        checkCudaErrors(cuerr);

        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

        start = std::chrono::high_resolution_clock::now();

        //
        // Kick off device
        //

        //
        // W=T=fp32 and I=fp32 and O=TC32
        // lu_nopiv_rl
        //     <T, 8>
        //     (cuhandle, m, m, nb, d_l, lda, d_inform, d_l_f16, lda);

        decompose_right_looking
            <T, // Working prec
            //remifa::compute_type::FP32, // Inner update compute type
            //remifa::compute_type::FP32, // Outer update compute type
            8 // Inner blocking
            // 256, // outer block size
            >
            (cuhandle, m, m, opts->nb, d_l, lda, d_inform, d_l_f16, lda);
        
        //DO SOMETHING HERE

        // Wait for completion
        cuerr = cudaStreamSynchronize(stream);
        checkCudaErrors(cuerr);

        end = std::chrono::high_resolution_clock::now();

        // Get matrix into host memory      
        custat = cublasGetMatrix(m, m, sizeof(T), d_l, lda, l, lda);
        // cudaMemcpy(l, d_l, lda*m*sizeof(T), cudaMemcpyDeviceToHost);
        cublas_check_error(custat);
        // Get info
        cuerr = cudaMemcpy(&info, d_inform, sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaErrors(cuerr);

        // Cleanup
        custat = cublasDestroy(cuhandle);
        cuerr = cudaFree(d_inform);
        checkCudaErrors(cuerr);

        // Done with device stuff

        cudaDeviceSynchronize();

        long ttotal =  
        std::chrono::duration_cast<std::chrono::nanoseconds>
        (end-start).count();
        double flops = ((double)2.0*m*m*m)/3.0;
        printf("factor time (s) = %e\n", 1e-9*ttotal);
        printf("GFlop/s = %.3f\n", flops/(double)ttotal);

        //print_matrix(m, l, lda);

           // Cleanup memory
        if (nullptr != d_l) {
            cuerr = cudaFree(d_l);
            checkCudaErrors(cuerr);
        }
        
        if (nullptr != d_l_f16) {
            cuerr = cudaFree(d_l_f16);
            checkCudaErrors(cuerr);
        }

        delete[] a;
        delete[] l;
        delete[] b;

        return 0;
}