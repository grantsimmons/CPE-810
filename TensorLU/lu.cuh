#pragma once

   #define BLOCKS 1
   #define BLOCK_SIZE 8 //Thread block size
   
   template<typename T, int ib=BLOCK_SIZE>
   void lu_nopiv_panel(
         const cudaStream_t stream,
         int m, int n,
         T const *const d, int ldd,
         T *l, int ldl,
         T *u, int ldu,
         int *stat);