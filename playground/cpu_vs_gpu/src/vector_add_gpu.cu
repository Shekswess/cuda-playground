#include <stdlib.h>
#include <stdio.h>

/**
 * @brief CUDA kernel to perform element-wise addition of two vectors.
 *
 * @param A Pointer to the first input vector.
 * @param B Pointer to the second input vector.
 * @param C Pointer to the output vector.
 * @param N The number of elements in the vectors.
 */
__global__ void vectorAddGPU(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}