#include <stdlib.h>
#include <stdio.h>

/**
 * @brief Adds two vectors A and B and stores the result in vector C.
 *
 * This function performs element-wise addition of two input vectors A and B,
 * and stores the result in the output vector C. All vectors are of length N.
 *
 * @param A Pointer to the first input vector.
 * @param B Pointer to the second input vector.
 * @param C Pointer to the output vector.
 * @param N The number of elements in each vector.
 */
void vectorAddCPU(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}