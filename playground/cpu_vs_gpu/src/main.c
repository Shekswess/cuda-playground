#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "vector_add_cpu.h"

/**
 * @brief Main function to perform vector addition on CPU.
 *
 * This function initializes two vectors with random values, performs vector addition
 * using a CPU function, and measures and prints the time taken for the computation.
 *
 * @return int Returns 0 upon successful execution.
 */
int main(){

    // Define the size of the vectors (100 M elements)
    int N = 100000000;

    // Allocate memory for the vectors
    float *A = (float *)malloc(N * sizeof(float));
    float *B = (float *)malloc(N * sizeof(float));
    float *C = (float *)malloc(N * sizeof(float));

    // Initialize the input vectors
    for (int i = 0; i < N; i++){
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    clock_t start = clock();

    // Perform the vector addition
    vectorAddCPU(A, B, C, N);

    clock_t end = clock();

    // Calculate the elapsed time
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    return 0;
}