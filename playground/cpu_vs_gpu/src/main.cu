#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "vector_add_gpu.h"
#include <time.h>

/**
 * @brief Main function to perform vector addition on GPU.
 *
 * This function initializes two vectors with random values, allocates memory on the GPU,
 * copies the vectors to the GPU, performs vector addition using a CUDA kernel, and then
 * copies the result back to the host. It also measures and prints the time taken for the
 * GPU computation.
 *
 * @return int Returns 0 upon successful execution.
 */
int main(){

    // Define the size of the vectors (100 M elements)
    int N = 100000000;

    float *h_A, *h_B, *h_C;
    size_t size = N * sizeof(float);

    // Allocate memory for the vectors
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize the vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaError_t err;

    err = cudaMalloc(&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        exit(EXIT_FAILURE);
    }

    // Copy the input vectors to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    clock_t start = clock();
    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    clock_t end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    // Copy the result back to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}