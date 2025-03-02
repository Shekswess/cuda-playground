#include <stdio.h>
#include <cuda_runtime.h>
#include "dummy_kernel.h"

/**
 * @brief Main function that launches the dummy CUDA kernel.
 * 
 * This function prints a message from the CPU, launches the dummy CUDA kernel
 * with a grid of 2 blocks and 4 threads per block, and then synchronizes the device.
 * 
 * @return int Returns 0 upon successful execution.
 */
int main(){
    printf("Hello from CPU!\n");

    dummyKernel<<<2,4>>>();
    cudaDeviceSynchronize();
    return 0;
}