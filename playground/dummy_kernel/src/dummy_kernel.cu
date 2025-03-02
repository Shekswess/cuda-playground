#include <stdio.h>
#include "dummy_kernel.h"

/**
 * @brief A dummy CUDA kernel that prints the block and thread indices.
 */
__global__ void dummyKernel(){
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}