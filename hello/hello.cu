#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from the GPU of block %d and thread (%d, %d) !\n", bid, tx, ty);
}

int main()
{
    dim3 blockSize(2, 4);
    hello_from_gpu<<<2, blockSize>>>();
    cudaDeviceSynchronize();
    return 0;
}