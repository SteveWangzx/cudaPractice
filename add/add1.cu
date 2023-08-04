#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*********************************************
* @author: Zhixiang Wang                     *
* @brief: example for large 1D matrix add    *
* @date: 2023/8/1                            *
*********************************************/

struct Matrix {
    int x;
    int y;
    int z;
};

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
__global__ void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *host_x = (double*)malloc(M);
    double *host_y = (double*)malloc(M);
    double *host_z = (double*)malloc(M);

    for(int n = 0; n < N; ++n)
    {
        host_x[n] = a;
        host_y[n] = b;
    }

    double *dev_x, *dev_y, *dev_z;
    cudaMalloc((void**)&dev_x, M);
    cudaMalloc((void**)&dev_y, M);
    cudaMalloc((void**)&dev_z, M);

    cudaMemcpy(dev_x, host_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, host_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N / block_size) + 1;
    add<<<grid_size, block_size>>>(dev_x, dev_y, dev_z, N);

    cudaMemcpy(host_z, dev_z, M, cudaMemcpyDeviceToHost);
    check(host_z, N);

    free(host_x);
    free(host_y);
    free(host_z);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
}

__global__ void add(const double *x, const double *y, double *z, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N)
    {
        z[idx] = x[idx] +y[idx];
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for(int n = 0; n < N; ++n)
    {
        if(fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }

    printf("%s\n", has_error ? "has errors" : " no errors");
}
