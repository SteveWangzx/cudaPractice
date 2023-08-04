#include "../error.cuh"
#include <stdio.h>

/****************************************************************************
* @author: Zhixiang Wang                                                    *
* @brief: Compute the sum for a matrix using GPU                            *
* @date: 2023/8/2                                                           *
****************************************************************************/

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(real *h_x, real *d_x, const int method);
void __global__ reduce_global(real *d_x, real *d_y);
void __global__ reduce_shared(real *d_x, real *d_y);
void __global__ reduce_dynamic(real *d_x, real *d_y);
real reduce(real *d_x, const int method);

int main()
{
    real *h_x = (real *)malloc(M);
    for(int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real* d_x;
    CHECK(cudaMalloc(&d_x, M));
    
    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    printf("\nUsing static shared memory:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing dynamic shared memory:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real* row = d_x + blockDim.x * blockIdx.x;

    for(int offset = 1; offset < blockDim.x; offset *= 2)
    {
        if((tid % (2 * offset))  == 0)
        {
            row[tid] += row[tid + offset];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        d_y[blockIdx.x] = row[0];
    }
}

void __global__ reduce_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if((tid % (stride * 2)) == 0)
        {
            s_y[tid] += s_y[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

void __global__ reduce_dynamic(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if((tid % (stride * 2)) == 0)
        {
            s_y[tid] += s_y[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

real reduce(real *d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *)malloc(ymem);

    switch(method)
    {
        case 0:
            reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;

        case 1:
            reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;

        case 2:
            reduce_dynamic<<<grid_size, BLOCK_SIZE, smem>>>(d_x, d_y);
            break;

        default:
            printf("Error: wrong method!\n");
            exit(1);
            break;
    }

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));
    real result = 0.0;
    for(int i = 0; i < grid_size; ++i)
    {
        result += h_y[i];
    }

    printf("current result = %f\n", result);
    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

void timing(real *h_x, real *d_x, const int method)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(d_x, method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f\n", sum);
}
