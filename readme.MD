<h1>Cuda Programming Practice</h1>
<p>
This repository is a practice records for cuda programming.
The code is based on <strong>《CUDA编程：基础与实践》</strong> by <strong><em>Zheyong Fan.</em></strong>
</p>

The github [repository](https://github.com/brucefan1983/CUDA-Programming) from <strong><em>Zheyong Fan</em></strong>

> A guide compile the <code>.cu</code> file

1. Run terminal as adminstrators  
   
2. CD to directories  
   
3. Using `nvcc` to compile  
`YOUR_PATH/cudaPractice/neighbor> nvcc -o neighbor1cpu neighbor1cpu.cu`  

4.run the generated `.exe` file
`YOUR_PATH/cudaPractice/neighbor> ./neighbor1cpu.exe`  

> Some code example

```cpp
#include <stdio.h>
    
__global__ void kernel(void); // Declaration first.

int main()
{
    kernel<<<1, 4>>>(); // Define grid_size and block_size;
    cudaDeviceSynchronize();
}

__global__ void kernel(void)
{
    int idx = threadIdx.x;
    printf("Hello World from thread %d!\n", idx)；
}
```


