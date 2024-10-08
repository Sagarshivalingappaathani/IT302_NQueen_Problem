#include <iostream>
#include <cuda.h>
#include <chrono>


__device__ void solveNQueensIterativeGPU(int rowMask, int ldMask, int rdMask, int n, int *localCount) {
    int allRows = (1 << n) - 1;
    int stack[100]; 
    int sp = 0;    
    int safe, p;

    stack[sp++] = rowMask;
    stack[sp++] = ldMask;
    stack[sp++] = rdMask;

    while (sp > 0) {
        rdMask = stack[--sp];
        ldMask = stack[--sp];
        rowMask = stack[--sp];

        if (rowMask == allRows) {
            (*localCount)++; 
            continue;
        }

        safe = allRows & (~(rowMask | ldMask | rdMask));
        while (safe) {
            p = safe & (-safe);  
            safe -= p;
            stack[sp++] = rowMask | p;
            stack[sp++] = (ldMask | p) << 1;
            stack[sp++] = (rdMask | p) >> 1;
        }
    }
}

__global__ void nQueensKernel(int n, int *globalCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int rowMask = 1 << idx;
    int ldMask = rowMask << 1;
    int rdMask = rowMask >> 1;

    if (idx >= n) return;

    int localCount = 0;  
    solveNQueensIterativeGPU(rowMask, ldMask, rdMask, n, &localCount);

    atomicAdd(globalCount, localCount); 
}

int main() {
    std::cout << "N       Number of Solutions     Execution Time (seconds)" << std::endl;

    for (int n = 1; n <= 15; n++) {
        int *d_count;
        cudaMalloc(&d_count, sizeof(int));
        cudaMemset(d_count, 0, sizeof(int));  

        auto start = std::chrono::high_resolution_clock::now();

        int blockSize = 1000;  
        int gridSize = (n + blockSize - 1) / blockSize; 

        nQueensKernel<<<gridSize, blockSize>>>(n, d_count);
        cudaDeviceSynchronize();  

        int h_count;
        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double>(end - start).count();

        std::cout << n << "       " << h_count << "                      " << execution_time << std::endl;

        cudaFree(d_count); 
    }

    return 0;
}
