#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <chrono>
using namespace std;

__device__ void solveNQueensIterativeGPU(int rowMask, int ldMask, int rdMask, int n, int *localCount) {
    int allRows = (1 << n) - 1;
    int stack[32]; 
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

__device__ void solveNQueensRecursiveGPU(int rowMask, int ldMask, int rdMask, int n, int allRows, int *localCount) {
    if (rowMask == allRows) { 
        (*localCount)++;
        return;
    }

    int safe = allRows & (~(rowMask | ldMask | rdMask));
    
    while (safe) {
        int p = safe & (-safe); 
        safe -= p;            

        solveNQueensRecursiveGPU(
            rowMask | p,            
            (ldMask | p) << 1,      
            (rdMask | p) >> 1,      
            n, allRows, localCount  
        );
    }
}

__global__ void nQueensIterativeKernel(int n, int *globalCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int rowMask = 1 << idx;
    int ldMask = rowMask << 1;
    int rdMask = rowMask >> 1;

    if (idx >= n) return;

    int localCount = 0;  
    solveNQueensIterativeGPU(rowMask, ldMask, rdMask, n, &localCount);

    atomicAdd(globalCount, localCount); 
}

__global__ void nQueensRecursiveKernel(int n, int *globalCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int rowMask = 1 << idx;
    int ldMask = rowMask << 1;
    int rdMask = rowMask >> 1;
    int allRows = (1 << n) - 1;

    if (idx >= n) return;

    int localCount = 0;
    solveNQueensRecursiveGPU(rowMask, ldMask, rdMask, n, allRows, &localCount);

    atomicAdd(globalCount, localCount);
}

int main() {
    cout << "\n+-----+-------------------------+--------------------------------+" << endl;
    cout << "| " << setw(3) << "N" 
         << " | " << setw(23) << "Recursive Solutions" 
         << " | " << setw(30) << "Recursive Time (seconds)" << " |" << endl;
    cout << "+-----+-------------------------+--------------------------------+" << endl;

    for (int n = 1; n <= 15; n++) {
        int *globalCount;
        cudaMalloc(&globalCount, sizeof(int));
        cudaMemset(globalCount, 0, sizeof(int));  

        auto start = chrono::high_resolution_clock::now();

        int threadsPerBlock = n;  
        int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        nQueensRecursiveKernel<<<numBlocks, threadsPerBlock>>>(n, globalCount);
        cudaDeviceSynchronize();

        int recursiveSolutionCount;
        cudaMemcpy(&recursiveSolutionCount, globalCount, sizeof(int), cudaMemcpyDeviceToHost);

        auto end = chrono::high_resolution_clock::now();
        double recursiveTime = chrono::duration<double>(end - start).count();

        cudaFree(globalCount);

        // Output with fixed formatting and desired precision
        cout << "| " << setw(3) << n 
             << " | " << setw(23) << recursiveSolutionCount 
             << " | " << fixed << setprecision(6) << setw(29) << recursiveTime << "s |" << endl;

        cout << "+-----+-------------------------+--------------------------------+" << endl;
    }

    cout << "+-----+-------------------------+-------------------------------+" << endl;
    cout << "| " << setw(3) << "N" 
         << " | " << setw(23) << "Iterative Solutions" 
         << " | " << setw(29) << "Iterative Time (seconds)" << " |" << endl;
    cout << "+-----+-------------------------+-------------------------------+" << endl;

    for (int n = 1; n <= 15; n++) {
        int *globalCount;
        cudaMalloc(&globalCount, sizeof(int));
        cudaMemset(globalCount, 0, sizeof(int));  

        auto start = chrono::high_resolution_clock::now();

        int threadsPerBlock = n;  
        int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        nQueensIterativeKernel<<<numBlocks, threadsPerBlock>>>(n, globalCount);
        cudaDeviceSynchronize();

        int iterativeSolutionCount;
        cudaMemcpy(&iterativeSolutionCount, globalCount, sizeof(int), cudaMemcpyDeviceToHost);

        auto end = chrono::high_resolution_clock::now();
        double iterativeTime = chrono::duration<double>(end - start).count();

        cudaFree(globalCount);

        // Output with fixed formatting and desired precision
        cout << "| " << setw(3) << n 
             << " | " << setw(23) << iterativeSolutionCount 
             << " | " << fixed << setprecision(6) << setw(28) << iterativeTime << "s |" << endl;

        cout << "+-----+-------------------------+-------------------------------+" << endl;
    }

    return 0;
}
