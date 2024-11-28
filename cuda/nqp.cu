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
        (*localCount)++;  // Base case: All rows are filled, we have a valid solution.
        return;
    }

    int safe = allRows & (~(rowMask | ldMask | rdMask)); // Calculate safe positions.
    
    while (safe) {
        int p = safe & (-safe); // Find the rightmost available position.
        safe -= p;              // Remove this position from safe positions.

        // Recursively solve for the next row:
        solveNQueensRecursiveGPU(
            rowMask | p,            // Place the queen in this row at position `p`.
            (ldMask | p) << 1,      // Update the left diagonal mask.
            (rdMask | p) >> 1,      // Update the right diagonal mask.
            n, allRows, localCount  // Continue solving for the next rows.
        );
    }
}


__global__ void nQueens(int n, int *globalCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int rowMask = 1 << idx;
    int ldMask = rowMask << 1;
    int rdMask = rowMask >> 1;

    if (idx >= n) return;

    int localCount = 0;  
    solveNQueensIterativeGPU(rowMask, ldMask, rdMask, n, &localCount);
    
    //int allRows=(1<<n)-1;
    //solveNQueensRecursiveGPU(rowMask, ldMask, rdMask, n, allRows, &localCount);

    atomicAdd(globalCount, localCount); 
}

int main() {
    cout << "+-----+-------------------------+--------------------------------+" << endl;
    cout << "| " << setw(3) << "N" 
              << " | " << setw(23) << "Number of Solutions" 
              << " | " << setw(30) << "Execution Time (seconds)" << " |" << endl;
    cout << "+-----+-------------------------+--------------------------------+" << endl;

    for (int n = 1; n <= 15; n++) {
        int *globalcount;
        cudaMalloc(&globalcount, sizeof(int));
        cudaMemset(globalcount, 0, sizeof(int));  

        auto start = chrono::high_resolution_clock::now();

        int threadsPerBlock = n;  
        int numBlocks = (n + threadsPerBlock - 1) /threadsPerBlock; 

        //this function initiating the execution of a specific function on the GPU from the host (CPU). 
        nQueens<<<numBlocks, threadsPerBlock>>>(n, globalcount);
        cudaDeviceSynchronize();  
        
        //it copying the globalcount from GPU to cpu finalSolutionCount
        int finalSolutionCount;
        cudaMemcpy(&finalSolutionCount, globalcount, sizeof(int), cudaMemcpyDeviceToHost);

        auto end = chrono::high_resolution_clock::now();
        double execution_time = chrono::duration<double>(end - start).count();

        cout << "| " << setw(3) << n 
                  << " | " << setw(23) << finalSolutionCount 
                  << " | " << fixed << setprecision(6)<< setw(29) << execution_time<<"s" << " |" << endl;

        cout << "+-----+-------------------------+--------------------------------+" << endl;

        cudaFree(globalcount); 
    }

    return 0;
}
