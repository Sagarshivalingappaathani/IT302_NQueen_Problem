#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>  // For formatting the output

using namespace std;

int N;  // Board size and number of queens

// Function to check if placing a queen in the column is safe
bool isSafe(int queens[], int row, int column) {
    for (int i = 0; i < row; i++) {
        if (queens[i] == column || abs(queens[i] - column) == abs(row - i)) {
            return false;
        }
    }
    return true;
}

// Recursive function to solve N-Queens problem
void placeQ(int queens[], int row, int &localSol) {
    if (row == N) {
        // Found a solution, increment the local solution count
        localSol++;
        return;
    }

    for (int col = 0; col < N; col++) {
        if (isSafe(queens, row, col)) {
            queens[row] = col;
            placeQ(queens, row + 1, localSol);
        }
    }
}

void solve(int rank, int size, int &localSol) {
    // Each process will work on specific columns in the first row
    for (int col = rank; col < N; col += size) {
        int *queens = new int[N];
        queens[0] = col;  // Place queen in the first row, column = col
        placeQ(queens, 1, localSol);  // Recursively solve for next rows
        delete[] queens;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print table header in root process (rank 0)
    if (rank == 0) {
        cout << "+-----+-------------------------+--------------------------------+" << endl;
        cout << "| " << setw(3) << "N"
             << " | " << setw(23) << "Number of Solutions"
             << " | " << setw(30) << "Execution Time (seconds)" << " |" << endl;
        cout << "+-----+-------------------------+--------------------------------+" << endl;
    }

    // Loop over N from 4 to 15
    for (int n = 4; n <= 15; n++) {
        N = n;
        int localSol = 0;  // Number of solutions found by this process
        int globalSol = 0; // Total number of solutions across all processes

        // Start timing
        double startTime = MPI_Wtime();

        // Solve the N-Queens problem in parallel
        solve(rank, size, localSol);

        // Reduce the local solutions into global total solutions
        MPI_Reduce(&localSol, &globalSol, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // End timing
        double endTime = MPI_Wtime();

        // Output the results from the root process (rank 0)
        if (rank == 0) {
            cout << "| " << setw(3) << n
                 << " | " << setw(23) << globalSol
                 << " | " << setw(29) << fixed << setprecision(6) << endTime - startTime << "s" << " |" << endl;
            cout << "+-----+-------------------------+--------------------------------+" << endl;
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
