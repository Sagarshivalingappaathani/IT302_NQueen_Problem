#include <iostream>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <iomanip> 
using namespace std;

// Number of solutions found
int numofSol = 0;

// Board size and number of queens
int N;

void placeQ(int queens[], int row, int column) {
    for(int i = 0; i < row; i++) {
        // Check vertical and diagonal threats
        if (queens[i] == column || abs(queens[i] - column) == (row - i)) {
            return;
        }
    }

    // Set the queen
    queens[row] = column;

    if(row == N - 1) {
        // Placed the final queen, found a solution
        #pragma omp atomic
        numofSol++;
    } else {
        // Recursively place queens in the next row
        for(int i = 0; i < N; i++) {
            placeQ(queens, row + 1, i);
        }
    }
}

void solve() {
    #pragma omp parallel
    #pragma omp single
    {
        for(int i = 0; i < N; i++) {
            // New task for the first row and each column recursion
            #pragma omp task
            {
                placeQ(new int[N], 0, i);
            }
        }
    }
}

int main() {
    // Loop over N from 4 to 12
    cout << "+-----+-------------------------+--------------------------------+" << endl;
        cout << "| " << setw(3) << "N"
             << " | " << setw(23) << "Number of Solutions"
             << " | " << setw(30) << "Execution Time (seconds)" << " |" << endl;
        cout << "+-----+-------------------------+--------------------------------+" << endl;
    for (int n = 1; n <= 15; n++) {
        N = n;
        numofSol = 0; // Reset the number of solutions for each N

        // Start timing
        double startTime = omp_get_wtime();

        // Solve the N-Queens problem
        solve();

        // End timing
        double endTime = omp_get_wtime();

         cout << "| " << setw(3) << N
                 << " | " << setw(23) << numofSol
                 << " | " << setw(29) << fixed << setprecision(6) << endTime - startTime << "s" << " |" << endl;
            cout << "+-----+-------------------------+--------------------------------+" << endl;
    }

    return 0;
}

