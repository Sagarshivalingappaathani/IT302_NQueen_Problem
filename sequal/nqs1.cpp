#include <bits/stdc++.h>
#include <chrono>  // For time calculation

using namespace std;

class Solution {
  public:
    bool isSafe1(int row, int col, vector<string> board, int n) {
      int duprow = row;
      int dupcol = col;

      // check upper diagonal
      while (row >= 0 && col >= 0) {
        if (board[row][col] == 'Q') return false;
        row--;
        col--;
      }

      col = dupcol;
      row = duprow;

      // check left row
      while (col >= 0) {
        if (board[row][col] == 'Q') return false;
        col--;
      }

      row = duprow;
      col = dupcol;

      // check lower diagonal
      while (row < n && col >= 0) {
        if (board[row][col] == 'Q') return false;
        row++;
        col--;
      }

      return true;
    }

  public:
    void solve(int col, vector<string> &board, vector<vector<string>> &ans, int n) {
      if (col == n) {
        ans.push_back(board);
        return;
      }
      for (int row = 0; row < n; row++) {
        if (isSafe1(row, col, board, n)) {
          board[row][col] = 'Q';
          solve(col + 1, board, ans, n);
          board[row][col] = '.';
        }
      }
    }

  public:
    vector<vector<string>> solveNQueens(int n) {
      vector<vector<string>> ans;
      vector<string> board(n, string(n, '.'));
      solve(0, board, ans, n);
      return ans;
    }
};

int main() {
  Solution obj;

  // Print table header
  cout << "+-----+-------------------------+----------------------------------+\n";
  cout << "|  n  | Number of Solutions     | Time Taken (seconds)             |\n";
  cout << "+-----+-------------------------+----------------------------------+\n";

  // Loop from 1 to 15
  for (int n = 1; n <= 15; n++) {
    // Start time measurement
    auto start = chrono::high_resolution_clock::now();

    // Get the solutions
    vector<vector<string>> ans = obj.solveNQueens(n);

    // End time measurement
    auto end = chrono::high_resolution_clock::now();

    // Calculate the duration in seconds with high precision
    chrono::duration<double> duration = end - start;

    // Output each row in the required format
    cout << "| " << setw(3) << n << " | " << setw(23) << ans.size() 
         << " | " << setw(30) << fixed << setprecision(6) << duration.count() << "s  |\n";
    cout << "+-----+-------------------------+----------------------------------+\n";
  }

  return 0;
}
