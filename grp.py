import matplotlib.pyplot as plt # type: ignore

# Sample data (replace with actual values)
N_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# Number of solutions for each method
sequential_solutions = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184]
openmp_solutions = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184]
mpi_solutions = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184]
cuda_solutions = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184]

# Execution times for each method
sequential_times = [0.0000165, 0.0000065, 0.0000205, 0.0000685, 0.0002075, 0.0008375, 0.0022985, 0.0049265, 
                    0.0238425, 0.1023175, 0.5861275, 3.520360, 22.5312945, 146.4018095, 988.893746]
openmp_times = [0.025739, 0.006968, 0.004969, 0.006995, 0.006992, 0.006979, 0.006989, 0.005990, 
                0.004534, 0.004948, 0.010397, 0.047028, 0.255498, 1.699859, 11.923959]
mpi_times = [0.000057, 0.000001, 0.000001, 0.000031, 0.000004, 0.000008, 0.000025, 0.000101, 0.000587, 0.002634, 0.012700, 
             0.068830, 0.357723, 2.217376, 15.556984]
cuda_times = [0.000135, 0.000024, 0.000022, 0.000024, 0.000035, 0.000061, 0.000151, 0.000455, 
              0.001673, 0.006234, 0.012879, 0.042400, 0.188004, 0.959638, 5.746145]

# Plot N vs Number of Solutions
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(N_values, sequential_solutions, label='Sequential', marker='o')
plt.plot(N_values, openmp_solutions, label='OpenMP', marker='s')
plt.plot(N_values, mpi_solutions, label='MPI', marker='^')
plt.plot(N_values, cuda_solutions, label='CUDA', marker='x')
plt.xlabel('N')
plt.ylabel('Number of Solutions')
plt.title('N vs Number of Solutions')
plt.legend()
plt.grid(True)

# Plot N vs Execution Time
plt.subplot(1, 2, 2)
plt.plot(N_values, sequential_times, label='Sequential', marker='o')
plt.plot(N_values, openmp_times, label='OpenMP', marker='s')
plt.plot(N_values, mpi_times, label='MPI', marker='^')
plt.plot(N_values, cuda_times, label='CUDA', marker='x')
plt.xlabel('N')
plt.ylabel('Execution Time (seconds)')
plt.title('N vs Execution Time')
plt.yscale('log')  # Log scale for better visibility of time differences
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('graph.png')
 # type: ignore