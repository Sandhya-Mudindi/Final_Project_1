import numpy as np
import time

# Matrix size
N = 500

# Un-optimized Python list-based matrix multiplication
def unOptimized_matrix_mult(A, B):
    result = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Optimized NumPy-based matrix multiplication
def optimized_matrix_mult(A, B):
    return np.dot(A, B)

# Generate random matrices
A_list = [[np.random.rand() for _ in range(N)] for _ in range(N)]
B_list = [[np.random.rand() for _ in range(N)] for _ in range(N)]

A_np = np.array(A_list)
B_np = np.array(B_list)

# Measure time for unoptimized approach
start_time = time.time()
unOptimized_result = unOptimized_matrix_mult(A_list, B_list)
unOptimized_time = time.time() - start_time

# Measure time for optimized approach
start_time = time.time()
optimized_result = optimized_matrix_mult(A_np, B_np)
optimized_time = time.time() - start_time

# Display results
print(f"Un-optimized List-Based Execution Time: {unOptimized_time:.4f} sec")
print(f"Optimized NumPy Execution Time: {optimized_time:.4f} sec")
print(f"Speedup Factor: {unOptimized_time / optimized_time:.2f}x")
