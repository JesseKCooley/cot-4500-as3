import numpy as np


def func_problem1(t, y):
    return t - y**2


def euler_method(func, t_range, num_iterations, initial_point):
    t_min, t_max = t_range
    t_values = np.linspace(t_min, t_max, num_iterations + 1)
    y_values = [initial_point[1]]  # Initial y value
    h = (t_max - t_min) / num_iterations  # Step size

    for i in range(num_iterations):
        t = t_values[i]
        y = y_values[-1]
        y += h * func(t, y)
        y_values.append(y)

    return t_values, y_values

def runge_kutta_method(func, t_range, num_iterations, initial_point):
    t_min, t_max = t_range
    t_values = np.linspace(t_min, t_max, num_iterations + 1)
    y_values = [initial_point[1]]  # Initial y value
    h = (t_max - t_min) / num_iterations  # Step size

    for i in range(num_iterations):
        t = t_values[i]
        y = y_values[-1]

        k1 = h * func(t, y)
        k2 = h * func(t + h/2, y + k1/2)
        k3 = h * func(t + h/2, y + k2/2)
        k4 = h * func(t + h, y + k3)

        y += (k1 + 2*k2 + 2*k3 + k4) / 6
        y_values.append(y)

    return t_values, y_values


def gaussian_elimination(A):
    n = len(A)
    U = np.copy(A).astype(float)

    for i in range(n):
        #pivoting: Find the pivot row with maximum absolute value
        max_index = i
        for j in range(i+1, n):
            if abs(U[j, i]) > abs(U[max_index, i]):
                max_index = j
        # Swap rows to bring the row with maximum absolute value to the pivot 
        U[[i, max_index]] = U[[max_index, i]]

        # Elimination step
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            U[j, i:] -= factor * U[i, i:]

    return U

def backward_substitution(U):
    n = len(U)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = (U[i, -1] - np.dot(U[i, i+1:n], x[i+1:])) / U[i, i]

    return x

def lu_factorization(matrix):
    n = len(matrix)
    L = np.eye(n)
    U = np.zeros((n, n))

    # Perform LU factorization
    for k in range(n):
        # Compute U matrix
        for j in range(k, n):
            U[k, j] = matrix[k, j] - np.dot(L[k, :k], U[:k, j])

        # Compute L matrix
        for i in range(k+1, n):
            L[i, k] = (matrix[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    # Calculate determinant
    det = np.prod(np.diag(U)) * (-1) ** np.sum(np.arange(n) % 2 != 0)

    return det, L, U


def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        diagonal_element = abs(matrix[i, i])
        row_sum = np.sum(np.abs(matrix[i, :])) - diagonal_element
        if diagonal_element < row_sum:
            return False
    return True

def is_positive_definite(matrix):
    # Check if the matrix is symmetric
    if not np.array_equal(matrix, matrix.T):
        return False
    
    # Check if all eigenvalues are positive
    eigenvalues = np.linalg.eigvals(matrix)
    if all(eigenvalues > 0):
        return True
    return False