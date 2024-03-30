
#import main.assignment_1 as assignment_1
from ..main import assignment3
import numpy as np

# Problem 1
t_range = (0, 2)
num_iterations = 10
initial_point = (0, 1)
t_values, y_values = assignment3.euler_method(assignment3.func_problem1, t_range, num_iterations, initial_point)
print(y_values[-1])
print('\n')

#Problem 2
t_values2, y_values2 = assignment3.runge_kutta_method(assignment3.func_problem1, t_range, num_iterations, initial_point)
print(y_values2[-1])
print('\n')


#Problem 3
# Define the augmented matrix
A = np.array([[2, -1, 1, 6],
              [1, 3, 1, 0],
              [-1, 5, 4, -3]], dtype=float)

# Perform Gaussian elimination
U = assignment3.gaussian_elimination(A)
# Perform backward substitution to find the solution
x = assignment3.backward_substitution(U)
print(x)
print('\n')

#Problem 4
# Define the input matrix
matrix = np.array([[1, 1, 0, 3],
                   [2, 1, -1, 1],
                   [3, -1, -1, 2],
                   [-1, 2, 3, -1]], dtype=float)


det, L, U = assignment3.lu_factorization(matrix)
print(det)
print('\n')
print(L)
print('\n')
print(U)
print('\n')

#Problem 5
matrix2 = np.array([[9,0,5,2,1],
                    [3,9,1,2,1],
                    [0,1,7,2,3],
                    [4,2,3,12,2],
                    [3,2,4,0,8]],dtype=float)

trueFalse = assignment3.is_diagonally_dominant(matrix2)
print(trueFalse)
print('\n')


#Problem 6
matrix3 = np.array([[2,2,1],
                    [2,3,0],
                    [1,0,2]], dtype=float)

trueFalse2 = assignment3.is_positive_definite(matrix3)
print(trueFalse2)