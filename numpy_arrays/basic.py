"""
Python allows us to create arrays of all kinds of data types and dimensions. We
can do this using the built-in list functionality, but this quickly
becomes very tiresome. Instead, let's use NumPy.

We can create a NumPy array by starting with a python list, and then wrapping
it with the np.array() function.
"""

import numpy as np

regular_python_array = [1, 2, 3, 4, 5]
numpy_array = np.array(regular_python_array)

regular_python_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
numpy_array = np.array(regular_python_array)

"""
NumPy also provides us with some useful functions for creating arrays. For
example, we can create an array of zeros, ones, or a range of numbers. The
np.zeros() function creates an array of zeros, the np.ones() function creates
an array of ones, and the np.arange() function creates an array of numbers in a
given range (like [0, 1, 2, 3, 4]). And, thanks to NumPy, we can perform
operations on these arrays to impact all the values at once.
"""

# fmt:off
zeros_array = np.zeros(5)          # [0, 0, 0, 0, 0]
ones_array  = np.ones(5)           # [1, 1, 1, 1, 1]
fives_array = np.ones(5) * 5       # [5, 5, 5, 5, 5]
range_array = np.arange(5)         # [0, 1, 2, 3, 4]
range_array = np.arange(5, 10)     # [5, 6, 7, 8, 9]
range_array = np.arange(5, 10, 2)  # [5, 7, 9]
# fmt:on

"""
We don't need to make just one-dimensional arrays. We can also create
multi-dimensional arrays in a bunch of different ways. For example, we can
create a multi-dimensional array by passing a list of lists to the np.array(),
or we can use the np.zeros() and np.ones() functions to create multi-dimensional
arrays of zeros and ones. We just have to specify the shape of the array we want
as a tuple.
"""

# fmt:off
multi_dimensional_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
zeros_array = np.zeros((3, 4))    # 3 rows, 4 columns of zeros
ones_array = np.ones((3, 4))      # 3 rows, 4 columns of ones
fives_array = np.ones((3, 4)) * 5 # 3 rows, 4 columns of fives
# fmt:on

"""
We can access the elements of this array in similar ways to how we access the
elements of a regular python list. We can use the index of the element we want
to access, and we can use negative indices to access elements from the end of
the array. We can also use slicing to access a range of elements in the array.

Where numpy arrays are really helpful here, though, is that we can access
multidimensional elements using a comma-separated tuple of indices. We can also
use slicing to access a range of elements in a multi-dimensional array.
"""

# fmt:off
array = np.array([1, 2, 3, 4, 5])
print(array[0])    # 1
print(array[-1])   # 5
print(array[1:4])  # [2, 3, 4]

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(array[0, 0])       # 1
print(array[1, 2])       # 6
print(array[1:3, 1:3])   # [[5, 6], [8, 9]]
# fmt:on

"""
Another great thing about NumPy is that we can perform operations on arrays
that impact all the values at once. For example, we can add two arrays together
element-wise, multiply an array by a scalar, or perform matrix multiplication
using the np.dot() function.
"""

array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([6, 7, 8, 9, 10])
result_array_add = array1 + array2
result_array_multiply = array1 * 2

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result_matrix_multiply = np.dot(matrix1, matrix2)

"""
There are also a handful of techniques to reshape and manipulate arrays.
One of the most common is transposing an array, which switches the rows and
columns of a multi-dimensional array. We can also reshape an array to a new
shape, flatten an array to a single dimension, and stack arrays vertically or
horizontally.
"""

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

transposed_array = array.T
reshaped_array = array.reshape(1, -1)  # using -1 infers the number of columns
flattened_array = array.flatten()

vertical_stack = np.vstack((array, transposed_array))
horizontal_stack = np.hstack((array, transposed_array))

"""
NumPy also provides us with some handy helper functions for performing
statistical operations on arrays. We can calculate the mean, sum, max, and min
of an array using the np.mean(), np.sum(), np.max(), and np.min() functions.
"""

array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array_mean = np.mean(array)
array_sum = np.sum(array)
array_max = np.max(array)
array_min = np.min(array)

"""
We can also get kinda fancy with NumPy and perform operations along rows and
columns of multi-dimensional arrays. We can sum along rows and columns using
the axis parameter, and we can perform element-wise operations like squaring
each element of an array.
"""
# Sum along rows and columns
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_sum = np.sum(array_2d, axis=1)
col_sum = np.sum(array_2d, axis=0)
array_squared = np.square(array_2d)
print("Sum along rows:", row_sum)
print("Sum along columns:", col_sum)
print("Squared array:", array_squared)
