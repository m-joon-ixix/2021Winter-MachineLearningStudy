# indexing or selecting a part from ndarray

import numpy as np

# 1. single indexing
array1 = np.arange(start=1, stop=10)  # 1~9
print(array1)
print(array1[2], array1[-1], array1[-2])
array1_2D = array1.reshape(3, 3)
print(array1_2D)
print(array1_2D[0, 0], array1_2D[0, 1], array1_2D[2, 2])

print()

# 2. Slicing (selecting continuous data elements)
print(array1)
print(array1[0:4])
print(array1[:3])
print(array1[:])

print(array1_2D)
print(array1_2D[1:, :2])  # same as [1:3, 0:2]
print(array1_2D[1], '\t shape:', array1_2D[1].shape)  # row number 1 -> 1-D array

print()

# 3. Fancy Indexing - providing a list of indexes to select
print(array1_2D[[0, 1], 0:2])  # rows 0, 1 / columns 0~1
print(array1_2D[[0, 1]])  # row 0, row 1

print()

# 4. Boolean Indexing - providing a condition inside the brackets
print(array1[array1 > 5])
bool_array = np.array([False, True, True, False, False, True, False, False, True])
print(array1[bool_array])
