# matrix sorting & computation

import numpy as np

# sort()
array = np.array([3, 1, 9, 5])
print(array)
print(np.sort(array))  # doesn't change 'array' itself
print(np.sort(array)[::-1])  # in descending order
print(array.sort())  # returns nothing, but changes 'array' itself
print(array)
print()

array2 = np.array([[8, 12], [7, 1]])
print(array2)
array2.sort(axis=0)
print(array2)
print()


# argsort(): returns an array of the original indexes when sorted
names = np.array(['Julio', 'David', 'Trevor', 'Clayton', 'Walker', 'Dustin'])
values = np.array([131, 118, 158, 152, 142, 145])
idx_sorted = np.argsort(values)[::-1]  # from the highest to lowest
print(idx_sorted)
print(names[idx_sorted])  # sorted the names based on their own values
print()


# dot product of matrices
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
print(A)
print(B)
print(np.transpose(A))
print(np.dot(A, B))
