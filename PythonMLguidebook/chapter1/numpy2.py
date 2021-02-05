# data types in numpy

import numpy as np

list1 = [1, 2, 3]
print(type(list1))  # list
array1 = np.array(list1)
print(type(array1))  # 1-D array
print(array1, array1.dtype)  # dtype: shows the data type of elements in the array

print()

# arrays with different data type elements: automatically set to the largest type
list2 = [1, 2, 'test']
array2 = np.array(list2)
print(array2, array2.dtype)  # unicode string
list3 = [1, 2, 3.0]
array3 = np.array(list3)
print(array3, array3.dtype)  # float 64-bit

print()

# astype(): can change the types of elements in array
array4_int = np.array([1, 2, 3])
print('array4_int: ', array4_int)
array4_float = array4_int.astype('float64')
print('array4_float: ', array4_float)
array5_float = np.array([1.5, 3.1, 4.7])
print('array5_float: ', array5_float)
array5_int = array5_float.astype('int32')
print('array5_int: ', array5_int)
