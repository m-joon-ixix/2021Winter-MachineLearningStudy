# conveniently constructing ndarray & changing shape

import numpy as np

# arange(): set numbers in range(n)
sequence_array = np.arange(10)  # makes up 1-D array with 0 ~ 9
print(sequence_array)
print('dtype: ', sequence_array.dtype, '\t', 'size: ', sequence_array.shape)

# zeros(), ones(): puts in 0s, 1s in each entry of array
zero_array = np.zeros((2, 3), dtype='int32')
print(zero_array)
print('dtype: ', zero_array.dtype, '\t', 'size: ', zero_array.shape)

one_array = np.ones((3, 4))  # default data type: float64
print(one_array)
print('dtype: ', one_array.dtype, '\t', 'size: ', one_array.shape)

print()

# reshape(): changes dimension & shape
print(sequence_array)  # 1-D array, size: 10
sequence_array_25 = sequence_array.reshape(2, 5)  # reshape to 2x5 array
print(sequence_array_25)
# sequence_array.reshape(4, 3) : impossible, error occurs

# using -1 as parameter of reshape(): automatically calculates the proper shape
sequence_array_52 = sequence_array.reshape(-1, 2)
print(sequence_array_52)
# sequence_array.reshape(-1, 3): still impossible

# also can make 3-D array
sequence_array = np.arange(8)
sequence_array_3D = sequence_array.reshape((2, 2, 2))
print(sequence_array_3D.tolist())  # list is better to see in this case

# able to change 3-D into 2-D
sequence_array_2D = sequence_array_3D.reshape(-1, 1)  # 8x1 shape
print(sequence_array_2D.tolist())
