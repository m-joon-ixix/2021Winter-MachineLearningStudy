# numpy introduction

import numpy as np

array1 = np.array([1, 2, 3]) # 1-D array
print('array1 type: ', type(array1))
print('array1 array 형태: ', array1.shape)

array2 = np.array([[1, 2, 3], [2, 3, 4]]) # 2x3 array
print('array2 type: ', type(array2))
print('array2 array 형태: ', array2.shape)

array3 = np.array([[1, 2, 3]]) # 1x3 array
print('array3 type: ', type(array3))
print('array3 array 형태: ', array3.shape)
