# How to use pandas

import numpy as np
import pandas as pd

file_path = 'C:/Users/minjo/MLdata/titanic/train.csv'
titanic_df = pd.read_csv(file_path)

titanic_df.head(10)
titanic_df.info()
titanic_df.describe()

titanic_Pclass = titanic_df['Pclass']
value_counts = titanic_Pclass.value_counts()
type(value_counts)

# conversion between numpy array & pandas dataframe ---------------
# array -> df
array1 = np.array([1, 2, 3])
colnames1 = ['numbers']
array1_df = pd.DataFrame(array1, columns=colnames1)
array1_df

array2 = np.array([[1, 2, 3], [11, 22, 33]])
colnames2 = ['ones', 'twos', 'threes']
array2_df = pd.DataFrame(array2, columns=colnames2)
array2_df

# dictionary -> df
dict1 = {'col1': [1, 10], 'col2': [2, 20], 'col3': [3, 30]}
dict1_df = pd.DataFrame(dict1)
dict1_df

# df -> array, list, dict
array3 = dict1_df.values
type(array3)
array3

list3 = array3.tolist()
type(list3)
list3

dict3 = dict1_df.to_dict('list')  # each value of dict is list
type(dict3)
dict3

# --------------------------------------------------

# making new columns
titanic_df['Age10'] = 0  # set all values to 0
titanic_df['Age10'] = titanic_df['Age'] * 10  # same as 'mutate' in R

# deleting rows & columns
titanic_df.drop(['Age10', 'Survived'], axis=1, inplace=True)  # drop columns, 'inplace' means to drop from the dataframe itself
titanic_df.drop([0, 1, 2], axis=0, inplace=True)  # drop rows with index 0, 1, 2
titanic_df
# if 'inplace'=False -> doesn't change df itself, but returns the changed df
