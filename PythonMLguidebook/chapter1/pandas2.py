# Using pandas with Titanic Passenger Data

import pandas as pd

titanic_df = pd.read_csv('C:/Users/minjo/MLdata/titanic/train.csv')

# Index Object
indexes = titanic_df.index
type(indexes)  # index type
indexes
type(indexes.values)  # 1-D numpy array
indexes[:5].values

# reset_index(): resets indexes to 0, 1, 2, ...
# also makes another column with original indexes
value_counts = titanic_df['Pclass'].value_counts()
value_counts_df = value_counts.reset_index()  # returns a dataframe type
value_counts_df

# ------------------------------------------------------------

# *** Data Selection & Filtering
# select several columns
titanic_df[['Pclass', 'Survived']].head(5)

# select rows
titanic_df[0]  # impossible
titanic_df[0:4]  # only slicing is possible when selecting rows
titanic_df[titanic_df['Pclass'] == 3].head(5)  # filtering rows by boolean indexing


data = {'Name': ['Mookie', 'Corey', 'Justin', 'Max', 'Cody'], 'Birth': [1992, 1994, 1984, 1990, 1995], 'Position': ['RF', 'SS', '3B', '1B', 'CF']}
data_df = pd.DataFrame(data, index=['Betts', 'Seager', 'Turner', 'Muncy', 'Bellinger'])
data_df

# df.iloc[] operation - location based indexing
data_df.iloc[2, 0]
data_df.iloc[3:5, :]

# df.loc[] operation - name based indexing
data_df.loc['Betts', 'Position']
data_df.loc['Turner':'Bellinger', ['Birth', 'Position']]

# boolean indexing
titanic_df[titanic_df['Age'] > 60][['Name', 'Age']]
# using several conditions
cond1 = titanic_df['Age'] > 60
cond2 = titanic_df['Sex'] == 'female'
cond3 = titanic_df['Pclass'] == 1
titanic_df[cond1 & cond2 & cond3][['Name', 'Age']]
