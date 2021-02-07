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
# titanic_df[0]  # impossible
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

# sorting dataframe or series: sort_values()
titanic_sorted = titanic_df.sort_values(by=['Pclass', 'PassengerId'], ascending=False, inplace=False)
titanic_sorted.head(5)

# aggregation functions: min, max, sum, count()
titanic_df.mean()  # mean for each column
titanic_df[['Age', 'Fare']].mean()

# using groupby() on dataframe
titanic_groupby = titanic_df.groupby('Pclass')
type(titanic_groupby)
titanic_groupby.mean()  # computes mean for all columns except 'Pclass'
titanic_groupby[['Age', 'Survived']].mean()  # only for these columns
titanic_groupby['Age'].agg([max, min])  # computes several aggregation functions

agg_format = {'Age': 'max', 'SibSp': 'sum', 'Fare': 'mean'}  # using dictionary, select function for each column
titanic_groupby.agg(agg_format)

# dealing with missing data
titanic_df[['Age', 'Name', 'Fare', 'Survived', 'Cabin']].isna().head(5)  # each entry of DF tells if it's NaN
titanic_df.isna().sum()  # for each column, counts the number of NaN

# filling missing data
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Embarked'].fillna('S', inplace=True)

titanic_df[['Cabin', 'Age', 'Embarked']].head(30)
titanic_df.isna().sum()  # no more NaN left

# using 'lambda' function definition & apply() on each record in DF
titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x : len(x))
titanic_df[['Name', 'Name_len']].head(10)

titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <= 15 else 'Adult')
titanic_df[['Age', 'Child_Adult']].head(10)

def age_category(age):
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenage'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else: cat = 'Elderly'

    return cat

titanic_df['Age_category'] = titanic_df['Age'].apply(lambda x : age_category(x))
titanic_df[['Age', 'Age_category']].head(10)
