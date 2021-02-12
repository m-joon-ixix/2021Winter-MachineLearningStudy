# Functions needed to transform the features (X) of titanic data DF

from sklearn.preprocessing import LabelEncoder

# Deal with Null data
def fill_na(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Cabin'].fillna('N', inplace=True)
    df['Embarked'].fillna('N', inplace=True)
    df['Fare'].fillna(0, inplace=True)
    return df

# Get rid of unnecessary columns
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

# Encode columns with non-numeric data
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']  # string columns to encode
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# Every preprocessing procedure in this function
def transform_features(df):
    df = fill_na(df)
    df = drop_features(df)
    df = format_features(df)
    return df
