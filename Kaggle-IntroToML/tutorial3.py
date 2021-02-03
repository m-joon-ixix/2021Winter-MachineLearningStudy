import pandas as pd
melbourne_file_path = "./Kaggle-IntroToML/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

print(melbourne_data.columns)
print()

# Drop the rows containing NA data
melbourne_data = melbourne_data.dropna(axis = 0)

# Select Prediction Target (single column dataframe)
y = melbourne_data.Price # going to predict the Price

# Features for prediction (columns that are used to predict the target)
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]

print(X.describe())
print()

# Defining your model: Define -> Fit -> Predict -> Evaluate
from sklearn.tree import DecisionTreeRegressor

# 1. Define model
melbourne_model = DecisionTreeRegressor(random_state = 1)
# 2. Fit model with whole training data
melbourne_model.fit(X, y)
# 3. Making predictions
print("Making predictions for the first 5 houses:")
print(X.head())
print("The predictions are:")
print(melbourne_model.predict(X.head()))
