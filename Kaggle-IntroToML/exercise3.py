import pandas as pd
iowa_file_path = "./Kaggle-IntroToML/train.csv"
home_data = pd.read_csv(iowa_file_path)

from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *
print("--- Setup Complete ---")
print()

# Step 1: Specify Prediction Target
print(home_data.columns)
y = home_data.SalePrice # target: sales price
step_1.check()


# Step 2: Create X (features)

# list of features
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# dataframe containing selected features
X = home_data[feature_names]
step_2.check()

# print(X.describe())
# print(X.head())


# Step 3: Fit the Model

# Set up the model
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit the model
iowa_model.fit(X, y)
step_3.check()


# Step 4: Make Predictions
predictions = iowa_model.predict(X)
print(predictions)
step_4.check()


# Compare the predictions and actual values of the target
print("Actual values of prediction target:")
print(y.head())
# Same to each other : because this data(X) was the training data used in fitting this model!