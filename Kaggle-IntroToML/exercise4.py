# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = './Kaggle-IntroToML/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")

print("------------------ From previous problems ------------------------")
print()

# Step 1: Split the data (Training vs Validation)
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
step_1.check()

# Step 2: Fit the model with only Training data
iowa_model = DecisionTreeRegressor(random_state = 1)
iowa_model.fit(train_X, train_y)
step_2.check()

# Step 3: Make predictions with validation data
val_predictions = iowa_model.predict(val_X)
step_3.check()

# Step 4: Mean Absolute Error in Validation data
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)
step_4.check()

# Evaluate the model validation results!
print(y.describe())
# mean SalePrice is 180921, and MAE of the model is 29652.