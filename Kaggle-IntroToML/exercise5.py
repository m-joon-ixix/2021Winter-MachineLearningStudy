# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = './Kaggle-IntroToML/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex5 import *

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

print("\nSetup complete")
print()
# --------------- from previous problems ----------------

# Step 1: Compare different tree sizes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

min_MAE = get_mae(candidate_max_leaf_nodes[0], train_X, val_X, train_y, val_y)
best_tree_size = candidate_max_leaf_nodes[0]

for max_number in candidate_max_leaf_nodes:
    this_mae = get_mae(max_number, train_X, val_X, train_y, val_y)
    if this_mae < min_MAE:
        min_MAE = this_mae
        best_tree_size = max_number

step_1.check()


# Step 2: Fit the model using all data, with best tree size
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size)
final_model.fit(X, y) # using all data, not only training data

step_2.check()
