# When making a model based on Decision Trees,
# Overfitting: tree is too deep :: too many groups (leaves)
# Underfitting: tree is too shallow :: too few groups

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

import pandas as pd
melbourne_file_path = "./Kaggle-IntroToML/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis = 0)

y = melbourne_data.Price # going to predict the Price
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# ----------------- From previous problem ---------------------

# Function that computes MAE
def get_MAE(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    val_predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, val_predictions)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_MAE(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))
    