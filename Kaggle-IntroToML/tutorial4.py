import pandas as pd
melbourne_file_path = "./Kaggle-IntroToML/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis = 0)

y = melbourne_data.Price # going to predict the Price
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X, y)
# --------------------- From Lecture 3 ---------------------

# Compute Mean-Absolute-Error (MAE)
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
print("Training and Validation is done with same data (in-sample data): MAE = ", mean_absolute_error(y, predicted_home_prices))

# Use separate data for fitting & validation (training data & validation data)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# build a new model only using training data
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

val_predictions = melbourne_model.predict(val_X) # predicted data made with validation data (val_X)
print("Training data and Validation data is separated (out-of-sample data): MAE = ", mean_absolute_error(val_y, val_predictions))
