# Checking the Setup - Need to Install Package (learntools) from Kaggle GitHub

from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("---------- Setup Complete ---------")


# Step 1: Loading Data

import pandas as pd

iowa_file_path = "./Kaggle-IntroToML/train.csv"
home_data = pd.read_csv(iowa_file_path)
# check my answer
step_1.check()


# Step 2: Review the Data

# print the summary statistics
print(home_data.describe())

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = 10517

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 2021 - 2010

# check my answer
step_2.check()
