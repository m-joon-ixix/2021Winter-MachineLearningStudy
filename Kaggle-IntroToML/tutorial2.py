print()

import pandas as pd

melbourne_file_path = "C:/Users/minjo/Documents/GitHub/2021Winter-MachineLearningStudy/Kaggle-IntroToML/melb_data.csv"

melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())
