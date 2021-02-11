# GridSearchCV: automatically provides the best combination of parameters going into the model when defining the model

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=121)
dTree = DecisionTreeClassifier()

# parameter values to put in - dictionary
parameters = {'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]}

grid_dTree = GridSearchCV(dTree, parameters, cv=3, refit=True)  # refit: fits the model with best parameters at the end
# adapt K-fold Cross Validation method only on the training data
grid_dTree.fit(X_train, y_train)
scores_df = pd.DataFrame(grid_dTree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]

print('Best Combination of Parameters: ', grid_dTree.best_params_)
print('Best Prediction Accuracy Score = ', grid_dTree.best_score_)

# now, let's use the estimator with best parameters on the separate test data
estimator = grid_dTree.best_estimator_
pred = estimator.predict(X_test)
print('Accuracy on Test Data = ', np.round(accuracy_score(y_test, pred), 4))
