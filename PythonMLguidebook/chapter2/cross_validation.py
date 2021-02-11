# 1. K-fold Cross Validation

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# Define a K-fold validation model to spilt the data into K datasets
kfold = KFold(n_splits=5)
accuracies = []
print('Number of Records: ', features.shape[0])

times = 0
# kfold.split() - returns the row indexes of train_data, test_data each in a tuple
for train_idx, test_idx in kfold.split(features):
    # for each (train, test) idx combination
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    times += 1
    # compute the accuracy and put each record in list
    accuracy = np.round(accuracy_score(y_test, pred), 4)  # rounded to 4th decimal place
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('Validation #{0} accuracy = {1}, train data size = {2}, test data size = {3}'.format(times, accuracy, train_size, test_size))
    print('Test data set index: ', test_idx)
    print()
    accuracies.append(accuracy)

print('Average Accuracy of this K-fold Cross Validation = ', np.mean(accuracies))


# 2. Stratified K-fold Validation
from sklearn.model_selection import StratifiedKFold

# Define a Stratified K-fold validation model to spilt the data
# This makes the label distribution of each training dataset identical to the whole label data distribution
dt_clf = DecisionTreeClassifier(random_state=156)
skFold = StratifiedKFold(n_splits=3)
accuracies = []
times = 0

# skFold.split() - also needs label data as parameter to consider the label distribution
for train_idx, test_idx in skFold.split(features, label):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    times += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('Validation #{0} accuracy = {1}, Train data size: {2}, Test data size: {3}'.format(times, accuracy, train_size, test_size))
    print('Test data index: ', test_idx)
    accuracies.append(accuracy)

print('Average Accuracy of Stratified K-fold Validation = ', np.round(np.mean(accuracies), 4))


# 3. cross_val_score(): does all the tasks above, and returns an array with evaluation scores
from sklearn.model_selection import cross_val_score, cross_validate
scores = cross_val_score(dt_clf, features, label, cv=3, scoring='accuracy')  # cv: number of cross-val folds
print('Score for each Validation: ', np.round(scores, 4))
print('Average Score = ', np.round(np.mean(scores), 4))
