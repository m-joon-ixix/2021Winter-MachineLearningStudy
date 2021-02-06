import numpy as np
import pandas as pd

train_data = pd.read_csv('C:/Users/minjo/MLdata/titanic/train.csv')
test_data = pd.read_csv('C:/Users/minjo/MLdata/titanic/test.csv')

# did women really survive more than men?
women = train_data[train_data['Sex'] == 'female']['Survived']
rate_women_survived = sum(women) / len(women)
print("% of women who survived:", rate_women_survived)

men = train_data[train_data['Sex'] == 'male']['Survived']
rate_men_survived = sum(men) / len(men)
print("% of men who survived:", rate_men_survived)
# -------------------------------------

from sklearn.ensemble import RandomForestClassifier

y = train_data['Survived']
features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('titanic_submission.csv', index=False)
print("Submission file successfully generated!")
