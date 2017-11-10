import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("amazonReviews.train.csv")
features = dataset.drop('ID', axis=1).drop('Class', axis=1);
labels = dataset['Class']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
trainingset = pd.read_csv("amazonReviews.700.test.csv")
id = trainingset['ID']
result = clf.predict(trainingset.drop("ID", axis=1))
result_dataframe = pd.DataFrame(data=result, index=id, columns=["Class"])
print(result_dataframe)
result_dataframe.to_csv("solution.csv")
# print(zip(labels, clf.predict(features)))