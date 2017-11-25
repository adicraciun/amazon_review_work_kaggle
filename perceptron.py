import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import timeit
import time

dataset = pd.read_csv("amazonReviews.train.csv")
train, test = train_test_split(dataset, test_size=0.30)

features = train.drop('ID', axis=1).drop('Class', axis=1)
labels = train['Class']

bestacc = 0.0
i = 0.1
while i < 1:
    curTime = time.time()
    avgAcc = 0.0
    for _ in range(1):
        clf = Perceptron()
        clf = clf.fit(features, labels)
        avgAcc = avgAcc + accuracy_score(clf.predict(test.drop("ID", axis=1).drop('Class', axis=1)), test['Class'], normalize=True)
        print "ANOTHER"
    avgAcc = avgAcc / 1
    #print "\\bcbar{%.2f}" % avgAcc
    # if (bestacc < acc):
    #     bestclf = clf
    #     bestacc = acc

    nextTime = time.time()
    print "\\bcbar[label=]{%.2f}" % (nextTime - curTime)

    # print i

# clf = bestclf

trainingset = pd.read_csv("amazonReviews.700.test.csv")
id = trainingset['ID']
result = clf.predict(trainingset.drop("ID", axis=1))
result_dataframe = pd.DataFrame(data=result, index=id, columns=["Class"])
print(result_dataframe)
result_dataframe.to_csv("solution.csv")
# print(zip(labels, clf.predict(features)))