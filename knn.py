import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

dataset = pd.read_csv("amazonReviews.train.csv")
train, test = train_test_split(dataset, test_size=0.20)

features = train.drop('ID', axis=1).drop('Class', axis=1)
labels = train['Class']

for i in range(1,20):
    curTime = time.time()
    avgAcc = 0.0
    # for _ in range(3):
    clf = KNeighborsClassifier(n_neighbors=i, weights="distance")
    clf = clf.fit(features, labels)
    nextTime = time.time()
    avgAcc += accuracy_score(clf.predict(test.drop("ID", axis=1).drop('Class', axis=1)), test['Class'],
                             normalize=True)

    avgAcc = avgAcc / 3
    # print "\\bcbar[label=Number of neighbors: %d]{%.2f}" % (i, avgAcc)
    # if (bestacc < acc):
    #     bestclf = clf
    #     bestacc = acc


    print "\\bcbar[text=Number of neighbors: %d]{%.2f}" % (i, nextTime - curTime)

    if i < 100:
        i += 2
    elif i < 400:
        i += 100
    else:
        i += 200

trainingset = pd.read_csv("amazonReviews.700.test.csv")
id = trainingset['ID']
result = clf.predict(trainingset.drop("ID", axis=1))
result_dataframe = pd.DataFrame(data=result, index=id, columns=["Class"])
print(result_dataframe)
result_dataframe.to_csv("solution.csv")
# print(zip(labels, clf.predict(features)))
