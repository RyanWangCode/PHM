import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


dataIn15 = pd.read_csv('train/15/15_data.csv')
normalPer15 = pd.read_csv('train/15/15_normalInfo.csv')
failurePer15 = pd.read_csv('train/15/15_failureInfo.csv')
timeIn15 = dataIn15['time']
dataIn21 = pd.read_csv('train/21/21_data.csv')
normalPer21 = pd.read_csv('train/21/21_normalInfo.csv')
failurePer21 = pd.read_csv('train/21/21_failureInfo.csv')
timeIn21 = dataIn21['time']
dataIn08 = pd.read_csv('test/08/08_data.csv')

# for i in range(len(timeIn15)):
#     timeIn15[i] = dt.datetime.strptime(timeIn15[i], '%Y-%m-%d %H:%M:%S')

# for i in range(len(normalPer15)):
#     normalPer15.loc[i, 'startTime'] = dt.datetime.strptime(normalPer15.loc[i, 'startTime'], '%Y-%m-%d %H:%M:%S')
#     normalPer15.loc[i, 'endTime'] = dt.datetime.strptime(normalPer15.loc[i, 'endTime'], '%Y-%m-%d %H:%M:%S')


def dataPreProcess(dataIn, timeIn, normalPer, failurePer):
    dataIn['label'] = -1
    for i in range(len(timeIn)):
        for j in range(len(normalPer)):
            if timeIn[i] >= normalPer.loc[j]['startTime'] and timeIn[i] >= normalPer.loc[j]['endTime']:
                dataIn['label'] = 0
        for j in range(len(failurePer)):
            if timeIn[i] >= failurePer.loc[j]['startTime'] and timeIn[i] >= failurePer.loc[j]['endTime']:
                dataIn['label'] = 1
    return dataIn


def featProcess(dataIn):
    pass


data15 = dataPreProcess(dataIn15, timeIn15, normalPer15, failurePer15)
data21 = dataPreProcess(dataIn21, timeIn21, normalPer21, failurePer21)


dataX = featProcess()
datay =
testX = featProcess(dataIn08)

k_range = range(10, 201)
k_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    cross_val_score(knn, dataX, datay, cv=10, scoring='accuracy')

plt.plot(k_range, k_score)
k_best = k_range[np.array(k_score).argmax()]


