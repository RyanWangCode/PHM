# -*- coding: utf-8 -*-
"""
Created on April 16 2018

@author: WangRui
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# 归一化
def normalize(Arr):
    m, n = Arr.shape
    maxArr = Arr.max(0)
    minArr = Arr.min(0)
    outArr = (Arr - minArr) / (maxArr - minArr)
    return outArr

# 计算余弦相似度
def corr(Arr, Vec):
    m, n = Arr.shape
    corrOut = np.zeros(m)
    num = (Arr * Vec).sum(1)
    for i in range(m):
        den = np.linalg.norm(Arr[i]) * np.linalg.norm(Vec)
        corrOut[i] = num[i] / den
    return corrOut

# 根据相似度计算最近邻，k=5
def findKNN5(Vec):
    knn5 = []
    tempList = list(Vec)
    indexList = list(range(200))
    for i in range(5):
        index = tempList.index(max(tempList))
        knn5.append(indexList[index])
        del(tempList[index])
        del(indexList[index])
    return knn5

# 计算指定试验中最相似的路段，计算速度较慢，有待优化，暂不使用
def calCorrW(KNN5, Arr):
    disBest = 10000
    iBest = 0
    jBest = 0
    for i in range(5):
        m, n = dataTraining[int(KNN5[i])].shape
        for j in range(m - 49):
            disbList = []
            for b in range(6, 10):
                disSum = 0
                for k in range(50):
                    disSum += pow(dataTraining[int(KNN5[i])].loc[j+k]['f_126'] - Arr.loc[k]['f_126'], 2)
                disbList.append(disSum)
            disMean = np.mean(disbList)
            if disMean < disBest:
                disBest = disMean
                iBest = i
                jBest = j
    return iBest, jBest

# 特征处理
def fetProcess(Arr):
    outArr = np.zeros(155)
    posArr = []
    for i in range(5):
        outArr[0 + i] = Arr[0 + i] / Arr[5 + i]; posArr.append([0, 4])
        outArr[5 + i] = Arr[0 + i] / Arr[10 + i]; posArr.append([1, 5])
        outArr[10 + i] = Arr[0 + i] / Arr[15 + i]; posArr.append([0, 6])
        outArr[15 + i] = Arr[0 + i] / Arr[20 + i]; posArr.append([1, 7])
        outArr[20 + i] = Arr[45 + i] / Arr[50 + i]; posArr.append([2, 8])
        outArr[25 + i] = Arr[45 + i] / Arr[55 + i]; posArr.append([3, 9])
        outArr[30 + i] = Arr[45 + i] / Arr[60 + i]; posArr.append([2, 10])
        outArr[35 + i] = Arr[45 + i] / Arr[65 + i]; posArr.append([3, 11])
        outArr[40 + i] = Arr[5 + i] / Arr[25 + i]; posArr.append(4)
        outArr[45 + i] = Arr[10 + i] / Arr[30 + i]; posArr.append(5)
        outArr[50 + i] = Arr[15 + i] / Arr[35 + i]; posArr.append(6)
        outArr[55 + i] = Arr[20 + i] / Arr[40 + i]; posArr.append(7)
        outArr[60 + i] = Arr[50 + i] / Arr[70 + i]; posArr.append(8)
        outArr[65 + i] = Arr[55 + i] / Arr[75 + i]; posArr.append(9)
        outArr[70 + i] = Arr[60 + i] / Arr[80 + i]; posArr.append(10)
        outArr[75 + i] = Arr[65 + i] / Arr[85 + i]; posArr.append(11)
        outArr[80 + i] = Arr[10 + i] / Arr[50 + i]; posArr.append([4, 8])
        outArr[85 + i] = Arr[15 + i] / Arr[55 + i]; posArr.append([5, 9])
        outArr[90 + i] = Arr[20 + i] / Arr[60 + i]; posArr.append([6, 10])
        outArr[95 + i] = Arr[25 + i] / Arr[65 + i]; posArr.append([7, 11])
        outArr[100 + i] = Arr[0 + i] / Arr[45 + i]; posArr.append([0, 1, 2, 3])
        outArr[105 + i] = Arr[0 + i]; posArr.append([0, 1])
        outArr[110 + i] = Arr[45 + i]; posArr.append([2, 3])
        outArr[115 + i] = Arr[5 + i]; posArr.append(4)
        outArr[120 + i] = Arr[10 + i]; posArr.append(5)
        outArr[125 + i] = Arr[15 + i]; posArr.append(6)
        outArr[130 + i] = Arr[20 + i]; posArr.append(7)
        outArr[135 + i] = Arr[50 + i]; posArr.append(8)
        outArr[140 + i] = Arr[55 + i]; posArr.append(9)
        outArr[145 + i] = Arr[60 + i]; posArr.append(10)
        outArr[150 + i] = Arr[65 + i]; posArr.append(11)
    return outArr, posArr

def loess1(testPoint, xArr, yArr, k = 1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = xMat.shape[0]
    weights = np.mat(np.eye((m)))
    print(m, xArr.shape)
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))      # 计算权重对角矩阵
    xTx = xMat.T * (weights * xMat)                                 # 奇异矩阵不能计算
    theta = xTx.I * (xMat.T * (weights * yMat))                     # 计算回归系数
    print(theta)
    return testPoint * theta

# 局部加权回归
def loess(testArr, xArr, yArr, k = 1.0):
    m = testArr.shape[0]
    yOut = np.zeros(m)
    for i in range(m):
        yOut[i] = loess1(testArr[i], xArr, yArr, k)
    return yOut

# 数据导入
dataTraining = []
dataMedian = np.zeros([200, 90])
dataTesting = []
dataMedianTesting = np.zeros([200, 90])
conditionTraining = pd.read_csv('training.csv')
conditionTesting = pd.read_csv('testing.csv')
for i in range(1,201):
    dataTraining.append(pd.read_csv('training/Experiment-%i.csv' % i))
    dataTesting.append(pd.read_csv('testing/Experiment-%i.csv' % i))
    dataMedian[i-1] = np.median(dataTraining[i-1], 0)
    dataMedianTesting[i-1] = np.median(dataTesting[i-1], 0)

# 固定速度下负载大小与振动响应之间关系的分析，可以看出负载与振动响应有较大的相关性，可用回归分析进行训练
azs1 = dataMedian[:, :5]
azp1 = (dataMedian[:, 5:10] + dataMedian[:, 10:15]) / 2
az1 = (dataMedian[:, 25:30] + dataMedian[:, 30:35]) / 2
condS126 = conditionTraining[conditionTraining['Speed'] == 1.26]
dataS126 = []
for i in condS126['ExperimentID']:
    dataS126.append(dataMedian[i])

dataS126 = np.array(dataS126)

# 根据相似度提取训练集数据
conditionTrainingArr = np.array(conditionTraining.loc[:, ['Track', 'Payload', 'Speed']])
conditionTestingArr = np.array(conditionTesting.loc[:, ['Track', 'Payload', 'Speed']])
corrArr = np.zeros([200, 200])
corrBestList5 = np.zeros([200, 5])
for i in range(200):
    corrArr[i] = corr(conditionTrainingArr, conditionTestingArr[i])
    corrBestList5[i] = findKNN5(corrArr[i])

# corrBestList = np.zeros([200, 2])
# dataMeanBest = np.zeros([200, 90])
# for i in range(200):
#     corrBestList[i] = calCorrW(corrBestList5[i], dataTesting[i])
#     dataMeanBest[i] = np.mean(dataTraining[int(corrBestList[i][0])].loc  \
#                                   [int(corrBestList[i][1]): int(corrBestList[i][1]) + 50], 0)

# 训练集、测试集数据处理
labelTraining = np.zeros([200, 155])
labelTesting = np.zeros([200, 155])
loadTraining = np.zeros([200, 1])
loadTesting = np.zeros([200, 1])
fetPos = []
for i in range(200):
    labelTraining[i], tmp = fetProcess(dataMedian[int(corrBestList5[i][0])])
    labelTesting[i], tmp = fetProcess(dataMedianTesting[i])
    loadTraining[i] = conditionTraining.loc[int(corrBestList5[i][0])]['Payload']
    loadTesting[i] = conditionTesting.loc[i, ['Payload']]
    if i == 199: fetPos = tmp

# 训练回归模型，以线性模型简化，局部加权可训练出均方误差更小的模型
linregList = []
for i in range(155):
    linreg = LinearRegression()
    linreg.fit(loadTraining, labelTraining[:, i])
    linregList.append(linreg)

# 计算置信区间
yPredict = np.zeros([200, 155])
resRatio = np.zeros([200, 155])
tval = stats.t.isf(0.01 / 2, 200 - 2)
for i in range(200):
    for j in range(155):
        yPredict[i, j] = linregList[j].predict([loadTesting[i]])
        pstd = np.sqrt(sum((linregList[j].predict(loadTraining) - labelTraining[:, j].reshape(200)) ** 2) / (200 - 2))
        sxx = np.sum(loadTesting ** 2) - np.sum(loadTesting) ** 2 / 200  # np.var(loadTraining) * 200
        interval = tval * pstd * np.sqrt(1 + 1 / 200 + (loadTesting[i] - np.mean(loadTesting)) ** 2 / sxx)
        resRatio[i, j] = abs(yPredict[i, j] - labelTesting[i, j]) / interval

# 故障试验的判定
# 由于没有测试集准确率数据，根据参赛队员论文描述可知约有一半的测试集数据是故障试验，所以设定阈值超参数，根据故障
# 试验数量来选择阈值从而判定故障试验。
resThreshold = 1
resRatioList = []
for i in np.linspace(0.01, 2, 200):
    resThreshold = 1 + i
    resRatioList.append(sum(np.max(resRatio, 1) > resThreshold))

threshold = 1 + 0.01 * abs(np.array(resRatioList) - 100).argmin()

# 定位故障位置
# 由问题描述可知，故障试验中的故障位置有1～2个，如果某一试验中某一特征的偏离度大于阈值则在此特征对应的故障位置上
# 累加此偏离度，之后针对每一次试验按故障位置的累加偏离度排序，在故障试验中累加偏离度最高的位置可判定为故障位置1，
# 对故障试验中累加偏离度第二大的位置做聚类分析，从而将一部分划分为故障位置2。
votScore = np.zeros([200, 12])
for i in range(200):
    if np.max(resRatio[i]) > threshold:
        for j in range(155):
            if resRatio[i][j] > threshold:
                votScore[i][fetPos[j // 5]] += resRatio[i][j]
votSortIdx = np.argsort(votScore, 1)
votSort = np.sort(votScore, 1)


canList = []
canIdxList = []
for i in range(200):
    if votSort[i][-1] > 0:
        canList.append(votSort[i])
        canIdxList.append(i)

canArr = np.array(canList)
# print(canList)
clu_pred = KMeans(n_clusters=3, random_state=0).fit_predict(np.concatenate((canArr[:, -2].reshape([len(canList), 1]), canArr[:, -2].reshape([len(canList), 1])), 1))

print(clu_pred)
plt.figure(0)
colorList = ['b', 'g', 'r']
print([colorList[idx] for idx in clu_pred])
plt.scatter(canArr[:, -2], canArr[:, -2], c=[colorList[idx] for idx in clu_pred])

doubleFault = np.zeros(200)
for i in range(len(canIdxList)):
    if clu_pred[i] >= 1:
        doubleFault[int(canIdxList[i])] = 1
# 故障部件类型判定
# 判别故障部件，基于悬架简化的二自由度振动模型，分析传递函数的幅频曲线可知：当故障位置为上层悬架时，阻尼影响高频响应
# 弹簧影响低频响应；当故障为下层悬架时，阻尼不改变固有频率但影响共振响应峰值，而弹簧会改变固有频率。由上述分析可定位
# 故障部件。
comFault = np.zeros([200, 2])
for i in range(200):
    if np.max(votSort[i]) > 0:
        if votSortIdx[i][-1] < 4:
            startIdx = (votSortIdx[i][-1] // 2) * 5
            specFault = np.argmax(resRatio[i][startIdx: startIdx + 5]) + 1
            if specFault > 3:
                comFault[i][0] = 1
            else:
                comFault[i][0] = 2
        else:
            startIdx = (votSortIdx[i][-1] - 4) * 5 + 40
            specFault = np.argmax(resRatio[i][startIdx: startIdx + 5]) + 1
            if specFault == 4:
                comFault[i][0] = 1
            else:
                comFault[i][0] = 2
        if doubleFault[i] == 1:
            if votSortIdx[i][-2] < 4:
                startIdx = (votSortIdx[i][-2] // 2) * 5
                specFault = np.argmax(resRatio[i][startIdx: startIdx + 5]) + 1
                if specFault > 3:
                    comFault[i][1] = 1
                else:
                    comFault[i][1] = 2
            else:
                startIdx = (votSortIdx[i][-2] - 4) * 5 + 40
                specFault = np.argmax(resRatio[i][startIdx: startIdx + 5]) + 1
                if specFault == 4:
                    comFault[i][1] = 1
                else:
                    comFault[i][1] = 2
# print(comFault)
# print(votSortIdx)
# plt.plot(np.arange(5), resRatio[1, :5])

plt.figure(1)
plt.scatter(np.arange(200), comFault[:, 0], label='fault1')
plt.scatter(np.arange(200), comFault[:, 1], label='fault2')
plt.legend(loc=0, bbox_to_anchor=(0.2, 0.8))
# 绘制回归曲线
plt.figure(2)
plt.scatter(loadTraining, labelTraining[:, 1], c='b', label='training')
pltx = np.linspace(0.5, 1.7, 200)
plty = linregList[1].predict(pltx.reshape([200, 1]))
plty2 = linregList[1].predict(loadTraining)
pstd = np.sqrt(sum((plty2 - labelTraining[:, 1].reshape(200)) ** 2) / (200 - 2))
sxx = np.var(loadTraining) * 200
plt.plot(pltx, plty)

plty3 = loess(np.concatenate((loadTesting.reshape([200, 1]), np.ones([200, 1])), 1), np.concatenate((loadTraining.reshape([200, 1]), np.ones([200, 1])), 1), labelTraining[:, 1].reshape(200), 0.1)
plt.scatter(loadTesting, plty3, c='r', marker='*', label='loess')

# 绘制置信区间
plt.plot(pltx, plty + tval * pstd * np.sqrt(1 + 1/200 + (pltx - np.mean(loadTraining)) ** 2 / sxx))
plt.plot(pltx, plty - tval * pstd * np.sqrt(1 + 1/200 + (pltx - np.mean(loadTraining)) ** 2 / sxx))
plt.title('predicting interval')
plt.scatter(loadTesting, labelTesting[:, 1], marker='s', c='g', label='testing')
plt.legend(loc='upper left')
plt.show()
