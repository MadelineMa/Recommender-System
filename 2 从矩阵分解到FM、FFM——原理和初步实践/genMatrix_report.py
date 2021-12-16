# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats

#函数genDiscNum用于产生取值在0~9之间的离散随机数
def genDiscNum(size, dist):
    pSeries = [0] * 10
    
    #若分布列给定，直接使用即可。否则，使用正态分布构造一个离散分布列
    if len(dist)>2:
        pSeries = dist
    else:
        for i in range(0, 10):
            pSeries[i] = stats.norm.pdf(i, dist[0], dist[1])
        pSum = sum(pSeries)
        pSeries = [c * (1 / pSum) for c in pSeries]
        
    #求出pSeries的累积概率
    pAdd = pSeries
    for i in range(1, 10):
        pAdd[i] = pAdd[i - 1] + pAdd[i]
    
    #产生(0, 1)上均匀分布随机数
    ran = np.random.uniform(0, 1, size)
    
    #使用轮盘赌方式，把均匀分布随机数转换为与分布列pSeries对应的随机数
    for i in range(0, size):
        j = 0
        while ran[i]>pAdd[j]:
            j = j + 1
        ran[i] = j
    return ran


#主函数genMatrix用于模拟产生共现矩阵
def genMatrix(nUser, nItem, sLim, sRound = True, sigma2 = 16, default = True):
    #初始化，产生全是-1的矩阵，表示无评分
    score = np.zeros((nUser, nItem)) - 1
    
    #要求输入特征分布，或者使用默认值
    if default!=True:
        uaDist = [0, 0]
        uaDist[0] = int(input("请输入顾客评分偏好分布均值："))
        uaDist[1] = int(input("请输入顾客评分偏好分布方差："))
        ubDist = [0, 0]
        ubDist[0] = int(input("请输入顾客购物频繁度分布均值："))
        ubDist[1] = int(input("请输入顾客购物频繁度分布方差："))
        iaDist = [0, 0]
        iaDist[0] = int(input("请输入商品品质分布均值："))
        iaDist[1] = int(input("请输入商品品质分布方差："))
        ibDist = [0, 0]
        ibDist[0] = int(input("请输入商品人气分布均值："))
        ibDist[1] = int(input("请输入商品人气分布方差："))
    else:
        uaDist = [7, 3]
        ubDist = [2, 4]
        iaDist = [7, 3]
        ibDist = [3, 3]
        
    #产生所有用户和商品的特征
    ua = genDiscNum(nUser, uaDist)
    ub = genDiscNum(nUser, ubDist)
    ia = genDiscNum(nItem, iaDist)
    ib = genDiscNum(nItem, ibDist) 
    
    #遍历矩阵计算打分
    for i in range(0, nUser):
        for j in range(0, nItem):
            
            #判断是否有分数
            isScore = np.random.binomial(1, (ub[i] + 1) * (ib[j] + 1) / 100, 1)
            
            #若有分数，给出分数，并调整到[0, 100]区间
            if isScore==1:
                score[i, j] = np.random.normal((ua[i] + ia[j]) * 5, sigma2, 1)
                if score[i, j]>100:
                    score[i, j] = 100
                elif score[i, j]<0:
                    score[i, j] = 0
            
                #将分数映射到真实打分区间sLim，并按需取整
                score[i, j] = sLim[0] + (sLim[1] - sLim[0]) * score[i, j] / 100
                if sRound==True:
                    score[i, j] = round(score[i, j])
    
    #将ua、ub、ia、ib等信息加入到共现矩阵的行列名称中
    ind = [""] * nUser
    col = [""] * nItem                
    for i in range(0, nUser):
        istr = str(i + 1)
        while len(istr)<len(str(nUser)):
            istr = "0" + istr
        ind[i] = "User" + istr + "a" + str(ua[i])[0] + "b" + str(ub[i])[0]
    for i in range(0, nItem):
        istr = str(i + 1)
        while len(istr)<len(str(nItem)):
            istr = "0" + istr
        col[i] = "Item" + istr + "a" + str(ia[i])[0] + "b" + str(ib[i])[0]
    score = pd.DataFrame(score, index = ind, columns = col)     
    
    return score

#示例矩阵
mat = genMatrix(100, 100, [1, 5], default = True)