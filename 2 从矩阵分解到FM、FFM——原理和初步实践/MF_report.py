# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import genMatrix_report as gM
import matplotlib.pyplot as plt

def MF(m, k, gma, lda, par, e = 1, flag = -1, adj = True, maxI = 1000):
    #变量解释： m:    共现矩阵
    #          k:    隐向量维数
    #          gma:  学习率
    #          lda:  正则化参数
    #          par:  用户和物品隐向量初始值服从正态分布的方差
    #          e:    均方误差小于该值跳出循环
    #          flag: 共现矩阵中用来表示缺省的标识
    #          adj:  是否根据用户和物品偏差进行调整
    #          maxI: 最大迭代次数
    #输出参数： r_ui: 共现矩阵估计值
    #          P:    用户隐向量矩阵
    #          Q:    物品隐向量矩阵
    #          mu:   全局均值（仅当adj=True输出）
    #          b_u:  用户偏差（仅当adj=True输出）
    #          b_i:  物品偏差（仅当adj=True输出）
    #          E:    每轮迭代的均方误差
    mat = copy.deepcopy(m).values
    n_u, n_i = np.shape(mat)                          #用户和物品数量
    P = np.random.normal(0, par, [n_u, k])            #用户隐向量矩阵
    Q = np.random.normal(0, par, [k, n_i])            #物品隐向量矩阵
    
    if adj:
        mu = np.mean(mat[mat!=flag])                  #全局偏差
    else:
        mu = 0
    b_u = np.zeros([n_u, 1])                          #用户偏差初始化
    b_i = np.zeros([1, n_i])                          #物品偏差初始化
    
    loc_u = np.array([])                              #有记录的位置坐标：用户
    loc_i = np.array([])                              #有记录的位置坐标：物品
    for i in range(n_u):
        tmp = sum(mat[i, ]!=flag)                     #每一行的记录数
        if tmp>0:
            loc_u = np.append(loc_u, i * np.ones(tmp))
            loc_i = np.append(loc_i, np.array(range(n_i))[mat[i, ]!=flag])
    loc_u = loc_u.astype(int)                         #将坐标转为整型
    loc_i = loc_i.astype(int)
    
    E = np.zeros(maxI)                                #E存储每次迭代的均方误差
    for ite in range(maxI):                           #主循环
        for k in range(len(loc_u)):                   #遍历所有有记录的位置
            uk = loc_u[k]                             #用户序号
            ik = loc_i[k]                             #物品序号
                               
            r_ui = mu + b_u[uk, 0] + b_i[0, ik] + np.dot(P[uk, ], Q[:, ik])
                                                      #计算矩阵估计值
            r_dif = mat[uk, ik]- r_ui                 #计算估计偏差
            P[uk, ] = P[uk, ] + gma * (r_dif * Q[:, ik] - lda * P[uk, ])
                                                      #更新用户隐向量
            Q[:, ik] = Q[:, ik] + gma * (r_dif * P[uk, ] - lda * Q[:, ik])
                                                      #更新物品隐向量
            if adj:                                   #下面更新用户和物品偏差                 
                b_u[uk, 0] = b_u[uk, 0] + gma * (r_dif - lda * b_u[uk, 0])
                b_i[0, ik] = b_i[0, ik] + gma * (r_dif - lda * b_i[0, ik])
                                                      
        r_ui = np.dot(P, Q) + mu + b_u + b_i          #计算整个矩阵的估计值
        a = sum((mat[loc_u, loc_i] - r_ui[loc_u, loc_i])**2) / len(loc_u)
                                                      #均方误差
        E[ite] = a
        if a<e:                                       #如果均方误差小于e就跳出
            E = E[E!=0]                               #去掉后面为0的项
            break
    if adj:
        return r_ui, P, Q, mu, b_u, b_i, E
    else:
        return r_ui, P, Q, E
        
mat = gM.genMatrix(100, 100, [1, 10])                 #生成共现矩阵

#下面查看不同隐向量维数带来的影响
e = 0                                                 #设置不主动跳出
fig, ax = plt.subplots()                              #画图展示
for i in range(2, 6):                                 #隐向量维数和均方误差关系
    np.random.seed(0)
    ans, P, Q, mu, b_u, b_i, E = MF(mat, k = i, gma = 0.005, lda = 1e-3, par = 0.1, 
                                    e = e, flag = -1, adj = True, maxI = 300)
    ax.plot(range(300), E, label = i)
ax.set_xlabel("iterations")
ax.set_ylabel("mean squared error")
ax.legend()
plt.show()    

#下面选择k=3，e=1，分别在调整和不调整条件下进行计算
ans1, P1, Q1, E1 = MF(mat, k = 3, gma = 0.005, lda = 1e-3, par = 0.1, e = 1, 
                      flag = -1, adj = False, maxI = 300) 
ans2, P2, Q2, mu, b_u, b_i, E2 = MF(mat, k = 3, gma = 0.005, lda = 1e-3, 
                                    par = 0.1, e = 1, flag = -1, adj = True, 
                                    maxI = 300)      