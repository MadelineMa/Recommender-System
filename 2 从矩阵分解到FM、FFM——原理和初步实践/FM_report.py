# -*- coding: utf-8 -*-

import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#矩阵版本的FM
def FM(xx, yy, txx, k, maxI, sig = 0.05, eta = 1e-3, lda = 1e-4, reg = True):
    #变量解释：xx:    训练集解释变量数据
    #         yy:    训练集响应变量数据
    #         txx:   测试集解释变量数据
    #         k:     隐向量维数
    #         maxI:  最大迭代次数（遍历所有样本次数）
    #         sig:   隐向量初始值服从正态分布的方差
    #         eta:   学习率
    #         lda:   正则化参数
    #         reg:   是否进行回归，为False代表二分类判别
    #输出变量：y1:    训练集回判结果
    #         y2:    测试集预测结果
    #         w0:    常数项参数
    #         w:     一次项参数
    #         V:     隐向量矩阵
    X = copy.deepcopy(xx).values                #深拷贝，防止传递引用
    y = copy.deepcopy(yy).values
    if reg==False:
        y[y==0] = -1
    tX = copy.deepcopy(txx).values
    n = np.shape(X)[0]                          #训练集观测数
    p = np.shape(X)[1]                          #变量个数
    
    #给出参数初始值
    w0 = 0
    w = np.zeros([p, 1])
    V = np.random.normal(0, sig, [p, k])
    
    #主循环
    for ite in range(maxI * n + 1):
        i = ite % n                             #取余数，找到当前观测
        xi = X[i, :].reshape(p, 1)              #提前计算以节省时间
        xi2 = xi**2
        yhat = w0 + np.dot(xi.T, w) + 0.5 * np.dot((np.dot(xi.T, V)**2 - \
               np.dot(xi2.T, V**2)), np.ones((k, 1)))
        regular = 2 * eta * lda                 #正则化所需常数
        if reg:                                 #回归情形
            ydif = yhat - y[i]
            cons = 2 * eta * ydif
        else:                                   #二分类判别情形
            ysig = 1 / (1 + np.exp(-yhat * y[i]))
            cons = eta * (ysig - 1) * y[i]
        
        #参数矩阵更新
        w0 = w0 - cons - regular * w0
        w = w - cons * xi - regular * w
        V = V - cons * np.dot((np.dot(xi, xi.T) - np.diag(xi2[:, 0])), V) - \
            regular * V

            
    #回判                
    XV = np.dot(X, V)
    X2 = X**2        
    y1 = w0 + np.dot(X, w) + 0.5 * np.dot(XV**2 - np.dot(X2, V**2), np.ones((k, 1)))
    if reg==False:
        y1 = 1 / (1 + np.exp(-y1))
        
    #预测
    XV = np.dot(tX, V)
    X2 = tX**2
    y2 = w0 + np.dot(tX, w) + 0.5 * np.dot(XV**2 - np.dot(X2, V**2), np.ones((k, 1)))
    if reg==False:
        y2 = 1 / (1 + np.exp(-y2))
         
    return y1, y2, w0, w, V


#完全使用for循环的FM（旧版本，未加入判别部分和正则化项）
def forFM(xx, yy, txx, k, maxI, sig = 0.1, eta = 0.001):
    X = copy.deepcopy(xx).values                #深拷贝，防止传递引用
    y = copy.deepcopy(yy).values
    tX = copy.deepcopy(txx).values
    n = np.shape(X)[0]                          #训练集观测数
    p = np.shape(X)[1]                          #变量个数
    
    #给出参数初始值
    w0 = 0
    w = np.zeros(p)
    V = np.random.normal(0, sig, [p, k])
    
    #主循环，完全按照for循环逐个更新参数
    for ite in range(maxI * n + 1):
        i = ite % n
        xi = X[i, :]
        ydif = w0 + np.dot(xi, w) - y[i]
        for f in range(k):
            ydif = ydif + 0.5 * (np.dot(xi, V[:, f])**2 - np.dot(xi**2, V[:, f]**2))    #更新yif
        w0 = w0 - 2 * eta * ydif   #更新w0
        for l in range(p):
            w[l] = w[l] - 2 * eta * xi[l] * ydif   #更新wl
            for f in range(k):
                V[l, f] = V[l, f] - 2 * eta * xi[l] * ydif * (np.dot(xi, V[:, f]) - V[l, f] * xi[l])   #更新vlf
    
    #回判
    y1 = np.zeros(n)
    for i in range(n):
        xi = X[i, :]
        y1[i] = w0 + np.dot(xi, w)
        for f in range(k):
            y1[i] = y1[i] + 0.5 * (np.dot(xi, V[:, f])**2 - np.dot(xi**2, V[:, f]**2))
            
    #预测
    y2 = np.zeros(np.shape(tX)[0])
    for i in range(np.shape(tX)[0]):
        xi = tX[i, :]
        y2[i] = w0 + np.dot(xi, w)
        for f in range(k):
            y2[i] = y2[i] + 0.5 * (np.dot(xi, V[:, f])**2 - np.dot(xi**2, V[:, f]**2))
         
    return y1, y2, w0, w, V
  

#这部分是数据读取，和教师版本相似，但也有不少改动
def load_process_data(ratio, path, name, y_name, dumlist = [], normlist = []): 
    #变量解释：ratio:   测试集占总样本量的比例
    #         path:    数据文件路径
    #         name:    数据文件名称
    #         y_name:  数据集中响应变量的名称
    #         dumlist: 需要转化为哑变量的变量编号列表
    #         normlist:需要标准化的变量编号列表       
    #输出变量：train_x: 训练集解释变量
    #         train_y: 训练集响应变量
    #         test_x:  测试集解释变量
    #         test_y:  测试集响应变量
    data = pd.read_csv(path + name)
    data_rebuild = data

    for f_name in data.columns[dumlist]:          #将指定变量转为one-hot编码
        f_name_Df = pd.get_dummies(data_rebuild[f_name], prefix=f_name)
        data_rebuild = pd.concat([data_rebuild, f_name_Df], axis=1)
        data_rebuild.drop(f_name, axis=1, inplace=True)
        
    for f_name in data.columns[normlist]:         #将指定变量标准化
        m = np.mean(data_rebuild[f_name])
        sd = np.std(data_rebuild[f_name])
        data_rebuild[f_name] = (data_rebuild[f_name] - m) / sd
        
    datlen = int(len(data_rebuild) * ratio)       #训练集观测数
    trlist = np.sort(np.random.choice(range(len(data_rebuild)), datlen, replace = False))
    telist = np.array(range(len(data_rebuild)))
    telist = np.delete(telist, trlist)            #按比例随机产生训练集和测试集
    
    train_data, test_data = data_rebuild.iloc[trlist], data_rebuild.iloc[telist]
    train_x, train_y = train_data, train_data.pop(y_name)
    test_x, test_y = test_data, test_data.pop(y_name)
    return train_x, train_y, test_x, test_y   


#下面这个函数用来对得到的预测结果绘制ROC曲线和输出正确率
def reportROC(y2, test_y, grid, c = 0.5):
    #变量解释：y2:    模型输出的预测结果
    #         test_y:真实的测试集响应变量
    #         grid:  ROC曲线绘制时的阈值列表
    #         c:     人为选择的阈值
    TPR = np.zeros(len(grid) + 1)
    FPR = np.zeros(len(grid) + 1)
    pos = 1
    auc = 0
    for k in np.flip(grid):
        pred_y = copy.deepcopy(y2[:, 0])
        pred_y[pred_y>=k] = 1
        pred_y[pred_y<k] = 0
        TPR[pos] = sum((pred_y==1)&(test_y==1)) / sum(test_y==1)
        FPR[pos] = sum((pred_y==1)&(test_y==0)) / sum(test_y==0)
        auc = auc + (TPR[pos] + TPR[pos - 1]) * (FPR[pos] - FPR[pos - 1]) / 2
        pos = pos + 1
    
    plt.plot(FPR, TPR)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()   
    print("AUC = " + str(auc))

    pred_y[y2[:, 0]>=c] = 1
    pred_y[y2[:, 0]<c] = 0
    COR = sum(pred_y==test_y) / len(test_y)
    TP = sum((pred_y==1)&(test_y==1))
    TPR = TP / sum(test_y==1)
    FP = sum((pred_y==1)&(test_y==0))
    FPR = FP / sum(test_y==0)
    TN = sum((pred_y==0)&(test_y==0))
    TNR = TN / sum(test_y==0)
    FN = sum((pred_y==0)&(test_y==1))
    FNR = FN / sum(test_y==1)
    print("阈值：" + str(c) + "\n正确率：" + str(COR) + "\nTPR：" + str(TPR) + \
          "\nFNR：" + str(FNR) + "\nFPR：" + str(FPR) + "\nTNR：" + str(TNR))
    mat = pd.DataFrame([[TP, FN, sum(test_y==1)], [FP, TN, sum(test_y==0)], 
                        [sum(pred_y==1), sum(pred_y==0), len(test_y)]], 
                       index = ["真阳性", "真阴性", ""], 
                       columns = ["预测阳性", "预测阴性", ""])
    print(mat)


#下面这个函数用来比较FM和forFM的时间差异
def time_compare(train_x, train_y, test_x, test_y, k = 5, maxI = 5):
    #
    #for循环版本
    start1 = time.perf_counter()
    y1, y2, w0, w, V = forFM(train_x, train_y, test_x, k = k, maxI = maxI)
    end1 = time.perf_counter()
    #矩阵版本
    start2 = time.perf_counter()
    y3, y4, w0, w, V = FM(train_x, train_y, test_x, k = k, maxI = maxI)
    end2 = time.perf_counter()
    #计时
    t1 = end1 - start1
    t2 = end2 - start2
    return t1, t2


####################
#### 主程序部分 #####
####################
    
#阈值列表
grid = np.arange(start = 0, stop = 1, step = 0.001)
        
#下面对mushrooms数据集进行分析，先读取数据
path = "C:\\Users\\Ouyang\\Desktop\\One\\Workspace\\github repo\\"
name = "mushrooms.csv"
dumlist = range(1, 23)
yname = "class"
np.random.seed(42)
train_x, train_y, test_x, test_y = load_process_data(0.64, path, name, yname,
                                                     dumlist, [])
'''train_y = train_y.replace('e', 0)                  #将响应变量化为01变量
train_y = train_y.replace('p', 1)
test_y = test_y.replace('p', 1)
test_y = test_y.replace('e', 0)'''

train_y=train_y.map({'e':0,'p':1})
test_y=test_y.map({'e':0,'p':1})

#按照如下参数运行
y1, y2, w0, w, V = FM(train_x, train_y, test_x, k = 4, maxI = 4, sig = 0.05, 
                      eta = 1e-3, lda = 1e-4, reg = False)

#查看运行结果
reportROC(y2, test_y, grid, 0.5)
print("\n\n")



#下面对heart数据集进行分析
name = "heart.csv"
dumlist = [6, 10, 11, 12]
normlist = [0, 3, 4, 7, 9]
yname = "target"
np.random.seed(0)
train_x, train_y, test_x, test_y = load_process_data(0.64, path, name, yname,
                                                     dumlist, normlist)

#按照如下参数运行
y1, y2, w0, w, V = FM(train_x, train_y, test_x, k = 4, maxI = 4, sig = 0.05, 
                      eta = 1e-3, lda = 1e-4, reg = False)

#查看运行结果
reportROC(y2, test_y, grid, 0.5)

#时间比较
t1, t2 = time_compare(train_x, train_y, test_x, test_y, k = 5, maxI = 5)
print('for循环版本：'+str(t1))
print('矩阵版本：'+str(t2))



