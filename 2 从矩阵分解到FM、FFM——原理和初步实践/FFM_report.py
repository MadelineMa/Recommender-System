# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#辅助函数，用于计算单个观测和参数的FFM值
def FFMkernel(xi, field, w0, w, W, one_hot = False, all_one = False, loc1 = 0, 
              loc2 = 0):
    #变量解释：xi:        单个观测解释变量向量
    #         field:     域起始点向量
    #         w0:        常数项参数
    #         w:         一次项参数
    #         W:         交互项参数
    #         one-hot:   每个域是否仅对应一个非零向量？
    #         all_one:   变量的值是否全为1？
    #         loc1:      坐标序列一，默认由FFM程序构造并传入，若非one-hot则自行构造
    #         loc2:      坐标序列二，同上
    #输出参数：ffm:       本观测对应的FFM核的计算结果
    pos = np.array(range(len(xi)))[xi!=0]         #储存非零变量位置
    t = len(pos)                                  #非零变量个数
    k = np.shape(W)[2]                            #隐向量维数
    field_i = field[pos]                          #当前观测的非零变量的域
    ffm = w0 + np.dot(xi, w)                      #计算FFM常数和一次项部分
    #'''
    if one_hot==False:                            #需根据非零元素个数构造坐标向量
        pmat = np.array(range(t)).repeat(t).reshape(t, t) + 1
                                                  #t维方阵，第i行元素全为i
        loc1 = (pmat - np.tril(pmat)).reshape(t**2, 1)
                                                  #上三角（对角线为0）横向拉直
        loc2 = (pmat - np.triu(pmat)).T.reshape(t**2, 1)
                                                  #下三角（对角线为0）纵向拉直
        loc1 = loc1[loc1!=0] - 1                  #去除为0部分，减1变为坐标，下同
        loc2 = loc2[loc2!=0] - 1
    
    Wx = copy.deepcopy(W)
    if all_one==False:                            #如果非零变量不全为1
        for j in pos:
            Wx[j, :, ] = Wx[j, :, ] * xi[j]       #W的每个变量“切片”都乘上变量值

    vecW1 = Wx[pos[loc1], field_i[loc2], ].reshape(len(loc1) * k, 1)
                                                  #提取所需向量，全部拉直，下同
    vecW2 = Wx[pos[loc2], field_i[loc1], ].reshape(len(loc1) * k, 1)
    
    ffm = ffm + np.dot(vecW1.T, vecW2)[0, 0]      #求向量内积
    #'''
    ''' 以下程序为for循环版本
    for j1 in range(len(pos) - 1):
            for j2 in range(j1 + 1, len(pos)):    #不重复地对变量进行两两组合
                ja = pos[j1]                      #变量j1位置
                jb = pos[j2]                      #变量j2位置
                f1 = field_i[j1]                  #变量j1的域
                f2 = field_i[j2]                  #变量j2的域
                ffm = ffm + xi[ja] * xi[jb] * np.dot(W[ja, f2, ], W[jb, f1, ])
    '''
    return ffm  

#主函数，随机梯度下降+自适应学习率+提前停止法，求解参数估计
def FFM(xx, yy, fp, k, r_va, lda, eta, maxI, one_hot = False, all_one = False,
        report = 1000):
    #变量解释：xx:        训练集解释变量资料矩阵
    #         yy:        训练集响应变量向量
    #         fp:        域起始点向量
    #         k:         隐向量维数
    #         r_va:      验证集观测数占比
    #         lda:       正则化参数
    #         eta:       学习率
    #         maxI:      最大迭代次数
    #         one-hot:   每个域是否仅对应一个非零向量？
    #         all_one:   变量的值是否全为1？
    #         report:    计算给定数量观测后汇报一次进度
    #输出变量：w0:        常数项参数
    #         w:         一次项参数
    #         W:         交互项参数
    #         loss:      最终损失函数值
    #         ite:       总迭代次数（以观测计）       
    X = copy.deepcopy(xx).values                  #深拷贝，防止传递引用
    y = copy.deepcopy(yy).values
    y[y==0] = -1
    
    n = round(np.shape(X)[0] * (1 - r_va))        #实际训练集观测数
    m = np.shape(X)[0] - n                        #验证集观测数
    p = np.shape(X)[1]                            #变量个数
    trlist = np.sort(np.random.choice(range(n + m), n, replace = False))
    valist = np.array(range(n + m))
    valist = np.delete(valist, trlist)            #按比例随机产生训练集和验证集
    trX = X[trlist, ]                             #实际训练集
    trY = y[trlist]
    vaX = X[valist, ]                             #验证集
    vaY = y[valist]
    f = len(fp)                                   #域总数
    fp = np.append(fp, 2147483647)                #域分割点向量末位添加极大的数
    field = pd.Series(range(p)).apply(lambda x: np.argmin(x>=fp)).values - 1
                                                  #计算并储存各个变量的域
    w0 = 0                                        #常数项
    w0_c = copy.deepcopy(w0)                      #参数备份
    g0 = 1                                        #常数项累积平方梯度
    w = np.zeros(p)                               #一次项
    w_c = copy.deepcopy(w)
    g = np.ones(p)                                #一次项累积平方梯度
    W = np.random.uniform(0, k**-0.5, [p, f, k])   #所有隐向量的初始值
    W_c = copy.deepcopy(W)
    G = np.ones([p, f, k])                        #隐向量对应累积平方梯度
    loss_c = 2147483647                           #初始化上一个验证集损失
    
    
    if one_hot:                                   #若满足one-hot编码，则可矩阵化
        pmat = np.array(range(f)).repeat(f).reshape(f, f) + 1
                                                  #f维方阵，第i行元素全为i
        pmat_u = pmat - np.tril(pmat)             #pmat的上三角部分（对角线为0）
        pmat_l = pmat - np.triu(pmat)             #pmat的下三角部分（对角线为0）
        loc1 = pmat_u.reshape(f**2, 1)            #上三角横向拉直
        loc2 = pmat_l.T.reshape(f**2, 1)          #下三角纵向拉直
        loc1 = loc1[loc1!=0] - 1                  #去除0部分，减1变为坐标，下同
        loc2 = loc2[loc2!=0] - 1
    else:                                         #不满足one_hot时
        loc1 = 0                                  #方便函数调用，loc1此时无意义
        loc2 = 0
    
    ite = 0
    for ite in range(maxI * n + 1):
        i = ite % n
        ite = ite + 1
        xi = trX[i, :]                            #当前观测
        
        ffm = FFMkernel(xi, field, w0, w, W, one_hot = one_hot, 
                        all_one = all_one, loc1 = loc1, loc2 = loc2)
        kappa = -trY[i] / (1 + np.exp(trY[i] * ffm))
                                                  #损失函数导数值
        pos = np.array(range(p))[xi!=0]           #储存非零变量位置
        field_i = field[pos]                      #当前观测的非零变量的域   
        
        g0_tmp = lda * w0 + kappa                 #w0的梯度
        g0 = g0 + g0_tmp**2                       #w0的累积平方梯度
        w0 = w0 - (eta / g0**0.5) * g0_tmp        #更新w0
        
        g_tmp = lda * w + kappa * xi              #w的梯度
        g = g + g_tmp**2                          #w的累积平方梯度
        w = w - (eta / g**0.5) * g_tmp            #更新w
        
        if one_hot:                               #one_hot时，参数更新互不影响
            Wxx = copy.deepcopy(W)
            if all_one==False:                    #如果非零变量不全为1
                for j1 in pos:                    #对W第一维度按xi[pos]加权
                    Wxx[j1, :, ] = Wxx[j1, :, ] * xi[j1]
                for j2 in pos:                    #对W第二维度再按xi[pos]加权
                    Wxx[:, field[j2], ] = Wxx[:, field[j2], ] * xi[j2]
            
            p1 = pos[loc1]                        #坐标简洁写法，下同
            f2 = field_i[loc2]
            p2 = pos[loc2]
            f1 = field_i[loc1]
            
            
            G_tmp = W[p1, f2, ] * lda + kappa * Wxx[p2, f1, ] 
                                                  #隐向量梯度
            G[p1, f2, ] = G[p1, f2, ] + G_tmp**2  #累积平方梯度更新
            W[p1, f2, ] = W[p1, f2, ] - (eta / G[p1, f2, ]**0.5) * G_tmp
                                                  #隐向量更新（“小对大”）                                 
                                                  
            G_tmp = W[p2, f1, ] * lda + kappa * Wxx[p1, f2, ]
            G[p2, f1, ] = G[p2, f1, ] + G_tmp**2
            W[p2, f1, ] = W[p2, f1, ] - (eta / G[p2, f1, ]**0.5) * G_tmp
                                                  #隐向量更新（“大对小”）                                
                                                  
            
        else:                                     #multi_hot时，不能用矩阵描述
            for j1 in range(len(pos) - 1):
                for j2 in range(j1 + 1, len(pos)):
                    ja = pos[j1]                  #变量j1位置
                    jb = pos[j2]                  #变量j2位置
                    f1 = field_i[j1]              #变量j1的域
                    f2 = field_i[j2]              #变量j2的域
                    if all_one==False:
                        cons = kappa * xi[ja] * xi[jb]
                    else:
                        cons = kappa
                    g12 = lda * W[ja, f2, ] + cons * W[jb, f1, ]
                    g21 = lda * W[jb, f1, ] + cons * W[ja, f2, ]
                                                  #隐向量w_j1_f2和w_j2_f1的梯度
                    G[ja, f2, ] = G[ja, f2, ] + g12**2
                    G[jb, f1, ] = G[jb, f1, ] + g21**2
                                                  #累积平方梯度更新
                    W[ja, f2, ] = W[ja, f2, ] - (eta / G[ja, f2, ]**0.5) * g12
                                                  #隐向量更新（“小对大”）
                    W[jb, f1, ] = W[jb, f1, ] - (eta / G[jb, f1, ]**0.5) * g21
                                                  #隐向量更新（“大对小”）
        
        if i==(n - 1):                            #此时将所有观测遍历了一遍
            loss = 0                                  #初始化验证集损失函数
            ffm = np.zeros(m)
            for j in range(m):
                ffm[j] = FFMkernel(vaX[j, ], field, w0, w, W, one_hot = one_hot,
                                   all_one = all_one, loc1 = loc1, loc2 = loc2)
            loss = loss + sum(np.log(1 + np.exp(-vaY * ffm))) / m
            #loss = loss + 0.5 * lda * (w0**2 + sum(w**2) + sum(sum(sum(W**2)))) 
            epoch = ite // n
            print("第" + str(epoch) + "轮迭代：" + str(loss))
                                                  #计算损失函数
            if loss>loss_c:                       #损失增加表示不需要继续迭代
                print("损失值增大，终止迭代")
                return w0_c, w_c, W_c, loss_c, ite
        
            w0_c = w0                             #损失仍在减小则更新参数继续迭代
            w_c = w                               #用新参数替换参数备份
            W_c = W
            loss_c = loss
            
        if (ite % report==0):                       #每迭代1000次汇报一次
            print("观测" + str(ite))
    
    return w0, w, W, loss, ite

#预测函数，给定参数和测试集数据，
def FFMpredict(xx, fp, w0, w, W, one_hot = False, all_one = False):
    #变量解释：xx:        训练集解释变量资料矩阵
    #         fp:        域起始点向量 
    #         w0:        常数项参数
    #         w:         一次项参数
    #         W:         交互项参数
    #         one-hot:   每个域是否仅对应一个非零向量？
    #         all_one:   变量的值是否全为1？
    #输出变量：ffm:       向量，每一维度对应一个观测的ffm预测值
    X = copy.deepcopy(xx).values
    n = np.shape(X)[0]
    p = np.shape(X)[1]
    f = len(fp)
    fp = np.append(fp, 2147483647)                #域分割点向量末位添加极大的数
    field = pd.Series(range(p)).apply(lambda x: np.argmin(x>=fp)).values - 1
    
    if one_hot:                                   #若满足one-hot编码，则可矩阵化
        pmat = np.array(range(f)).repeat(f).reshape(f, f) + 1
                                                  #f维方阵，第i行元素全为i
        pmat_u = pmat - np.tril(pmat)             #pmat的上三角部分（对角线为0）
        pmat_l = pmat - np.triu(pmat)             #pmat的下三角部分（对角线为0）
        loc1 = pmat_u.reshape(f**2, 1)            #上三角横向拉直
        loc2 = pmat_l.T.reshape(f**2, 1)          #下三角纵向拉直
        loc1 = loc1[loc1!=0] - 1                  #去除0部分，减1变为坐标，下同
        loc2 = loc2[loc2!=0] - 1      
    else:                                         #不满足one_hot时
        loc1 = 0                                  #方便函数调用，loc1此时无意义
        loc2 = 0
    
    ffm = np.zeros(n)
    for i in range(n):
        ffm[i] = FFMkernel(X[i, ], field, w0, w, W, one_hot = one_hot, 
                           all_one = all_one, loc1 = loc1, loc2 = loc2)
    ffm = 1 / (1 + np.exp(-ffm))
    return ffm
        
    
        
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
    

def reportROC(y2, test_y, grid, c = 0.5):
    #变量解释：y2:    模型输出的预测结果
    #         test_y:真实的测试集响应变量
    #         grid:  ROC曲线绘制时的阈值列表
    #         c:     人为选择的阈值
    #输出变量：TPR_c: 遍历过程中算出的所有TPR值
    #         FPR_c: 遍历过程中算出的所有FPR值
    TPR = np.zeros(len(grid) + 1)
    FPR = np.zeros(len(grid) + 1)
    pos = 1
    auc = 0
    for k in np.flip(grid):
        pred_y = copy.deepcopy(y2)
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
    TPR_c = copy.deepcopy(TPR)
    FPR_c = copy.deepcopy(FPR)

    pred_y[y2>=c] = 1
    pred_y[y2<c] = 0
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
    return TPR_c, FPR_c

#蘑菇数据集
path = "C:\\Users\\Ouyang\\Desktop\\One\\Workspace\\github repo\\"
name = "mushrooms.csv"
dumlist = range(1, 23)
yname = "class"
np.random.seed(0)
train_x, train_y, test_x, test_y = load_process_data(0.8, path, name, yname,
                                                     dumlist, [])
#训练集占比80%，是因为后续早停机制还要删去80%，最后达到64%从而与FM保持一致
train_y = train_y.replace('e', 0)                  #将响应变量化为01变量
train_y = train_y.replace('p', 1)
test_y = test_y.replace('p', 1)
test_y = test_y.replace('e', 0)

field = np.array([0, 6, 10, 20, 22, 31, 33, 35, 37, 49, 51, 56, 60, 64, 73, 82,
                  83, 87, 90, 95, 104, 110])

np.random.seed(0)
w0, w, W, loss, ite = FFM(train_x, train_y, fp = field, k = 4, r_va = 0.2, 
                          lda = 1e-3, eta = 0.1, maxI = 4, one_hot = True, 
                          all_one = True)

ffm1 = FFMpredict(test_x, field, w0, w, W, one_hot = True, all_one = False)

#阈值列表
grid = np.arange(start = 0, stop = 1, step = 0.001)

#查看运行结果
tpr1, fpr1 = reportROC(ffm1, test_y, grid, 0.5)


#下面使用心脏数据集
name = "heart.csv"
dumlist = [2, 6, 10, 11, 12]
normlist = [0, 3, 4, 7, 9]
yname = "target"
np.random.seed(0)
train_x, train_y, test_x, test_y = load_process_data(0.8, path, name, yname,
                                                     dumlist, normlist)

#心脏病数据集变量解释：响应变量：target: 1为心脏病，0为无心脏病
#解释变量：域0：age年龄，sex性别
#         域1：trestbps静息血压，restecg静息心电图结果（三分类），thalach最大心率
#         域2：chol血清胆固醇浓度，fbs空腹血糖浓度（大于120mg/dl为1，否则为0）
#         域3：cp胸痛类型（四分类），exang运动诱发的心绞痛（二分类）
#         域4：oldpeak运动带来的心电图ST段压低，slope运动时心电图ST段斜率类型（三分类）
#         域5：ca荧光检查着色的主要血管数（五分类），thal铊压力测试结果（四分类）
order = ["age", "sex", "trestbps", "restecg_0", "restecg_1", "restecg_2", 
         "thalach", "chol", "fbs", "cp_0", "cp_1", "cp_2", "cp_3", "exang",
         "oldpeak", "slope_0", "slope_1", "slope_2", "ca_0", "ca_1", "ca_2", 
         "ca_3", "ca_4", "thal_0", "thal_1", "thal_2", "thal_3"]
train_x = train_x[order]
test_x = test_x[order]
field = np.array([0, 2, 7, 9, 14, 18])

np.random.seed(0)
w01, w1, W1, loss1, ite1 = FFM(train_x, train_y, fp = field, k = 4, r_va = 0.2, 
                             lda = 1e-4, eta = 0.1, maxI = 4, one_hot = False, 
                             all_one = False)

ffm2 = FFMpredict(test_x, field, w01, w1, W1, one_hot = False, all_one = False)

tpr2, fpr2 = reportROC(ffm2, test_y, grid, 0.4)

