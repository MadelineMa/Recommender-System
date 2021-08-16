# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:15:41 2021
利用MLP对蘑菇毒性进行预测
@author: Mady
"""


import warnings
warnings.filterwarnings('ignore')

#导入处理数据包
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

CSV_COLUMN_NAMES = ['class','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment','gill-spacing','gill-size'
                    ,'gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',	'stalk-color-above-ring'
                    ,'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']



FEATURE_COLUMNS_NAME = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment','gill-spacing','gill-size'
                    ,'gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',	'stalk-color-above-ring'
                    ,'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

    
def load_process_data(y_name='class'):
    
    # data_precess
    data = pd.read_csv("D:\\Mady\\ShanxiProject\\mushrooms.csv")
    c1,c2 = data['class'].value_counts() # 4208:3916
    data_rebuild = data
    #所有特征的one hot 处理
    for f_name in FEATURE_COLUMNS_NAME:
        f_name_Df = pd.get_dummies(data_rebuild[f_name] , prefix=f_name )
        data_rebuild = pd.concat([data_rebuild,f_name_Df],axis=1)
        data_rebuild.drop(f_name,axis=1,inplace=True)
       
    train_data, test_data = data_rebuild[0:int(len(data_rebuild) * 0.8)], data_rebuild[int(len(data_rebuild) * 0.8):]
    train_x, train_y = train_data, train_data.pop(y_name)
    test_x, test_y = test_data, test_data.pop(y_name)
    train_y = train_y.replace('e',0)
    train_y = train_y.replace('p',1)
    test_y = test_y.replace('p',1)
    test_y = test_y.replace('e',0)
    return train_x, train_y,test_x, test_y

#预处理
train_x, train_y,test_x, test_y = load_process_data()
clf = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='logistic', solver='adam',
                    learning_rate_init = 0.001, max_iter=2000)
print(clf)

#训练模型
clf.fit(train_x,train_y)

#预测结果
y_test_pred = clf.predict(test_x)
#评估
report = classification_report(test_y, y_test_pred, digits=4)
print(report)

