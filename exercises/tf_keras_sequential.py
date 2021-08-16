# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:13:42 2021
tf keras的sequential的使用
心脏病的预测
@author: Mady
"""

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
#dataframe = pd.read_csv(URL)
path='C://Users//lenovo//Desktop//processed.cleveland.data'
data = pd.read_csv(path,sep=',',
                   names=['Age','Sex','CP','Trestbpd','Chol','FBS','RestECG','Thalach','Exang','Oldpeak','Slope','CA','Thal','Target'])
data.head()
data['CA'] = data['CA'].apply(lambda x: -1.0 if x == '?' else x)
data['CA'] = data['CA'].astype('float')#.astype('int')

train, test = train_test_split(data, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

#显示
batch_size = 5 # 小批量大小用于演示
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['Age'])
  print('A batch of targets:', label_batch )


# # 我们将使用该批数据演示几种特征列
# example_batch = next(iter(train_ds))[0]
# # 用于创建一个特征列
# # 并转换一批次数据的一个实用程序方法
# def demo(feature_column):
#   feature_layer = layers.DenseFeatures(feature_column)
#   print(feature_layer(example_batch).numpy())
age = feature_column.numeric_column("Age")
# demo(age)
feature_columns = []

# age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
# demo(age_buckets)

# thal = feature_column.categorical_column_with_vocabulary_list(
#       'Thal', ['3.0','6.0','7.0','?'])

# thal_one_hot = feature_column.indicator_column(thal)
# demo(thal_one_hot)

# thal_embedding = feature_column.embedding_column(thal, dimension=8)
# demo(thal_embedding)

# 数值列
for header in ['Age', 'Trestbpd', 'Chol', 'Thalach', 'Oldpeak', 'Slope', 'CA']:
  feature_columns.append(feature_column.numeric_column(header))

# 分桶列
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 分类列
thal = feature_column.categorical_column_with_vocabulary_list(
      'Thal', ['3.0','6.0','7.0','?'])
thal_one_hot = feature_column.indicator_column(thal)
#feature_columns.append(thal_one_hot)

# 嵌入列
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# 组合列
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              )

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)