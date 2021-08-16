# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 22:14:26 2021
Estimator的使用，利用Estimator的自带DNN进行花朵分类
@author: Mady
"""

import tensorflow as tf
import numpy as np
import pandas as pd
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train.head()
train_y = train.pop('Species')
test_y = test.pop('Species')

# 标签列现已从数据中删除
train.head()

# def input_evaluation_set():
#     features = {'SepalLength': np.array([6.4, 5.0]),
#                 'SepalWidth':  np.array([2.8, 2.3]),
#                 'PetalLength': np.array([5.6, 3.3]),
#                 'PetalWidth':  np.array([2.2, 1.0])}
#     labels = np.array([2, 1])
#     return features, labels

def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    # 将输入转换为数据集。
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 如果在训练模式下混淆并重复数据。
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# 特征列描述了如何使用输入。
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
    
# 构建一个拥有两个隐层，隐藏节点分别为 30 和 10 的深度神经网络。
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 隐层所含结点数量分别为 30 和 10.
    hidden_units=[30, 10],
    # 模型必须从三个类别中做出选择。
    n_classes=3)

# 训练模型。
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
    
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# 由模型生成预测
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    # 将输入转换为无标签数据集。
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        SPECIES[class_id], 100 * probability, expec))
    
    
    
    
    
    