# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:38:58 2020
原有样本进行分类训练 tensorflow搭建LR
@author: M.Ma
"""

import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers
CSV_COLUMN_NAMES = ['class','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment','gill-spacing','gill-size'
                    ,'gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',	'stalk-color-above-ring'
                    ,'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']



FEATURE_COLUMNS_NAME = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment','gill-spacing','gill-size'
                    ,'gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring',	'stalk-color-above-ring'
                    ,'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

    
def load_data(y_name='class'):
    
    # data_precess
    data = pd.read_csv("D:\\Mady_profile\\code\\vae-master\\mushrooms.csv")
    c1,c2 = data['class'].value_counts() # 4208:3916
#    e_sample = data[data['class'] == 'e']
#    p_sample = data[data['class'] == 'p']
#    data_rebuild = pd.merge(e_sample, p_sample[0:420],how = 'outer').sample(frac=1)
    data_rebuild = data
    train_data, validation_data = data_rebuild[0:int(len(data_rebuild) * 0.8)], data_rebuild[int(len(data_rebuild) * 0.8):]
    train_x, train_y = train_data, train_data.pop(y_name)
    validation_x, validation_y = validation_data, validation_data.pop(y_name)
    train_y = train_y.replace('e',0)
    train_y = train_y.replace('p',1)
    validation_y =validation_y.replace('p',1)
    validation_y =validation_y.replace('e',0)
    return (train_x, train_y), (validation_x, validation_y)

#数据的灌入
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    print('*******************dataset')
    print(dataset)
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:#预测集
        inputs = (features, labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset


#model_build
# Build Logistic regression
# 相当于输入层 +sigmoid 直接输出结果
def my_model(features, labels, mode, params):
#    input_layer = my_layers.input_layer(features, params['feature_columns'], scale=params['l2_reg'])
#    output_layer = tf.contrib.layers.fully_connected(input_layer, 1, activation_fn=tf.nn.sigmoid, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=params['l2_reg']))
#    logits = tf.reshape(output_layer, [-1])
#    score = logits
#    
    input_layer = tf.contrib.layers.input_from_feature_columns(columns_to_tensors=features, feature_columns=params['feature_columns'])
    with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
        logits = tf.layers.dense(input_layer, 1, activation=None, name='output')
        logits = tf.reshape(logits, [-1])
    
    score = tf.nn.sigmoid(logits)
    
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    auc = tf.metrics.auc(labels, score, name='auc_roc_op')
    metrics = {'auc': auc}
    tf.summary.scalar('auc', auc[1])
    tf.summary.scalar('loss',loss)


    # 模型预测
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'score': score,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 模型评估
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # 模型训练
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.005)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    # args = parser.parse_args(argv[1:])
    batch_size = 10
    # 下载数据，处理格式
    (train_x, train_y), (test_x, test_y) = load_data()
    # # Feature columns describe how to use the input in my_model.
    my_feature_columns = []
    for key in train_x.keys():
        #hash后embedding
        column = layers.sparse_column_with_hash_bucket(column_name=key, hash_bucket_size=30)#, dtype=tf.dtypes.int64)
        tmp_column = layers.embedding_column(column, dimension=4)
        my_feature_columns.append(tmp_column)
   
    
    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_steps = 1, 
        save_checkpoints_secs = None,
        keep_checkpoint_max = 4,     
    )
    # 构建Estimator
    classifier = tf.estimator.Estimator(
        model_fn=my_model, 
        params={'feature_columns':my_feature_columns,
                 'l2_reg': 1e-5},
        model_dir='model_mashroom_my_model_all5/',
        config=my_checkpointing_config,
        )
 
#    classifier1 = tf.estimator.DNNClassifier(
#    feature_columns=my_feature_columns,
#    # 隐层所含结点数量分别为 64 和 32.
#    hidden_units=[64, 32],
#     model_dir='model_mashroom_all/',
#    # 模型必须从2个类别中做出选择。
#    n_classes=2)
    
    try:
        global_step = classifier.get_variable_value("global_step")
        print("99999999999--------", global_step)
    except:
        global_step = 0
    max_steps = global_step + 800 #epoch = 数据量/bitch_size

    #train and evaluate
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:train_input_fn(train_x, train_y, batch_size),
        max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(test_x, test_y, batch_size))
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

