# coding=utf-8
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
# from lightgbm
RootPath = r'D:\Great_job_of_teammate'
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import adagrad
from collections import defaultdict
from itertools import chain
from deepctr.feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input
from tensorflow.python.keras.layers import Embedding
ACTION_LIST = ["read_comment"]#, "like", "click_avatar", "forward"]
from uauc import evaluate_model
import time


def DeepFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
           dnn_activation='relu', dnn_use_bn=False, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    #添加内容
    #embedding_layer=Embedding(input_dim=106444,output_dim=512,trainable=False,embeddings_initializer=tf.constant_initializer(embedding_matrix))

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)


    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])
    #添加内容
    #group_embedding_dict[DEFAULT_GROUP_NAME].append(embedding_layer(features['feedid']))
    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model

def generate_input():
    sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id','bgm_singer_id']
    dense_feature_list=[]
    data_path = 'user_action_8_version2.csv'
    data_path = os.path.join(RootPath, data_path)
    data_df = pd.read_csv(data_path)[sparse_features+ACTION_LIST]
    for i in range(9,14):
        data_path = 'user_action_{}_version2.csv'.format(i)
        data_path = os.path.join(RootPath, data_path)
        tmp = pd.read_csv(data_path)[sparse_features+ACTION_LIST]
        data_df=pd.concat([data_df,tmp])
    data_df=data_df.reset_index(drop=True)
    data_df[["bgm_song_id", "bgm_singer_id"]]+=1
    data_df[["bgm_song_id", "bgm_singer_id"]]=data_df[["bgm_song_id", "bgm_singer_id"]].fillna(0)
    #生成验证集
    evaluate_path='user_action_14_version2.csv'
    evaluate_path = os.path.join(RootPath, evaluate_path)
    evaluate_df = pd.read_csv(evaluate_path)[sparse_features+ACTION_LIST]
    evaluate_df[["bgm_song_id", "bgm_singer_id"]]+=1
    evaluate_df[["bgm_song_id", "bgm_singer_id"]]=evaluate_df[["bgm_song_id", "bgm_singer_id"]].fillna(0)
    evaluate_df=evaluate_df.reset_index(drop=True)
    return data_df,evaluate_df


dnn_hidden_units = (512, 256)
l2_reg_embedding=0.1
lr=0.03
embedding_dim=10
optimizer = adagrad(lr=lr)
batch_size=2048
epochs=5


def train_evaluate(train_df,evaluate_df,dense_features,sparse_features):
    print(train_df.columns)
    print('total_data_number:',train_df.shape[0])

    train_dict = {}  # 在online_train ,offline_train的时候存储结果
    label_dict = {}  # 存储真实结果
    user_id_dict = {}  # 存储user_id
    #存储predict阶段的字典
    evaluate_dict = {}
    evaluate_label_dict={}
    evaluate_user_dict={}
    for action in ACTION_LIST:
        # 将数据灌入模型=========================================
        dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_features]
        sparse_feature_columns = [SparseFeat(feat, int(train_df[feat].max() + 1), embedding_dim) for feat in sparse_features]
        fixlen_feature_columns = sparse_feature_columns + dense_feature_columns
        dnn_feature_columns = fixlen_feature_columns.copy()
        linear_feature_columns = fixlen_feature_columns.copy()
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        model_input = {name: train_df[name] for name in feature_names}
        model_label = train_df[action]
        # =========================================================
        model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=dnn_hidden_units,
                       l2_reg_embedding=l2_reg_embedding)
        model.compile(optimizer, loss='binary_crossentropy')
        #训练部分,将结果记录
        print("start training")
        history = model.fit(model_input, model_label, batch_size=batch_size,epochs=epochs)
        history = model.predict(model_input)
        logits = [x[0] for x in history]
        train_dict[action] = logits
        user_id_dict[action] = model_input['userid']
        label_dict[action] = model_label
        #evaluate部分
        model_input = {name: evaluate_df[name] for name in feature_names}
        model_label = evaluate_df[action]
        history=model.predict(model_input)
        logits = [x[0] for x in history]
        evaluate_dict[action] = logits
        evaluate_user_dict[action] = model_input['userid']
        evaluate_label_dict[action] = model_label


    evaluate_model(label_dict, train_dict, user_id_dict, ACTION_LIST)
    evaluate_model(evaluate_label_dict,evaluate_dict,evaluate_user_dict,ACTION_LIST)







if __name__=="__main__":
    start_time=time.time()
    train_df,evaluate_df=generate_input()
    dense_features=[]
    sparse_features=['userid', 'feedid', 'device', 'authorid', 'bgm_song_id','bgm_singer_id']
    train_evaluate(train_df,evaluate_df,dense_features,sparse_features)
    print("time cost:",time.time()-start_time)
    '''
    user_interest=os.path.join(RootPath,"feed_info_modified2.pkl")
    with open(user_interest,'rb') as f:
        interest=pickle.load(f)
    print(interest.columns)
    '''

