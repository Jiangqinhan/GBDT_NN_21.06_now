# coding=utf-8
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
# from lightgbm
from main import RootPath
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
    embedding_layer=Embedding(input_dim=106444,output_dim=512,trainable=False,embeddings_initializer=tf.constant_initializer(embedding_matrix))

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
    group_embedding_dict[DEFAULT_GROUP_NAME].append(embedding_layer(features['feedid']))
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
    embedding_dim=4
    id_feature_list=sparse_features = ['userid', 'feedid', 'device', 'authorid', 'bgm_song_id',
                       'bgm_singer_id']
    dense_feature_list=[]
    data_path = 'user_action_8_version2.csv'
    data_path = os.path.join(RootPath, data_path)
    data_df = pd.read_csv(data_path)
    for col in data_df.columns:
        if col in id_feature_list:
            continue
        if col.find('embedding_')==-1:
            dense_feature_list.append(col)
            print(col)





if __name__=="__main__":
    #generate_input()
    '''
    user_interest=os.path.join(RootPath,"feed_info_modified2.pkl")
    with open(user_interest,'rb') as f:
        interest=pickle.load(f)
    print(interest.columns)
    '''
    a={'userid':[1],'feedid':[np.array([1,2])]}
    a=pd.DataFrame(a)
    print(a)
    for i in range(a.shape[0]):
        a.set_value(i,'feedid',np.array([2,3]))
        a.loc[i]['userid']=2
    print(a)

