import pandas as pd
import numpy as np
from ast import literal_eval
import os
from scipy.spatial import distance
import pickle
from main import RootPath
import tensorflow.compat.v1 as tf
from several_layers import mmoe_layer,CrossNetMix

def test_data():
    pd.set_option('display.max_rows', None)
    data_path='user_action_8_version2.csv'
    data_path=os.path.join(RootPath,data_path)
    data_df=pd.read_csv(data_path)
    for column in data_df.columns:
        print(column)

def test_layers():
    a = tf.constant([[1, 2, 3]], dtype=tf.float32)
    #my_layer=mmoe_layer(3,2,1)
    my_layer=CrossNetMix()
    out_put=my_layer(a)

if __name__=="__main__":
   test_layers()
