import pandas as pd
import numpy as np
from ast import literal_eval
import os
from scipy.spatial import distance
import pickle
from main import RootPath

if __name__=="__main__":
    pd.set_option('display.max_rows', None)
    data_path='user_action_8_version2.csv'
    data_path=os.path.join(RootPath,data_path)
    data_df=pd.read_csv(data_path)
    for column in data_df.columns:
        print(column)