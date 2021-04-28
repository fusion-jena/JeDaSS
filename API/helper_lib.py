import yaml
import numpy as np 
import json
import sys
import os



#open the config file as read only and return it
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


#returns modified dataframe after removing units from the 1st row if present 
#takes as input the dataframe and the list of units from config
def remove_units(df,remove):
    row = df.iloc[0].tolist()
    commons = np.intersect1d(row,remove).tolist()
    if len(commons)>0: 
        df.drop(0,inplace=True)
        df.reset_index(inplace = True, drop = True)
    return df


#returns amount of not nulls in dataframe column
def get_dataframeColumn_notNull(df, col):
    return df[col].count()


#return amount of nulls in dataframe column
def get_dataframeColumn_Null(df, col):
    return len(df) - df[col].count()


#return a pandas series with counts of nulls per column
def get_seriesof_nulls_perColumn(x):
    return x.isnull().sum() 


# used in prediction() function inside ClassifierSemantic.py
# prediction() is called from the API 
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# block all print() commands after call
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# enable all print() commands after call 
def enablePrint():
    sys.stdout = sys.__stdout__