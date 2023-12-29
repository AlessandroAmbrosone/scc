import json
import argparse
import pandas as pd
import numpy as np 
from pathlib import Path

from sklearn.model_selection import train_test_split

def _load_data(args):

    # Loading Training and Test Data
    train_data_file = "train.csv"
    #test_data_file = "test.csv"

    df_X_train = pd.read_csv(train_data_file, index_col='Id')
    #X_test = pd.read_csv(test_data_file, index_col='Id')

    df_X_train.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y_train = df_X_train.SalePrice              
    df_X_train.drop(['SalePrice'], axis=1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(df_X_train, y_train, test_size=0.2)
    #x_train.head()
    
    x_train, x_test, y_train , y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    # Creates `data` structure to save and 
    # share train and test datasets.
    data = {'x_train': x_train.tolist(), 'y_train': y_train.tolist(), 'x_test': x_test.tolist(), 'y_test': y_test.tolist()}

    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(args.data, 'w') as out_file:
        json.dump(data_json, out_file)

if __name__ == '__main__':
    
    # This component does not receive any input
    # it only outpus one artifact which is `data`.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    
    args = parser.parse_args()
    
    # Creating the directory where the output file will be created 
    # (the directory may or may not exist).
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    _load_data(args)