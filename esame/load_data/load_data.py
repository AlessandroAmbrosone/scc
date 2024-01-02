import json
import argparse
import pandas as pd
import numpy as np 
from pathlib import Path
from sklearn.model_selection import train_test_split

def _load_data(args):

    # Loading Training Data
    train_data_file = "car_price_data1.csv"
    

    df_X_train = pd.read_csv(train_data_file)
    
    df_X_train.drop(['Model'], axis=1, inplace=True)
    df_X_train.dropna()
    q = df_X_train['Price'].quantile(0.99)
    data_1 = df_X_train[df_X_train['Price']<q] #nuovo dataframe
    q1 = data_1['Mileage'].quantile(0.99)
    data_2 = data_1[data_1['Mileage']<q1] #nuovo dataframe
    q2 = data_2['Year'].quantile(0.01)
    data_3 = data_2[data_2['Year']>q2] #nuovo dataframe
    data_4 = data_3[data_3['EngineV']<6.5] #cilindrata espressa in l
    cleaned_data = data_4.reset_index(drop=True)

    #trasformo prezzi in log
    log_price = np.log(cleaned_data['Price'])

    # Then we add it to our data frame
    cleaned_data['log_price'] = log_price
    
    cleaned_data = cleaned_data.drop(['Price'],axis=1) 
    
    
    
    #cleaned_data_num = cleaned_data.to_numpy()
    #data = {'dataframe': cleaned_data_num.tolist()}
    cleaned_data.to_json(args.data,orient='columns')

    # Creates `data` structure to save and 
    # share train and test datasets.
    #data = {'x_train': x_train.tolist(), 'y_train': y_train.tolist(), 'x_test': x_test.tolist(), 'y_test': y_test.tolist()}

    # Creates a json object based on `data`
    #data_json = json.dumps(data)

    # Saves the json object into a file
    #with open(args.data, 'w') as out_file:
        #json.dump(data_json, out_file)

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