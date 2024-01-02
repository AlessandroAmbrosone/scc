import json
import argparse
import pandas as pd
import numpy as np 
from pathlib import Path

def _load_data(args):

    # Loading Training Data
    train_data_file = "car_price_data1.csv"
    
    #read csv file 
    df_X_train = pd.read_csv(train_data_file)
    
    #Drop the 'Model' column , cause in useless
    df_X_train.drop(['Model'], axis=1, inplace=True)
    #Drop the the row where there is at least one null value
    df_X_train.dropna()
    
    #In order to reduce the outlayer we calculate the percentile 
    q_price = df_X_train['Price'].quantile(0.99)
    data_1 = df_X_train[df_X_train['Price']<q_price] 

    q_mileage = data_1['Mileage'].quantile(0.99)
    data_2 = data_1[data_1['Mileage']<q_mileage] #nuovo dataframe

    q_year = data_2['Year'].quantile(0.01)
    data_3 = data_2[data_2['Year']>q_year] 



    # The situation with engine volume is very strange
    # In such cases it makes sense to manually check what may be causing the problem
    # In our case the issue comes from the fact that most missing values are indicated with 99.99 or 99
    # There are also some incorrect entries like 75
    # Car engine volumes are usually  below 6.5l
    data_4 = data_3[data_3['EngineV']<6.5] 

    cleaned_data = data_4.reset_index(drop=True)

    #See The patterns are quite exponentials in this condition log transformation is a common way to deal with this issue. 
    #log transformation is especially useful when facing exponential value.
    #Let's transform 'Price' with a log transformation
    log_price = np.log(cleaned_data['Price'])

    # Then we add it to our data frame
    cleaned_data['log_price'] = log_price
    
    cleaned_data = cleaned_data.drop(['Price'],axis=1) 
    
    #Save the data in our output path
    cleaned_data.to_json(args.data,orient='columns')


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