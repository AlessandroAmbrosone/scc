import json
import argparse
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




def _process_data(args):

    
    #Read the data file form inputh path
    pd_X = pd.read_json(args.data,orient='columns')
    
    
    #Creating Dummie Variables (One hot encoding)
    X_with_dummies = pd.get_dummies(pd_X,dtype=int)
    

    #Drop the Price column cause is useless
    X = X_with_dummies.drop('log_price',axis=1)
    #Save the label into y
    y = X_with_dummies['log_price']
    
    #StandardScaler is used to standardize the features by removing the mean and scaling to unit variance. 
    #This is achieved by subtracting the mean value of each feature and then dividing by the standard deviation
    scaler = StandardScaler()
    #this step is preparing the scaler to standardize the 'Mileage' and 'EngineV' columns. 
    #It calculates and stores the necessary statistics (mean and standard deviation) based on the data in these columns
    scaler.fit(X[['Mileage','EngineV']])
    joblib.dump(scaler, 'scaler.joblib')
    
    # It is not usually recommended to standardize dummy variables
    #For ML purposes we rarely put too much thought into it and go with the scale dummies as 
    #scaling has no effect on their predictive power.
    #Uses the scaler object to standardize the 'Mileage' and 'EngineV' features of the dataset X
    #Applies the scaling transformation based on the mean and standard deviation calculated in the fit method . 
    #This results in scaled versions of these features where each feature's mean is 0 and standard deviation is 1.
    inputs_scaled = scaler.transform(X[['Mileage','EngineV']])
    # This line converts inputs_scaled into a Pandas DataFrame.
    #The columns parameter names the columns as 'Mileage' and 'EngineV', reflecting the original names of the features that were scaled.
    scaled_data = pd.DataFrame(inputs_scaled,columns=['Mileage','EngineV'])
    # X.drop(['Mileage','EngineV'], axis=1) removes the original 'Mileage' and 'EngineV' columns from the dataset X, keeping all other features intact.
    #scaled_data.join(...) adds the columns from the result of the drop operation to scaled_data. 
    #This effectively replaces the original 'Mileage' and 'EngineV' columns in X with their scaled versions.
    #The final result, input_scaled2, is a DataFrame that includes the scaled 'Mileage' and 'EngineV' features alongside the other, 
    #unscaled features of the original dataset X.
    input_scaled2 = scaled_data.join(X.drop(['Mileage','EngineV'],axis=1))
    
    print(input_scaled2.columns)
    x_train, x_test, y_train, y_test = train_test_split(input_scaled2,y,test_size=0.2, random_state=365)

    # Creates a json object based on `data`
    x_train_n,x_test_n,y_train_n,y_test_n = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(),  y_test.to_numpy()
    
    processed_data = {'x_train': x_train_n.tolist(),'y_train' : y_train_n.tolist(),'x_test': x_test_n.tolist(), 'y_test' : y_test_n.tolist()} #forse da aggiungere qui anche X_test, se no come predicto dopo in train??

    # Saves the json object into a file
    with open(args.processed_data, 'w') as out_file:
        json.dump(processed_data, out_file)


if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Process the data and split into train_set and test_set')
    #Input
    parser.add_argument('--data', type=str)
    #Output
    parser.add_argument('--processed_data', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.processed_data).parent.mkdir(parents=True, exist_ok=True)
    
    _process_data(args)