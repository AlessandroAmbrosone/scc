import json
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#dovremmo vedere come passare le cose da load_data a qui , credo basti caricare il json, vedere come fa in decision_tree il prof
#magari questa parte si può anche mettere in load_data, anche se queste colonne gli servono poi per il modello, quindi boh..


# Checking columns
def _process_data(args):

    # Open and reads file "data"
    #with open(args.data) as data_file:
        #data = json.load(data_file)
    
    # The excted data type is 'dict', however since the file
    # was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get 
    # the dict-type object.
    #data = json.loads(data)

    pd_X = pd.read_json(args.data,orient='columns')
    
    #pd_X = data['dataframe']
    #pd_X = pd.DataFrame(pd_X)

    X_with_dummies = pd.get_dummies(pd_X,dtype=int)


    X = X_with_dummies.drop('log_price',axis=1)
    y = X_with_dummies['log_price']
    

    scaler = StandardScaler()
    scaler.fit(X[['Mileage','EngineV']])

    
    # It is not usually recommended to standardize dummy variables
    #For ML purposes we rarely put too much thought into it and go with the scale dummies as 
    #scaling has no effect on their predictive power.
    inputs_scaled = scaler.transform(X[['Mileage','EngineV']])
    scaled_data = pd.DataFrame(inputs_scaled,columns=['Mileage','EngineV'])
    input_scaled2 =scaled_data.join(X.drop(['Mileage','EngineV'],axis=1))
    

    x_train, x_test, y_train, y_test = train_test_split(input_scaled2,y,test_size=0.2, random_state=365)

    # Creates a json object based on `data`
    x_train_n,x_test_n,y_train_n,y_test_n = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(),  y_test.to_numpy()
    
    processed_data = {'x_train': x_train_n.tolist(),'y_train' : y_train_n.tolist(),'x_test': x_test_n.tolist(), 'y_test' : y_test_n.tolist()} #forse da aggiungere qui anche X_test, se no come predicto dopo in train??

    # Saves the json object into a file
    with open(args.processed_data, 'w') as out_file:
        json.dump(processed_data, out_file)


if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My process data')
    parser.add_argument('--data', type=str)
    parser.add_argument('--processed_data', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.processed_data).parent.mkdir(parents=True, exist_ok=True)
    
    _process_data(args)