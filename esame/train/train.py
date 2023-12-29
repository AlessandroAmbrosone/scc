from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import json
import argparse
import pandas as pd
from pathlib import Path
import joblib 


def align_features(input_data, trained_columns):
    # Adding missing columns
    for col in trained_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reordering columns
    return input_data[trained_columns]




def _train(args):
    # Open and reads file "data"
    with open(args.processed_data) as data_file:
        data = json.load(data_file)
    with open(args.best_parameter) as bp_file:
        best_p = json.load(bp_file)
    
    # Finding the key with the minimum value and extracting the parts
    min_key = min(best_p, key=best_p.get)
    n_estimators, lr = min_key.split('_')
    
    # Convert n_estimators to int and lr to float
    n_estimators = int(n_estimators)
    lr = float(lr)

    print("Minimum Value Variables:", n_estimators, lr)

    # Data preparation
    X_train, y_train,X_test = data['x_train'], data['y_train'],data['x_test']
    df_x, df_y, df_x_test= pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test)

    #La predict del modello non va a buon fine, poichè il numero di feaures del test è diverso da quello del train, ho dato un occhiata veloce
    #forse errore potrebbe essere in load_data.py, poiihcè su X_train facciamo dei drop e su x_test no, ma vedendo meglio forse non c'è bisogno 
    #quindi bohhhh!!!!
    

    # Model initialization and fitting
    xgb_regressor_model = XGBRegressor(n_estimators=n_estimators,
                                       learning_rate=lr,
                                       random_state=0,
                                       n_jobs=4)
    xgb_regressor_model.fit(df_x, df_y)
    trained_columns = df_x.columns.tolist()
    # Usage
    X_test_prepared = align_features(df_x_test, trained_columns)
    print(df_x.shape[1], df_x_test.shape[1])
    
    # Save the model
    joblib.dump(xgb_regressor_model, args.model)
    print(f"Model saved as {args.model}")
    # Replace this with the path to your saved model
    #model_path = 'path/to/your/saved_model.bin'

    # Load the model
    #loaded_model = joblib.load(model_path)

    #Prediction 
    predicts = xgb_regressor_model.predict(X_test_prepared)
    print("preds:", predicts)

if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My process data')
    parser.add_argument('--processed_data', type=str)
    parser.add_argument('--best_parameter', type=str)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    
    _train(args)