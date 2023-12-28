from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import json
import argparse
import pandas as pd
from pathlib import Path



def _get_scores(args,n_estimators, learning_rate):
        # Open and reads file "data"
    with open(args.processed_data) as data_file:
        data = json.load(data_file)
    
    # The excted data type is 'dict', however since the file
    # was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get 
    # the dict-type object.
    #data = json.loads(data)

    X_train = data['x_train']
    y_train = data['y_train']
    df_x = pd.DataFrame(X_train)
    df_y = pd.DataFrame(y_train)
    xgb_regressor_model = XGBRegressor(n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       random_state=0,
                                       n_jobs=4)

    
    
    scores = -1 * cross_val_score(xgb_regressor_model, df_x, df_y, cv=3, scoring="neg_mean_absolute_error")
    
    return scores.mean()

# Training the model and finding the appropriate values

def _collect_result(args):
    results = {}

    for i in range(8, 13):
        for j in range(3):
            key = f"{100*i}_{0.04 + 0.01*j}"  # Convert tuple to a string key
            results[key] = _get_scores(args, 100*i, 0.04 + 0.01*j)
    

    # Saves the json object into a file
    with open(args.results, 'w') as out_file:
        json.dump(results, out_file)



if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My process data')
    parser.add_argument('--processed_data', type=str)
    parser.add_argument('--results', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    
    _collect_result(args)