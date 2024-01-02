from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json
import argparse
import pandas as pd
import numpy
from pathlib import Path

def _linear_regression (args):
    with open(args.data) as file:
        data = json.load(file)

    x_train = data ['x_train']
    x_test = data ['x_test']
    y_train = data ['y_train']
    y_test = data ['y_test']

    pd_x_train = pd.DataFrame(x_train)
    pd_x_test = pd.DataFrame(x_test)
    pd_y_train = pd.DataFrame(y_train)
    pd_y_test = pd.DataFrame(y_test)

    rf = RandomForestRegressor()

    # Training Model
    rf.fit(pd_x_train,pd_y_train)

    # Model Summary
    y_pred_rf = rf.predict(pd_x_test)

    r_squared = r2_score(pd_y_test,y_pred_rf)
    rmse = numpy.sqrt(mean_squared_error(pd_y_test,y_pred_rf))
    r_squared_str = f"R-squared: {r_squared}"
    rmse_str = f"RMSE: {rmse}"
    final_metrics = r_squared_str + ", " + rmse_str

    #print("R_squared :",r_squared)
    #print("RMSE :",rmse)
    with open(args.rf_error, 'w') as error_file:
        error_file.write(str(final_metrics))




if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Linear regression model')
    parser.add_argument('--data', type=str)
    parser.add_argument('--rf_error', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.rf_error).parent.mkdir(parents=True, exist_ok=True)
    
    _linear_regression(args)