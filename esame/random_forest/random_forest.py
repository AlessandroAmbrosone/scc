from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json
import argparse
import pandas as pd
import numpy
from pathlib import Path


# Define the _random_forest function
def _random_forest (args):

    # Open and load the data file specified in args
    with open(args.data) as file:
        data = json.load(file)

    
    # Extracting training and test sets for features and target variable
    x_train = data ['x_train']
    x_test = data ['x_test']
    y_train = data ['y_train']
    y_test = data ['y_test']

    # Converting lists to pandas DataFrames for easier manipulation
    pd_x_train = pd.DataFrame(x_train)
    pd_x_test = pd.DataFrame(x_test)
    pd_y_train = pd.DataFrame(y_train)
    pd_y_test = pd.DataFrame(y_test)


    # Initialize a Random Forest Regressor model
    rf = RandomForestRegressor()

    # Train the model with the training data
    rf.fit(pd_x_train,pd_y_train)

    # Make predictions on the test set
    y_pred_rf = rf.predict(pd_x_test)

    # Calculate R-squared value for model evaluation
    r_squared = r2_score(pd_y_test,y_pred_rf)
    # Calculate RMSE (Root Mean Squared Error) for model evaluation
    rmse = numpy.sqrt(mean_squared_error(pd_y_test,y_pred_rf))
    # Preparing final metrics string for output
    r_squared_str = f"R-squared: {r_squared}"
    rmse_str = f"RMSE: {rmse}"
    final_metrics = r_squared_str + ", " + rmse_str

   # Writing the final metrics to a specified output file
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
    
    _random_forest(args)