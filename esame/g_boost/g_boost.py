from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json
import argparse
import pandas as pd
import numpy
import joblib
from pathlib import Path


# Defining the function _gboost_regressor
def _gboost_regressor (args):

    # Opening the JSON file containing the data
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
    
    # Define a parameter grid to search over
    param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
    }
    # Initializing the Gradient Boosting Regressor model
    gbt = GradientBoostingRegressor()

    # Initialize Grid Search with the Gradient Boosting Regressor model
    grid_search = GridSearchCV(gbt, param_grid, cv=5, scoring='neg_mean_squared_error')
    # Training the model on the training data
    pd_y_train_t = numpy.squeeze(pd_y_train) #reshape the pd_y_train in order to have a 1d vector

    grid_search.fit(pd_x_train,pd_y_train_t)

    # The best estimator after grid search
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, 'trained_gbt_model.joblib')  

    # Predicting the target variable for the test set
    y_pred_gbt = best_model.predict(pd_x_test)
    print("Best parameters:", grid_search.best_params_)

    # Calculating R-squared value for model evaluation
    r_squared = r2_score(pd_y_test,y_pred_gbt)
    # Calculating RMSE (Root Mean Squared Error) for model evaluation
    rmse = numpy.sqrt(mean_squared_error(pd_y_test,y_pred_gbt))
    # Preparing final metrics string for output
    r_squared_str = f"R-squared: {r_squared}"
    rmse_str = f"RMSE: {rmse}"
    final_metrics = r_squared_str + ", " + rmse_str

    # Writing the final metrics to a specified output file
    with open(args.gb_error, 'w') as error_file:
        error_file.write(str(final_metrics))




if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Gradient Boosting Regressor model')
    #Input 
    parser.add_argument('--data', type=str)
    #Output
    parser.add_argument('--gb_error', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.gb_error).parent.mkdir(parents=True, exist_ok=True)
    
    _gboost_regressor(args)