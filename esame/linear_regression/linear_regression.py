from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import json
import argparse
import pandas as pd
import numpy
from pathlib import Path


# Define the _linear_regression function
def _linear_regression (args):

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

    # Initialize a Linear Regression model
    lr = LinearRegression()

    # Train the model with the training data
    pd_y_train_t = numpy.squeeze(pd_y_train) #reshape the pd_y_train in order to have a 1d vector
    print(pd_x_train.head)
    lr.fit(pd_x_train,pd_y_train_t)

    # Make predictions on the test set
    y_pred_lr = lr.predict(pd_x_test)

    # Calculate R-squared value for model evaluation
    r_squared = r2_score(pd_y_test,y_pred_lr)
    # Calculate RMSE (Root Mean Squared Error) for model evaluation
    rmse = numpy.sqrt(mean_squared_error(pd_y_test,y_pred_lr))
    # Preparing final metrics string for output
    r_squared_str = f"R-squared: {r_squared}"
    rmse_str = f"RMSE: {rmse}"
    final_metrics = r_squared_str + ", " + rmse_str

    # Writing the final metrics to a specified output file
    with open(args.lr_error, 'w') as error_file:
        error_file.write(str(final_metrics))




if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Linear regression model')
    #Input
    parser.add_argument('--data', type=str)
    #Output
    parser.add_argument('--lr_error', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.lr_error).parent.mkdir(parents=True, exist_ok=True)
    
    _linear_regression(args)