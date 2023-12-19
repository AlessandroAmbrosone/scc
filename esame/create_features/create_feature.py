from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
import json
import argparse
import pandas as pd
from pathlib import Path

#dovremmo vedere come passare le cose da load_data a qui , credo basti caricare il json, vedere come fa in decision_tree il prof
#magari questa parte si può anche mettere in load_data, anche se queste colonne gli servono poi per il modello, quindi boh..


# Checking columns
def _process_data(args):

    # Open and reads file "data"
    with open(args.data) as data_file:
        data = json.load(data_file)
    
    # The excted data type is 'dict', however since the file
    # was loaded as a json object, it is first loaded as a string
    # thus we need to load again from such string in order to get 
    # the dict-type object.
    data = json.loads(data)

    

    X_train = data['x_train']
    df_x_train = pd.DataFrame(X_train)

    low_cardinality_columns = [col for col in df_x_train.columns if df_x_train[col].nunique() < 10 and df_x_train[col].dtype == "object"]
    num_columns = [col for col in df_x_train.columns if df_x_train[col].dtype in ["int64", "float64"]]

    required_columns = low_cardinality_columns + num_columns
    high_cardinality_columns = [col for col in df_x_train.columns if col not in required_columns]

    #print(required_columns)
    #print("Dropped_columns", high_cardinality_columns)


    #processor

    numerical_transformer = SimpleImputer(strategy="constant")

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("one_hot_encoding", OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_columns),
            ("categorical", categorical_transformer, low_cardinality_columns)
        ])

    processed_data = preprocessor.fit_transform(df_x_train)
    # Creates a json object based on `data`
    processed_data = {'x_train': processed_data.tolist()}

    # Saves the json object into a file
    with open(args.process_data, 'w') as out_file:
        json.dump(processed_data, out_file)


if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My process data')
    parser.add_argument('--data', type=str)
    parser.add_argument('--process_data', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.process_data).parent.mkdir(parents=True, exist_ok=True)
    
    _process_data(args)