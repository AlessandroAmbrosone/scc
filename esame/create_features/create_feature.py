

#dovremmo vedere come passare le cose da load_data a qui , credo basti caricare il json, vedere come fa in decision_tree il prof
#magari questa parte si può anche mettere in load_data, anche se queste colonne gli servono poi per il modello, quindi boh..


# Checking columns

low_cardinality_columns = [col for col in X.columns if X[col].nunique() < 10 and X[col].dtype == "object"]
num_columns = [col for col in X.columns if X[col].dtype in ["int64", "float64"]]

required_columns = low_cardinality_columns + num_columns
high_cardinality_columns = [col for col in X.columns if col not in required_columns]

print(required_columns)
print("Dropped_columns", high_cardinality_columns)
