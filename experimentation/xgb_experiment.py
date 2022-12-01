import warnings

import pandas as pd
import mlflow
import xgboost as xgb
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

THRESHOLD = 0.5
TEST_SIZE = 0.3
DATA_RANDOM_STATE = 0

def main():

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Load the csv dataset
    df = pd.read_csv("../data-layer/data/training/data.csv")
    print(f'\nUsing dataset:\n')
    print(df.head())
    print(df.tail())

    # Independent and Dependent features
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    # Train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=DATA_RANDOM_STATE)

    mlflow.set_experiment("Experimentation")
    with mlflow.start_run(run_name="xgboost-experimental-run") as run:
        mlflow.set_tag("mlflow.runName", "xgboost-experimental-run")

        # Initializing data matrices
        train_data = xgb.DMatrix(X_train, label=y_train)
        test_data =  xgb.DMatrix(X_test)

        print("x test:\n", X_test)
        print("test data:\n", test_data)

        # Training xgboost classifier
        model = xgb.train(dtrain=train_data, params={})
        print(f'\nTraining model: xgboost\n')
        
        y_probas = model.predict(test_data) 
        y_pred = [1 if  y_proba > THRESHOLD else 0 for y_proba in y_probas]

        # Evaluation  
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
        print('Final Testing RESULTS')
        print('/-------------------------------------------------------------------------------------------------------- /')
        print('Accuracy is ', accuracy)
        print('Precision is ', precision)
        print('Recall is ', recall)
        print('F1-Score is ', f1)
        print('/-------------------------------------------------------------------------------------------------------- /')

        # Saving parameters, metrics, and model
        mlflow.log_param("stratify", False)
        mlflow.log_param("data_random_state", DATA_RANDOM_STATE)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)
        mlflow.xgboost.log_model(model, "model")

if __name__ == "__main__":
    main()
        