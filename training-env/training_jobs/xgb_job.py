import warnings

import pandas as pd
import mlflow
import xgboost as xgb
import mlflow.xgboost
from sklearn.model_selection import train_test_split

THRESHOLD = 0.5
TEST_SIZE = 0.3
DATA_RANDOM_STATE = 0

def train():

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Load the csv dataset
    df = pd.read_csv("../data-layer/data/training/data.csv")

    # Independent and Dependent features
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    # Train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=DATA_RANDOM_STATE)

    mlflow.xgboost.autolog()
    with mlflow.start_run(run_name="train-model-xgboost") as run:
        mlflow.set_tag("mlflow.runName", "train-model-xgboost")

        # Initializing data matrices
        train_data = xgb.DMatrix(X_train, label=y_train)
        test_data =  xgb.DMatrix(X_test)

        # Training xgboost classifier
        model = xgb.train(dtrain=train_data, params={})
        
        y_probas = model.predict(test_data) 
        y_preds = [1 if  y_proba > THRESHOLD else 0 for y_proba in y_probas]

        test_prediction_results = pd.DataFrame(data={'y_test':y_test, 'y_pred':y_preds})
        result = test_prediction_results.reset_index(drop=True)
        result.to_csv("../data-layer/data/predictions/test_predictions.csv", index=False)

        # Saving train_test_split metrics
        mlflow.log_param("stratify", False)
        mlflow.log_param("data_random_state", DATA_RANDOM_STATE)

if __name__ == "__main__":
    train()
        

        
        