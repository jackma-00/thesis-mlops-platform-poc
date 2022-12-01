import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

THRESHOLD = 0.5
TEST_SIZE = 0.3
DATA_RANDOM_STATE = 0    

def main():

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Load the csv dataset 
    df = pd.read_csv("../data-layer/data/training/data.csv")

    # Independent and Dependent features
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]

    # Train test split 
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=TEST_SIZE, random_state=DATA_RANDOM_STATE)

    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="train_model_lr", nested=True) as run:
        mlflow.set_tag("mlflow.runName", "train_model_lr")

        # Initializing classifier with the best params 
        lr = LogisticRegression(penalty='l2', C=1e-5, class_weight=None)

        # Model training 
        lr.fit(X_train, y_train)

        # Prediction 
        y_probas = gs.predict(X_test)
        y_preds = [1 if  y_proba > THRESHOLD else 0 for y_proba in y_probas]

        test_prediction_results = pd.DataFrame(data={'y_test':y_test, 'y_pred':y_preds})
        result = test_prediction_results.reset_index(drop=True)
        result.to_csv("../data-layer/data/predictions/test_predictions.csv", index=False)

        # Saving train_test_split metrics
        mlflow.log_param("stratify", False)
        mlflow.log_param("data_random_state", DATA_RANDOM_STATE)

if __name__== "__main__":
	main()