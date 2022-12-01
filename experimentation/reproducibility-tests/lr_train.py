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

    mlflow.set_experiment("Experimentation")
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="train_model_lr", nested=True) as run:
        mlflow.set_tag("mlflow.runName", "train_model_lr")

        # Initializing classifier with the best params 
        lr = LogisticRegression(penalty='l2', C=1e-5, class_weight=None)

        # Model training 
        lr.fit(X_train, y_train)

        # Prediction 
        y_probas = lr.predict(X_test)
        y_preds = [1 if  y_proba > THRESHOLD else 0 for y_proba in y_probas]

        # Evaluation  
        accuracy = accuracy_score(y_test, y_preds)
        precision = precision_score(y_test, y_preds)
        recall = recall_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds)
    
        print('Final Testing RESULTS')
        print('/-------------------------------------------------------------------------------------------------------- /')
        print('Accuracy is ', accuracy)
        print('Precision is ', precision)
        print('Recall is ', recall)
        print('F1-Score is ', f1)
        print('/-------------------------------------------------------------------------------------------------------- /')

        # Saving parameters, metrics, and model
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)

if __name__== "__main__":
	main()