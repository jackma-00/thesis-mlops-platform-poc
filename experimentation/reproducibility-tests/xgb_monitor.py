import warnings

import xgboost as xgb
import mlflow
import pandas as pd
from evidently.tabs import ClassificationPerformanceTab
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evidently.dashboard import Dashboard

THRESHOLD = 0.5
TEST_SIZE = 0.3
DATA_RANDOM_STATE = 0

def main():

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Get reference dataset
    reference_data = pd.read_csv("../data-layer/data/training/data.csv",
                             names=["day{}".format(i) for i in range(0,14)]+["target"])
    reference_data = reference_data.drop([0], axis=0)
    reference_data = reference_data.reset_index()
    reference_data = reference_data.drop(["index"], axis=1)

    X = reference_data.iloc[:,:-1]
    y = reference_data.iloc[:,-1]

    reference, production, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=DATA_RANDOM_STATE, stratify=y)

    reference_train = xgb.DMatrix(reference, label=y_train)
    dproduction= xgb.DMatrix(production)
    dreference=xgb.DMatrix(reference)

    # Model training
    mlflow.set_experiment("Experimentation")
    mlflow.xgboost.autolog()
    with mlflow.start_run(run_name="monitoring-model-drift") as run:
        mlflow.set_tag("mlflow.runName", "monitoring-model-drift")

        model=xgb.train(dtrain=reference_train, params={})
    
        train_proba_predict = model.predict(dreference)
        test_proba_predict = model.predict(dproduction)

        test_predictions = [1. if y_cont > THRESHOLD else 0. for y_cont in test_proba_predict]
        train_predictions = [1. if y_cont > THRESHOLD else 0. for y_cont in train_proba_predict]

        reference['target'] = y_train
        reference['prediction'] = train_predictions
        production['target'] = y_test
        production['prediction'] = test_predictions

        classification_performance = Dashboard(tabs=[ClassificationPerformanceTab(verbose_level=1)])
        classification_performance.calculate(reference,production)

        # Evaluation  
        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions)
        recall = recall_score(y_test, test_predictions)
        f1 = f1_score(y_test, test_predictions)
    
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

if __name__ == "__main__":
    main()