import xgboost as xgb
import mlflow
import json
import pandas as pd
from evidently.tabs import ClassificationPerformanceTab
from sklearn.model_selection import train_test_split
from evidently.dashboard import Dashboard
from sklearn.metrics import  accuracy_score
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient

MODEL_NAME = "xgb-psystock"
MODEL_VERSION = 2
MODEL_STAGE = "Production"
THRESHOLD = 0.5

# Initialize MLflow client 
client = MlflowClient()

def get_ref_data():
    
    # Get reference data
    reference_X_test = pd.read_csv("../data-layer/data/monitoring/reference_X_test.csv", header=None,
                                names=["{}".format(i) for i in range(0, 14)])
    reference_y_test = pd.read_csv("../data-layer/data/monitoring/reference_y_test.csv", header=None)

    #print(reference_X_test)
    #print(reference_y_test)

    return [reference_X_test, reference_y_test]

def get_prod_data():

    # Get production data
    production_X_test = pd.read_csv("../data-layer/data/scoring/input.csv", header=None,
                                    names=["{}".format(i) for i in range(0, 14)])

    production_y_test = pd.read_csv("../data-layer/data/scoring/ground_truth.csv", header=None)

    #print(production_X_test)
    #print(production_y_test)

    return [production_X_test, production_y_test]

if __name__ == "__main__":

    # Model performance evaluation
    with mlflow.start_run(run_name="monitoring-model-drift") as run:
        mlflow.set_tag("mlflow.runName", "monitoring-model-drift")

        model_name = MODEL_NAME
        model_stage = MODEL_STAGE

        # Get the latest model version in production 
        model_version = client.get_latest_versions(model_name, stages=[model_stage])[0].version
        model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name, model_stage=model_stage)
        # Load the production model 
        model = mlflow.pyfunc.load_model(model_uri=model_uri) 

        # Log model's params 
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_stage", model_stage)

        # Get the data to score
        reference_X_test, reference_y_test = get_ref_data()
        production_X_test, production_y_test = get_prod_data()

        # Perform predictions 
        reference_probas = model.predict(reference_X_test)
        production_probas = model.predict(production_X_test)

        reference_predictions = [1. if y_cont > THRESHOLD else 0. for y_cont in reference_probas]
        production_predictions = [1. if y_cont > THRESHOLD else 0. for y_cont in production_probas]

        # Accuracy test
        reference_accuracy = accuracy_score(reference_y_test, reference_predictions)
        production_accuracy = accuracy_score(production_y_test, production_predictions)
        print(f'ref acc: {reference_accuracy}\n')
        print(f'prod acc: {production_accuracy}\n')
        mlflow.log_metric("ref_acc", reference_accuracy)
        mlflow.log_metric("prod_acc", production_accuracy)

        # Evidently AI Dashboard
        reference_X_test['target'] = reference_y_test
        reference_X_test['prediction'] = reference_predictions
        production_X_test['target'] = production_y_test
        production_X_test['prediction'] = production_predictions

        classification_performance = Dashboard(tabs=[ClassificationPerformanceTab(verbose_level=1)])
        classification_performance.calculate(reference_X_test, production_X_test)
        classification_performance.save("reports/model-drift/model_drift.html")
        mlflow.log_artifact("reports/model-drift/model_drift.html")

        # Saving status [trigger_retrain | hold]
        with open('status/status.json', 'w') as file:
            if production_accuracy < reference_accuracy:
                retrain = "trigger_retrain"
            else: 
                retrain = "hold"
            json.dump(retrain, file) 


