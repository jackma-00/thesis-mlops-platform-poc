import pandas as pd
import mlflow
from sklearn.metrics import  \
    accuracy_score, \
    f1_score, \
    precision_score, \
    recall_score


def classification_metrics(df):
    metrics={}
    metrics["accuracy_score"] = accuracy_score(df["y_test"], df["y_pred"])
    metrics["precision_score"] = precision_score(df["y_test"], df["y_pred"])
    metrics["recall_score"] = recall_score(df["y_test"], df["y_pred"])
    metrics["f1_score"] = f1_score(df["y_test"], df["y_pred"])
    return metrics
    
if __name__ == "__main__":

    with mlflow.start_run(run_name="evaluate_model", nested=True) as run:
        mlflow.set_tag("mlflow.runName", "evaluate_model")
        df = pd.read_csv("../data-layer/data/predictions/test_predictions.csv")
        metrics = classification_metrics(df)
        mlflow.log_metrics(metrics)