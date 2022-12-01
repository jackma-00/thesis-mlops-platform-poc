import xgboost as xgb
import mlflow
import pandas as pd
from evidently.tabs import ClassificationPerformanceTab
from sklearn.model_selection import train_test_split
from evidently.dashboard import Dashboard

# Get reference dataset
reference_data = pd.read_csv("../data-layer/data/training/data.csv",
                             names=["day{}".format(i) for i in range(0,14)]+["target"])
reference_data = reference_data.drop([0], axis=0)
reference_data = reference_data.reset_index()
reference_data = reference_data.drop(["index"], axis=1)

X=reference_data.iloc[:,:-1]
Y=reference_data.iloc[:,-1]

reference, production, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

reference_train = xgb.DMatrix(reference, label=y_train)
dproduction= xgb.DMatrix(production)
dreference=xgb.DMatrix(reference)

# Model training
THRESHOLD = 0.5
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
    classification_performance.save("reports/model-drift/model_drift.html")
    mlflow.log_artifact("reports/model-drift/")


