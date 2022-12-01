import xgboost as xgb
import mlflow
import pandas as pd
from evidently.tabs import ClassificationPerformanceTab
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
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

# Model training
THRESHOLD = 0.5
with mlflow.start_run(run_name="monitoring-model-drift") as run:
    mlflow.set_tag("mlflow.runName", "monitoring-model-drift")

    # Initializing classifier 
    lr = LogisticRegression()

    # Tuning parameters via grid search
    hyperparameters = {"penalty": ['l1', 'l2'],
                       "C": [1e-5, 1e-4, 1e-3, 1e-2, 1, 1e1, 1e2, 1e3],
                       "class_weight": [None, 'balanced']}
    gs = GridSearchCV(estimator=lr,
                      param_grid=hyperparameters,
                      scoring="accuracy",
                      cv=5)
    gs.fit(reference, y_train)

    
    train_proba_predict = gs.predict(reference)
    test_proba_predict = gs.predict(production)

    test_predictions = [1. if y_cont > THRESHOLD else 0. for y_cont in test_proba_predict]
    train_predictions = [1. if y_cont > THRESHOLD else 0. for y_cont in train_proba_predict]

    reference['target'] = y_train
    reference['prediction'] = train_predictions
    production['target'] = y_test
    production['prediction'] = test_predictions

    classification_performance = Dashboard(tabs=[ClassificationPerformanceTab(verbose_level=1)])
    classification_performance.calculate(reference, production)
    classification_performance.save("reports/model-drift/model_drift.html")
    
    # Saving parameters, report, and model
    mlflow.log_artifact("reports/model-drift/")
    mlflow.log_params(gs.best_params_)
    mlflow.sklearn.log_model(gs, "model")


