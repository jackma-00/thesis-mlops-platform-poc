import pandas as pd
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
import mlflow

# Load reference data 
reference_data = pd.read_csv("../data-layer/data/training/data.csv",
                             names=["day{}".format(i) for i in range(0,14)]+["target"])
reference_data = reference_data.drop([0], axis=0)
reference_data = reference_data.reset_index()
reference_data = reference_data.drop(["index"], axis=1)
#print(reference_data)

# Get production input data
latest_input_data = pd.read_csv("../data-layer/data/scoring/input.csv", header=None,
                                names=["day{}".format(i) for i in range(0,14)])
#print(latest_input_data)

with mlflow.start_run(run_name="monitoring-data-drift") as run:
    mlflow.set_tag("mlflow.runName", "monitoring-data-drift")
    drift_dashboard = Dashboard(tabs=[DataDriftTab(verbose_level=1)])
    drift_dashboard.calculate(reference_data.iloc[:,:14], latest_input_data)
    drift_dashboard.save("reports/data-drift/input_data_drift.html")
    drift_dashboard._save_to_json("reports/data-drift/input_data_drift.json")
    mlflow.log_artifacts("reports/data-drift/")
