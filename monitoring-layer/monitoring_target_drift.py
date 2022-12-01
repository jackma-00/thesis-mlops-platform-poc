import pandas as pd
from evidently.dashboard import Dashboard
from evidently.tabs import CatTargetDriftTab
import mlflow

# Load reference data 
reference_data = pd.read_csv("../data-layer/data/training/data.csv",
                             names=["day{}".format(i) for i in range(0,14)]+["target"])
reference_data = reference_data.drop([0], axis=0)
reference_data = reference_data.reset_index()
reference_data = reference_data.drop(["index"], axis=1)
#print(reference_data)

# Get production scored data
production_scored_data = pd.read_csv("../data-layer/data/scoring/output.csv",
                                names=["day{}".format(i) for i in range(0,14)]+["target"])
production_scored_data = production_scored_data.drop([0], axis=0)
production_scored_data = production_scored_data.reset_index()
production_scored_data = production_scored_data.drop(["index"], axis=1)                                
#print(production_scored_data)

with mlflow.start_run(run_name="monitoring-target-drift") as run:
    mlflow.set_tag("mlflow.runName", "monitoring-target-drift")
    drift_dashboard = Dashboard(tabs=[CatTargetDriftTab(verbose_level=1)])
    drift_dashboard.calculate(reference_data, production_scored_data)
    drift_dashboard.save("reports/target-drift/target_drift.html")
    drift_dashboard._save_to_json("reports/target-drift/target_drift.json")
    mlflow.log_artifacts("reports/target-drift/")