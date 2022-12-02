from __future__ import annotations
import json
from datetime import datetime
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.edgemodifier import Label

def read_status():
    with open('../monitoring-layer/status/status.json') as file:
        retrain = json.load(file)
    return retrain  

with DAG(
    dag_id="retrain_trigger",
    start_date=datetime(2022, 12, 1),
    catchup=False,
    schedule="@daily"
) as dag:

    model_monitoring = BashOperator(
    task_id='model_monitoring',
    bash_command='cd /home/jacopo/Documents/internship/content/projects/thesis-mlops-platform-poc/monitoring-layer/ && make',
    )

    branching = BranchPythonOperator(
        task_id='branching',
        python_callable=read_status,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="training_pipeline",  # Ensure this equals the dag_id of the DAG to trigger
    )

    hold = EmptyOperator(
        task_id='hold',
    )

    join = EmptyOperator(
        task_id='join',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    
    model_monitoring >> branching >> [trigger_retrain, hold] >> join
