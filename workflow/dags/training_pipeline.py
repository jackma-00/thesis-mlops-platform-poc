from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'Jack',
    #'depends_on_past': False,
    'start_date': datetime(2022, 12, 1),
    #'email': ['example@example.com'],
    #'email_on_failure': False,
    #'email_on_retry': False,
    #'retries': 1,
    #'retry_delay': timedelta(minutes=2),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

# Instantiates a directed acyclic graph
dag = DAG(
    'training_pipeline',
    default_args=default_args,
    description='Machine Learning data processing plus model training pipeline',
    schedule='@once',
)

# Instantiate tasks using Operators
data_processing = BashOperator(
    task_id='data_processing',
    bash_command='cd /home/jacopo/Documents/internship/content/projects/mlops-platform/data-layer/ && make',
    dag=dag,
)

model_training = BashOperator(
    task_id='model_training',
    bash_command='cd /home/jacopo/Documents/internship/content/projects/mlops-platform/training-env/ && make',
    dag=dag,
)

# Sets the ordering of the DAG
data_processing >> model_training