# Thesis MLOps Platform PoC
The platform contains the following components as described by the architecture diagram.
![image](https://user-images.githubusercontent.com/91907141/200011595-a9502ccb-487a-41c8-a747-9babd67bc7c1.png)

## MLOps level 1: continuous training
![image](https://user-images.githubusercontent.com/91907141/205116728-45cb827d-0563-4ff5-9ec0-9106abc3ff74.png)
- You go from deploying a model to deploying a pipeline that automatically trains models.
- It allows to accelerate the experimentation of different models (e.g., trained on different data).
- Operational symmetry: The same pipeline (same code) can be used in both production and development environments.
- Forces modularization of pipeline components and allows them to be developed and replaced independently.

## MLflow
This project largely uses MLflow, which enables the management of the ML life cycle, from iteration on model development up to deployment in a reliable and scalable environment.
https://www.mlflow.org/docs/latest/index.html

## Airflow
Apache Airflow is used to orchestrate the ML pipelines.   
https://airflow.apache.org/docs/apache-airflow/stable/

## How to run the platform
To start the automated process:
```
make
```

