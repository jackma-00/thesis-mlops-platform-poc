# Thesis MLOps Platform PoC
The platform contains the following components as described by the architecture diagram.
![image](https://user-images.githubusercontent.com/91907141/200011595-a9502ccb-487a-41c8-a747-9babd67bc7c1.png)

## MLOps level 0: manual process, plus model monitoring
![image](https://user-images.githubusercontent.com/91907141/200011708-b7f5297b-9ecf-40e3-a3ed-4c196e47360b.png)
- Collection of objective and timely metrics to determine the quality of models in production.
- In the event that problems are identified, the (manual) process of data analysis and train of a new model must be restarted.
- Monitoring can identify several opportunities for improvement.

## MLflow
This project largely uses MLflow, which enables the management of the ML life cycle, from iteration on model development up to deployment in a reliable and scalable environment.
https://www.mlflow.org/docs/latest/index.html

## How to run the platform
Step into the server directory:
```
cd tracking-server
```
Start the environment:
```
make
```
Now you can follow the ML end-to-end pipeline stepping in the corresponding directories and utilizing the make files.
