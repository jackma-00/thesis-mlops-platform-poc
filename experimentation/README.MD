# Experimentation Workflow
Provides the ability to rapidly test multiple approaches and techniques, thus allows for rapid onboarding and development of models
and analytics.

This folder provides an example of how to incorporate popular machine learning tools such as MLflow, and Hydra in your machine learning project. I make predictions on the Iris dataset as an example. 


# Hydra
With [Hydra](https://hydra.cc/), you can compose your configuration dynamically. To install Hydra, simply run
```bash
pip install hydra-core --upgrade
```
# MLflow
[MLflow](https://mlflow.org/) is a platform to manage the ML lifecycle, including experimentation, reproducibility, and deployment. Install MLflow with 
```bash
pip install mlflow
```

# Structure's explanation
* **src**: files for source code
* **data**: to keep datasets
* **mlruns**: files for mlflow runs
* **configs**: to keep config files
* **outputs and multirun**: results from the runs of Hydra. Each time you run your function nested inside Hydra's decoration, the output will be saved here. If you want to change the directory in mlflow folder, use
```python
import mlflow
import hydra
from hydra import utils
mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
```
* `src/train.py`: file for training and saving model

# How to run this file
To run the configs and see how these experiments are displayed on MLflow's server, clone this repo and run
```
python src/train.py
```
Once the run is completed, you can access to MLflow's server with
```
mlflow ui
```
Access http://localhost:5000/ from the same directory that you run the file, you should be able to see your experiment

# Hydra Mutli-run
If you want to run the same application with multiple different configurations.

E.g. you want to run src/train.py testing different models to see which one performs best.

```
python src/train.py -m models=lr,svc
```