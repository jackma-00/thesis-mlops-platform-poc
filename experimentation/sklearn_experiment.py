import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Print configurations
    #print(OmegaConf.to_yaml(cfg))

    # Load the csv dataset 
    df = pd.read_csv(cfg.datasets.path)
    print(f'\nUsing dataset: {cfg.datasets.dataset}\n')
    print(df.head())
    print(df.tail())

    # Independent and Dependent features
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]

    # Train test split 
    X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=cfg.train_test_split.test_size,
                                                   random_state=cfg.train_test_split.random_state)

    mlflow.set_experiment("Experimentation")
    with mlflow.start_run(run_name=cfg.models.model+"-experimental-run"):
        mlflow.set_tag("mlflow.runName", cfg.models.model+"-experimental-run")

        # Initializing classifier 
        clf = eval(cfg.models.model)()
        print(f'\nInstantiated model: {cfg.models.model}\n')

        # Tuning parameters via grid search
        print(f'Tuning hyper-parameters for {cfg.models.model} model ...\n')
        gs = GridSearchCV(estimator=clf,
                          param_grid=dict(cfg.models.hyperparameters),
                          scoring=cfg.GridSearchCV.scoring,
                          cv=cfg.GridSearchCV.cv)
        gs.fit(X_train, y_train)
        print(f"Best choice for {cfg.models.model}'s parameters is: {gs.best_params_}\n"
              f"Since it leads to the highest accuracy of: {gs.best_score_}\n")

        # Prediction 
        y_pred = gs.predict(X_test)

        # Evaluation  
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    
        print('Final Testing RESULTS')
        print('/-------------------------------------------------------------------------------------------------------- /')
        print('Accuracy is ', accuracy)
        print('Precision is ', precision)
        print('Recall is ', recall)
        print('F1-Score is ', f1)
        print('/-------------------------------------------------------------------------------------------------------- /')

        # Saving parameters, metrics, and model
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("gs_accuracy", gs.best_score_)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)
        mlflow.sklearn.log_model(gs, "model")

if __name__== "__main__":
	main()