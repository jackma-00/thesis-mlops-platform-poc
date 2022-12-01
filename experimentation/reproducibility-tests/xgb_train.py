import pandas as pd
import mlflow
import xgboost as xgb
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_test_split_pandas(pandas_df,t_size=0.33,r_tate=42):
    X=pandas_df.iloc[:,:-1]
    Y=pandas_df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    THRESHOLD = 0.5

    mlflow.set_experiment("Experimentation")
    mlflow.xgboost.autolog()
    with mlflow.start_run(run_name="train_model", nested=True) as run:
        mlflow.set_tag("mlflow.runName", "train_model")

        pandas_df = pd.read_csv("../data-layer/data/training/data.csv")

        X_train, X_test, y_train, y_test = train_test_split_pandas(pandas_df)

        train_data = xgb.DMatrix(X_train, label=y_train)
        test_data =  xgb.DMatrix(X_test)

        model = xgb.train(dtrain=train_data,params={})
        
        y_probas = model.predict(test_data) 
        y_preds = [1 if  y_proba > THRESHOLD else 0 for y_proba in y_probas]

        # Evaluation  
        accuracy = accuracy_score(y_test, y_preds)
        precision = precision_score(y_test, y_preds)
        recall = recall_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds)
    
        print('Final Testing RESULTS')
        print('/-------------------------------------------------------------------------------------------------------- /')
        print('Accuracy is ', accuracy)
        print('Precision is ', precision)
        print('Recall is ', recall)
        print('F1-Score is ', f1)
        print('/-------------------------------------------------------------------------------------------------------- /')

        # Saving parameters, metrics, and model
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)
        