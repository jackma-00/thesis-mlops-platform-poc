import warnings

import pandas as pd
import mlflow
import mlflow.pyfunc

MODEL_NAME = "xgb-psystock"

def score():

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Get production input data to be scored
    data = pd.read_csv("../data-layer/data/scoring/input.csv", header=None,
                                names=["{}".format(i) for i in range(0, 14)])
    print(data)
    #data.to_json(path_or_buf="df/df.json", orient='split')

    with mlflow.start_run(run_name="batch_scoring") as run:
        mlflow.set_tag("mlflow.runName", "batch_scoring")

        model_name = MODEL_NAME
        model_version = 1

        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

        y_probas = model.predict(data)

        y_preds = [1 if  y_proba > 0.5 else 0 for y_proba in y_probas]
        
        data[len(data.columns)] = y_preds
        
        result = data

        result.to_csv("../data-layer/data/scoring/output.csv", index=False)


if __name__ == "__main__":
    score()

        

