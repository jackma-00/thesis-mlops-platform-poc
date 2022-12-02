import warnings
from flask import Flask, request
import pandas as pd
from flasgger import Swagger
import mlflow
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient

MODEL_NAME = "xgb-psystock"
MODEL_VERSION = 2
MODEL_STAGE = "Production"

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

app = Flask(__name__)
Swagger(app)

client = MlflowClient()

@app.route('/')
def welcome():
    return "Welcome To The Inference API\nSubmit Here The Input Data"

@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    with mlflow.start_run(run_name="API_inference") as run:
        mlflow.set_tag("mlflow.runName", "API_inference")

        model_name = MODEL_NAME
        #model_version = MODEL_VERSION
        model_stage = MODEL_STAGE

        # Retrieve the model

        # Check the latest model version
        #model_version_infos = client.search_model_versions("name = '%s'" % model_name)
        #new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
        #print(new_model_version)

        # Get the latest model version in production 
        model_version = client.get_latest_versions(model_name, stages=[model_stage])[0].version
        #print(model_version)
        model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name, model_stage=model_stage)
        # Load the production model 
        model = mlflow.pyfunc.load_model(model_uri=model_uri) 

        # Log model's params 
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("model_stage", model_stage)
    
        # Retrieve the input data
        df_test = pd.read_csv(request.files.get("file"), header=None, names=["{}".format(i) for i in range(0, 14)])
        print(df_test.head())

        # Save data from production to monitor
        df_test.to_csv("../data-layer/data/scoring/input.csv", index=False, header=False)

        # Log data from production 
        mlflow.log_artifact("../data-layer/data/scoring/input.csv")
    
        # Predict on production data
        y_probas = model.predict(df_test)
        y_preds = [1 if  y_proba > 0.5 else 0 for y_proba in y_probas]

    return str(list(y_preds))


if __name__ == '__main__':

        # Launch the app
        app.run(debug=True, port=7777)
    