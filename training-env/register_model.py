import mlflow
import sys

if __name__ == "__main__":
    
    with mlflow.start_run(run_name="register_model", nested=True) as run:

        mlflow.set_tag("mlflow.runName", "register_model")

        model_uri = str(sys.argv[1])
        model_name = str(sys.argv[2])

        print("\nModel URI: " + model_uri)
        print("\nModel name: " + model_name + "\n")

        result = mlflow.register_model(model_uri, model_name)
        