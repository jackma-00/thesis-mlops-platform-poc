export MLFLOW_TRACKING_URI=http://localhost:5000

default: run

run:
	mlflow run --experiment-name Deployment_and_Inference_Environment .

serve:
	mlflow models serve -m "models:/xgb-psystock/2" -p 6000

test:
	curl localhost:6000/invocations \
	-H 'Content-Type:application/json; format=pandas-split' \
	--data '{"columns":["0","1","2","3","4","5","6","7","8","9","10","11","12","13"],"index":[0],"data":[[0,1,0,1,1,0,1,1,1,1,0,0,0,0]]}'