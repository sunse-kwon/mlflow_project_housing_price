import mlflow

experiment_name = "XGBRegressor"
entry_point = "Training"

mlflow.set_tracking_uri("http://ec2-13-209-26-166.ap-northeast-2.compute.amazonaws.com:5000/")
# import pdb; pdb.set_trace()
mlflow.projects.run(

    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name,
    env_manager="conda"
)
