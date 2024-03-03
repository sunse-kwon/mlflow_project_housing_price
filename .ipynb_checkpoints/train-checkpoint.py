import mlflow
import numpy as np
from data import X_train, X_val, y_train, y_val
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from params import ridge_param_grid, elasticnet_param_grid, xgb_param_grid
from utils import eval_metrics

# loop through the hypterparameter combinations and log results in seperate runs
for params in ParameterGrid(xgb_param_grid):
    with mlflow.start_run():
        lr = XGBRegressor(**params)

        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        metrics = eval_metrics(y_val, y_pred)

        # logging the inputs such as dataset
        mlflow.log_input(
            mlflow.data.from_numpy(X_train.toarray()),
            context='training dataset'
        )
        mlflow.log_input(
            mlflow.data.from_numpy(X_val.toarray()),
            context='validation dataset'
        )

        # logging hyperparameters
        mlflow.log_params(params)

        # logging metrics
        mlflow.log_metrics(metrics)

        # log the trained model
        mlflow.sklearn.log_model(
            lr,
            'XGBRegressor',
            input_example=X_train,
            code_paths=['train.py', 'data.py', 'params.py', 'utils.py']
        )
