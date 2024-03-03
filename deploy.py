import mlflow.sagemaker
from mlflow.deployments import get_deploy_client

endpoint_name = "prod-endpoint"
model_uri = "s3://mlflow-project-artifacts-housing-price/3/664894451ab84cafa66b7a6b0032ba22/artifacts/XGBRegressor"

config = {
    "execution_role_arn": "arn:aws:iam::785685275217:role/house-price-role",
    "bucket_name": "mlflow-project-artifacts-housing-price",
    "image_url": "785685275217.dkr.ecr.ap-northeast-2.amazonaws.com/xgb:2.11.0",
    "region_name": "ap-northeast-2",
    "archive": False,
    "instance_type":"ml.m5.xlarge",
    "instance_count": 1,
    "synchronous": True,
    }

# initialize a deployment client for sagemaker
client = get_deploy_client("sagemaker")

# create deployment

client.create_deployment(
    name=endpoint_name,
    model_uri=model_uri,
    flavor="python_function",
    config=config
)
