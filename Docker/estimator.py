# pip install sagemaker

from tacotron2.layers import ConvNorm, LinearNorm
import sagemaker
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator

role = get_execution_role()
sagemaker_session = sagemaker.Session()

image_config = {
    'RepositoryAccessMode': 'VPC'
}

estimator = Estimator(
    image_uri='715573459931.dkr.ecr.us-east-1.amazonaws.com/test:latest',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://vhrthrtyergtcere/output/',
    sagemaker_session=sagemaker_session,
    training_image_config=image_config
)

estimator.fit('s3://vhrthrtyergtcere/train/')
