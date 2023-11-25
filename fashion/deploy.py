import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()
role = get_execution_role()

model_artifact = 's3://vhrthrtyergtcere/images/output/pytorch-training-2023-11-24-19-07-41-689/output/model.tar.gz'

pytorch_model = PyTorchModel(model_data=model_artifact,
                             role=role,
                             py_version='py3',
                             framework_version='1.8.0',
                             entry_point='inference.py')

predictor = pytorch_model.deploy(initial_instance_count=1,
                                 instance_type='ml.m5.large')

endpoint_name = predictor.endpoint_name
print("Created endpoint: {}".format(endpoint_name))
