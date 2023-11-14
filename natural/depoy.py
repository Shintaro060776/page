import sagemaker
from sagemaker.pytorch import PyTorchModel

model_data = 's3://sagemaker-ap-northeast-1-715573459931/pytorch-training-2023-11-13-14-56-47-044/output/model.tar.gz'

pytorch_model = PyTorchModel(model_data=model_data,
                             role='arn:aws:iam::715573459931:role/test1111',
                             framework_version='1.8.0',
                             py_version='py3',
                             entry_point='inference.py',
                             source_dir='my_training_scripts')

predictor = pytorch_model.deploy(
    initial_instance_count=1, instance_type='ml.m5.large')
