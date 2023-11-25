import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

s3_input_data = 's3://vhrthrtyergtcere/images/'

s3_output_location = 's3://vhrthrtyergtcere/images/output/'

estimator = PyTorch(entry_point='train_model.py',
                    py_version='py3',
                    role=role,
                    framework_version='1.8.0',
                    train_instance_count=1,
                    train_instance_type='ml.m5.large',
                    output_path=s3_output_location,
                    hyperparameters={
                        'batch_size': 64,
                        'learning_rate': 0.0002,
                        'epochs': 50,
                        'nz': 100,
                        'data_dir': s3_input_data
                    })

estimator.fit({'training': s3_input_data})
