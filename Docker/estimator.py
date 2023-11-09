import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

estimator = PyTorch(entry_point='train_model.py',
                    role='arn:aws:iam::715573459931:role/test1111',
                    framework_version='1.5.0',
                    py_version='py3',
                    train_instance_count=1,
                    train_instance_type='ml.m5.large',
                    hyperparameters={
                        'epochs': 10,
                        'batch-size': 64,
                        'learning-rate': 0.01
                    })

train_data = 's3://vhrthrtyergtcere/'

estimator.fit({'training': train_data})
