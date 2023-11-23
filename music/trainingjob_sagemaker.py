import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

source_dir = 'my_training_scripts'
entry_point = 'train_model.py'
training_data_uri = 's3://vhrthrtyergtcere/preprocessed_lyrics.txt'

pytorch_estimator = PyTorch(
    entry_point=entry_point,
    source_dir=source_dir,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='1.8.0',
    py_version='py3',
    hyperparameters={
        'epochs': 20, 
        'batch_size': 32,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'seq_length': 10, 
        'lr': 0.0005
    }
)

pytorch_estimator.fit({'train': training_data_uri})