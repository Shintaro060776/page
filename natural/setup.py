import subprocess
import sys

timeout_value = 10800

subprocess.run(
    f'pip install --default-timeout={timeout_value} -r requirements.txt'.split())

subprocess.run(['python', 'train_model.py'] + sys.argv[1:])
