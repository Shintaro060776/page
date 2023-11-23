FROM python:3.8-slim

RUN pip install boto3 torch numpy

COPY inference.py /opt/ml/model/code/

ENTRYPOINT ["python", "/opt/ml/model/code/inference.py"]
