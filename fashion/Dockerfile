FROM pytorch/pytorch:latest

WORKDIR /app

RUN pip install torchvision

COPY . /app

CMD ["python", "train.py"]
