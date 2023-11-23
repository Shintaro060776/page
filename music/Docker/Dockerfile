FROM python:3.8

WORKDIR /workspace

RUN pip install --no-cache-dir torch torchvision torchaudio numpy pandas scikit-learn

COPY . /workspace

CMD ["python", "train.py"]