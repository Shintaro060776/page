FROM tensorflow/tensorflow:latest

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install numpy \
    && pip install pandas \
    && pip install nltk

RUN python -m nltk.downloader stopwords

COPY . /app

CMD ["python", "train_model.py"]