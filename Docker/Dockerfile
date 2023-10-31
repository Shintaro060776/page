# docker run -it --rm -v /page/wav:/data test
# docker run -it --rm -v /page/Docker:/app -v /page/wav:/data test /bin/bash

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/r9y9/wavenet_vocoder.git /app/wavenet_vocoder
ENV PYTHONPATH "${PYTHONPATH}:/app/wavenet_vocoder"

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN pip install inflect
RUN pip install tensorflow==2.8.0
RUN pip install protobuf==3.20.0
RUN pip install docopt
RUN pip install nnmnkwii
RUN pip install cython numpy scipy
RUN pip install tensorboardX

RUN pip install pysptk

COPY train.py /app/train.py

ENTRYPOINT ["python", "/app/train.py"]

RUN mkdir /data