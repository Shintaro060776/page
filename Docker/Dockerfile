FROM python:3.8-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY foldername.csv /usr/src/app/
COPY rendered_256x256/ /usr/src/app/rendered_256x256/

EXPOSE 80

CMD ["python", "train_model.py"]