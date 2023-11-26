FROM pytorch/torchserve:latest

RUN pip install torchvision boto3 pillow

COPY model.pth /home/model-server/
COPY config.properties /home/model-server/

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--ncs", "--model-store", "/home/model-server/model-store", "--models", "my_model=model.pth"]