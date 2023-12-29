FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_train.txt /pipeline
RUN pip install -r requirements_train.txt
COPY train.py /pipeline