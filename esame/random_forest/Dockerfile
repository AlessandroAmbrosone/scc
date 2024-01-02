FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_rf.txt /pipeline
RUN pip install -r requirements_rf.txt
COPY random_forest.py /pipeline