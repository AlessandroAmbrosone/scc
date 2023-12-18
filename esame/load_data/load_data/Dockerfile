FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_ld.txt /pipeline
RUN pip install -r requirements_ld.txt
COPY load_data.py /pipeline
COPY test.csv /pipeline
COPY train.csv /pipeline