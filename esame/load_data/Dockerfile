FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_cf.txt /pipeline
RUN pip install -r requirements_cf.txt
COPY create_f_data.py /pipeline
