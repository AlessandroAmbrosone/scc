FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_dd.txt /pipeline
RUN pip install -r requirements_dd.txt
COPY download_data.py /pipeline