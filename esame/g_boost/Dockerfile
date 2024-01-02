FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_gb.txt /pipeline
RUN pip install -r requirements_gb.txt
COPY g_boost.py /pipeline