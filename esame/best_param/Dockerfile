FROM python:3.8-slim
WORKDIR /pipeline
COPY requirements_bp.txt /pipeline
RUN pip install -r requirements_bp.txt
COPY best_param.py /pipeline