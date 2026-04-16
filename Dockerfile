FROM python:3.10-slim
WORKDIR /temperature

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src ./src

