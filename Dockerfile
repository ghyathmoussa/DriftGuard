FROM python:3.10-slim

COPY . /app
WORKDIR /app

RUN apt-get update
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app"
