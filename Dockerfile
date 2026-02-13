FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install dependencies
COPY requirements.txt /job/
RUN pip install -r requirements.txt