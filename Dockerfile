FROM python:3.10.2-buster

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD . /app


