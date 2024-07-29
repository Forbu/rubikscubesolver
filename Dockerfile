FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt