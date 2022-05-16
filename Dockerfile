FROM debian:10


COPY requirements.txt /tmp/requirements.txt


RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && \
    apt-get install -y build-essential \
                       python3 \
                       python3-pip \
                       zlib1g-dev \
                       libjpeg-dev \
                       libpng-dev && \
    pip3 install -r /tmp/requirements.txt
