FROM ubuntu:20.04
# Install git, python3.8, and pip3
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.8 \
    python3-pip
RUN pip3 install git+https://github.com/geohot/tinygrad.git