FROM ubuntu:20.04
# Install python3.8, and pip3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
# Create a directory called build to store all src files
WORKDIR /build
# Minimal build of tinygrad, only bring is neccesary files to install
COPY tinygrad /build/tinygrad
COPY setup.py /build
COPY README.md /build
# Pip3 install from local source(https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-a-local-src-tree)
RUN pip3 install /build
# Set back workdir to root directory
WORKDIR /