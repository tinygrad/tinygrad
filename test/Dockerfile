FROM ubuntu:20.04

# Install python3.8, and pip3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update 
RUN apt-get install -y python3-pip git

# Install python dependencies
COPY . ./tinygrad
WORKDIR tinygrad
RUN pip install -e ".[testing]"
