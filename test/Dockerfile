FROM ubuntu:20.04 AS base

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip git wget

WORKDIR /tinygrad

COPY accel /tinygrad/accel
COPY tinygrad /tinygrad/tinygrad
COPY README.md /tinygrad/
COPY setup.py /tinygrad/

RUN pip3 install --no-cache-dir /tinygrad

WORKDIR /src

CMD ["/usr/bin/env", "python3"]

FROM base AS testing

RUN pip3 install --no-cache-dir "tinygrad[testing]"
RUN wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O /usr/local/lib/python3.8/dist-packages/google/protobuf/internal/builder.py

WORKDIR /testing

COPY test ./test/
COPY extra ./extra/
COPY models ./models/
COPY datasets ./datasets/

ENV PYTHONPATH /testing
