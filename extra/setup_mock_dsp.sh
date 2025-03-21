#!/bin/bash -e

cd ./extra/dsp
docker build . -t mockdsp --platform=linux/amd64
