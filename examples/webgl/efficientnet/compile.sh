#!/bin/bash
WEBGL=1 python3 ../../compile_efficientnet.py
mv ../../net.js .
mv ../../net.safetensors .