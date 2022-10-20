#!/bin/bash -e
#OPENCL=1 DEBUGCL=1 python3 openpilot/compile.py
OPENCL=1 DEBUGCL=1 python3 openpilot/compile.py ../selfdrive/modeld/models/supercombo.onnx ../selfdrive/modeld/models/supercombo.thneed
#FLOAT32=1 python3 openpilot/run_thneed.py $PWD/../selfdrive/modeld/models/supercombo.thneed $PWD/../selfdrive/modeld/models/supercombo.onnx
