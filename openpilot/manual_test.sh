#!/bin/bash -e
OPENCL=1 DEBUGCL=1 python3 openpilot/compile.py ../selfdrive/modeld/models/supercombo.onnx ../selfdrive/modeld/models/supercombo.thneed
