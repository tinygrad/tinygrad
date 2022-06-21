#!/bin/bash -e
./compile.py
THNEED_DEBUG=1 ./run_thneed /tmp/output.thneed
