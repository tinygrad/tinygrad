#!/bin/bash
cd "/Users/qazal/Library/Application Support/Instruments/Templates"
rm -rf "MTLCounter.tracetemplate"
plutil -convert binary1 -o "MTLCounter.tracetemplate" "MTLCounter.xml"

cd "/Users/qazal/code/tinygrad"
LAUNCH=0 METAL=1 PRINT_MATCH_STATS=0 VIZ=1 python ./extra/gpu_counter_example.py

xctrace export --input /tmp/metal.trace/ --toc
xctrace export --input /tmp/metal.trace/ --xpath '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-info"]'
