#!/usr/bin/env bash

# TODO: generate (and/or run?) this logic with python compile script
# prereq: install emscripten: https://emscripten.org/docs/getting_started/downloads.html
source ~/emsdk/emsdk_env.sh
step="transformer"
# TODO: auto generate initial memories
initial_memory=71499776
# TODO: tune max memories
max_memory=2500001792
exported_functions='["_net", "_malloc", "_free", "_set_buf"]'

emcc "${step}.c" \
  -O3 -msimd128 -ffast-math -flto \
  -o "${step}.js" \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s EXPORTED_FUNCTIONS="${exported_functions}" \
  -s ENVIRONMENT='worker' \
  -s FILESYSTEM=0 \
  -s EVAL_CTORS \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY="$initial_memory" \
  -s MAXIMUM_MEMORY="$max_memory"