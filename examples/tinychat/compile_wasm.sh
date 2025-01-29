#!/usr/bin/env bash

# TODO: generate (and/or run?) this logic with python compile script
# point below path at your emscripten installation location
source ~/emsdk/emsdk_env.sh
which emcc
inputs=("transformer" "q6k_to_f32" "q6k_to_int8_2048_2048" "q6k_to_int8_512_2048" "q6k_to_int8_8192_2048" "q6k_to_int8_2048_8192")
# TODO: auto generate initial memories
initial_memories=(4456448 196608 655360 262144 2228224 2228224)
# TODO: tune max memories
maximum_memories=(2500001792 65536000 65536000 65536000 65536000 65536000)
for i in "${!inputs[@]}"; do
  input="${inputs[i]}"
  initial_memory="${initial_memories[i]}"
  maximum_memory="${maximum_memories[i]}"

  if [[ "$input" == "transformer" ]]; then
    exported_functions='["_net", "_malloc", "_free", "_set_buf"]'
  else
    exported_functions='["_net", "_malloc", "_free"]'
  fi

  echo "Processing $input with INITIAL_MEMORY=$initial_memory and MAXIMUM_MEMORY=$maximum_memory"
  echo $exported_functions

  emcc "${input}.c" \
    -O3 -msimd128 -ffast-math -flto \
    -o "${input}.js" \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORTED_FUNCTIONS="${exported_functions}" \
    -s ENVIRONMENT='worker' \
    -s FILESYSTEM=0 \
    -s EVAL_CTORS \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY="$initial_memory" \
    -s MAXIMUM_MEMORY="$maximum_memory"
done