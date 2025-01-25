#!/usr/bin/env bash

# TODO: generate (and/or run?) this logic with python compile script
# point below path at your emscripten installation location
source ~/emsdk/emsdk_env.sh
which emcc
inputs=("transformer" "q6k_to_f16" "q6k_to_int8_2048_2048" "q6k_to_int8_512_2048" "q6k_to_int8_8192_2048" "q6k_to_int8_2048_8192")
# TODO: auto generate initial memories
initial_memories=(1570701312 131072 393216 196608 1310720 1310720)
# TODO: tune max memories
maximum_memories=(2500001792 65536000 65536000 65536000 65536000 65536000)
for i in "${!inputs[@]}"; do
    input="${inputs[i]}"
    initial_memory="${initial_memories[i]}"
    maximum_memory="${maximum_memories[i]}"

    echo "Processing $input with INITIAL_MEMORY=$initial_memory and MAXIMUM_MEMORY=$maximum_memory"

    emcc "${input}.c" \
        -o "${input}.js" \
        -s MODULARIZE=1 \
        -s EXPORT_ES6=1 \
        -s EXPORTED_FUNCTIONS='["_net", "_malloc", "_free"]' \
        -s EXPORTED_RUNTIME_METHODS='["cwrap", "getValue", "setValue"]' \
        -s ENVIRONMENT='web,worker' \
        -s ALLOW_MEMORY_GROWTH=1 \
        -s INITIAL_MEMORY="$initial_memory" \
        -s MAXIMUM_MEMORY="$maximum_memory"
done