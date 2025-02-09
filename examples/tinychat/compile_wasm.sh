#!/usr/bin/env bash

# TODO: generate (and/or run?) this logic with python compile script
# point below path at your emscripten installation location
source ~/emsdk/emsdk_env.sh
which emcc
inputs=("module0" "module1" "module2" "module3" "module4" "module5" "module6" "module7" "module8" "module9" "module10" "module11" "module12" "module13" "module14" "module15" "module16")
# TODO: auto generate initial memories
#initial_memories=(3735552 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056 4653056)
initial_memories=(336330752 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968 74579968)

# TODO: tune max memories
maximum_memories=(416415744 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224 92340224)
for i in "${!inputs[@]}"; do
  input="${inputs[i]}"
  initial_memory="${initial_memories[i]}"
  maximum_memory="${maximum_memories[i]}"

  if [[ "$input" == "module0" ]]; then
    exported_functions='["_net0", "_net1", "_malloc", "_free", "_set_buf"]'
  else
    exported_functions='["_net", "_malloc", "_free", "_set_buf"]'
  fi

  echo "Processing $input with INITIAL_MEMORY=$initial_memory and MAXIMUM_MEMORY=$maximum_memory"
  echo $exported_functions

    #-s ALLOW_MEMORY_GROWTH=1 \
    #-s MAXIMUM_MEMORY="$maximum_memory"
  emcc "${input}.c" kernels.c \
    -O3 -msimd128 -ffast-math -flto \
    -o "${input}.js" \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s EXPORTED_FUNCTIONS="${exported_functions}" \
    -s ENVIRONMENT='worker' \
    -s FILESYSTEM=0 \
    -s EVAL_CTORS \
    -s INITIAL_MEMORY="$initial_memory"
done