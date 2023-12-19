# tinygrad c++ inference

## Setup
```
# Download stb_image and nlohmann::json libs
python3 setup_env.py
```

## Compilation
```
clang++ -std=c++17 -o tinygrad_exec main.cc tinygrad/*.cc -lOpenCL -Ithird_party/ -I.
```

## Compile weights
```
GPU=1 python3 examples/compile_efficientnet.py 
```

## Usage
```
tinygrad_exec path_to_weights/net.safetensors path_to_arch/net.json path_to_image/image.jpg
```

## TODO
[ ] Add other backends
[ ] Check on other architectures
