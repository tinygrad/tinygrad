#!/bin/bash
# brew install cmake ninja
# NOTE: requires cuda toolkit, so no OS X (why?)
git clone --recurse-submodules https://github.com/gpuocelot/gpuocelot.git
cd gpuocelot/ocelot
git checkout 18401f4245b27ca4b3af433196583cc81ef84480
mkdir build
cd build
cmake .. -Wno-dev -G Ninja -DOCELOT_BUILD_TOOLS=OFF -DCMAKE_BUILD_ALWAYS=0
ninja
