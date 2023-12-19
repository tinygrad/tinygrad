/* MIT License Copyright (c) https://tinygrad.org/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "tinygrad/tiny_model.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <weights (path to .safetensors file)> <arch (path to .json "
                 "file)> <imagepath (path to image)>"
              << std::endl;
    return 1;
  }

  std::string weights = argv[1];
  std::string arch = argv[2];
  std::string imagepath = argv[3];

  tinygrad::TinyModel tiny_model(weights, arch, nullptr, nullptr);

  float input0[602112];
  float output0[4000];
  int X = 0, Y = 0, chan = 0;

  stbi_uc *image = stbi_load(imagepath.c_str(), &X, &Y, &chan, 3);

  assert(image != NULL);

  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      // get sample position
      int tx = (x / 224.) * X;
      int ty = (y / 224.) * Y;
      for (int c = 0; c < 3; c++) {
        input0[c * 224 * 224 + y * 224 + x] =
            (image[ty * X * chan + tx * chan + c] / 255.0 - 0.45) / 0.225;
      }
    }
  }

  // Set inputs
  std::vector<void *> inputs;
  inputs.push_back(input0);
  tiny_model.SetInputs(inputs);

  // Inferense
  tiny_model.Inference();

  // Get outputs
  std::vector<void *> outputs;
  outputs.push_back(output0);
  tiny_model.GetOutputs(outputs);

  float best = -INFINITY;
  int best_idx = -1;
  for (int i = 0; i < 1000; i++) {
    if (output0[i] > best) {
      best = output0[i];
      best_idx = i;
    }
  }

  std::cout << "Best idx: " << best_idx << std::endl;

  return 0;
}
