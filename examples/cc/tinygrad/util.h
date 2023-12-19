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

#ifndef TINYGRAD_UTIL_H_
#define TINYGRAD_UTIL_H_

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "tinygrad/common.h"

namespace tinygrad {

TINYGRAD_INLINE auto SafeLoad(const std::string &filename)
    -> std::unordered_map<std::string, std::vector<float>> {
  std::unordered_map<std::string, std::vector<float>> weights_map;

  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cout << "Unable to open file: " << filename << std::endl;
    return weights_map;
  }

  // Read JSON length
  int64_t json_len;
  file.read(reinterpret_cast<char *>(&json_len), sizeof(json_len));

  // Read JSON
  std::vector<char> json_buffer(json_len);
  file.read(json_buffer.data(), json_len);
  auto json = nlohmann::json::parse(json_buffer.begin(), json_buffer.end());

  for (auto &element : json.items()) {
    if (element.key() == "__metadata__")
      continue;
    // TODO: Unused
    auto dtype_str = element.value()["dtype"].get<std::string>();
    std::vector<int> data_offsets =
        element.value()["data_offsets"].get<std::vector<int>>();
    size_t sz = static_cast<size_t>((data_offsets[1] - data_offsets[0]) /
                                    sizeof(float));

    file.seekg(8 + json_len + data_offsets[0], std::ios::beg);
    std::vector<float> weight(sz);
    file.read(reinterpret_cast<char *>(weight.data()), sz * sizeof(float));

    weights_map[element.key()] = std::move(weight);
  }

  return weights_map;
}

} // namespace tinygrad

#endif // TINYGRAD_UTIL_H_
