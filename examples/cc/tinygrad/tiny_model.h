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

#ifndef TINYGRAD_TINY_MODEL_H_
#define TINYGRAD_TINY_MODEL_H_

#include <string>
#include <vector>

#include "CL/opencl.hpp"
#include "nlohmann/json.hpp"

namespace tinygrad {

class TinyModel {
 public:
  TinyModel(const std::string& weights_file, const std::string& arch_file, cl::Context *context, cl::CommandQueue *command_queue);

  ~TinyModel() = default;

  void SetInputs(const std::vector<void*>& inputs);
  
  void Inference();

  void GetOutputs(std::vector<void*>& outputs);
  
  const std::string& backend() { return backend_; }

  const std::vector<nlohmann::json>& inputs() { return inputs_; }

  const std::size_t NumInputs() { return inputs_.size(); }

  const std::size_t NumOutputs() { return outputs_.size(); }

 private:
  std::vector<nlohmann::json> statements_;
  std::string backend_;
  std::vector<nlohmann::json> inputs_;
  std::vector<nlohmann::json> outputs_;
  cl::Device device_;
  std::shared_ptr<cl::Context> context_ = nullptr;
  std::shared_ptr<cl::CommandQueue> command_queue_ = nullptr;
  std::unique_ptr<cl::Program> program_ = nullptr;
  std::unordered_map<std::string, cl::Buffer> buffers_;
  std::vector<void*> output_buffers_;
};

}  // namespace tinygrad

#endif  // TINYGRAD_TINY_MODEL_H_
