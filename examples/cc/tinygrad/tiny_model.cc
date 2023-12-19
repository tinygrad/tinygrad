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

#include <fstream>
#include <iostream>

#include "tinygrad/util.h"

namespace tinygrad {

TinyModel::TinyModel(const std::string &weights_file,
                     const std::string &arch_file, cl::Context *context,
                     cl::CommandQueue *command_queue) {
  std::cout << "OpenCL initialization...\n";

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  auto platform = platforms.front();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  device_ = devices.front();

  if (context == nullptr) {
    context_ = std::make_shared<cl::Context>(device_);
  } else {
    context_.reset(context);
  }

  if (command_queue_ == nullptr) {
    command_queue_ =
        std::make_shared<cl::CommandQueue>(*context_.get(), device_);
  } else {
    command_queue_.reset(command_queue);
  }

  std::cout << "Load model Arch...\n";
  std::ifstream file(arch_file);
  nlohmann::json arch;
  file >> arch;

  std::cout << "Load model weights...\n";
  auto weights = SafeLoad(weights_file);

  statements_ = arch["statements"].get<std::vector<nlohmann::json>>();
  backend_ = arch["backend"].get<std::string>();
  inputs_ = arch["inputs"].get<std::vector<nlohmann::json>>();
  outputs_ = arch["outputs"].get<std::vector<nlohmann::json>>();

  auto &buffers = arch["buffers"];

  for (auto &[key, value] : buffers.items()) {
    std::string buffer_id = buffers[key]["id"].get<std::string>();
    int buffer_size = buffers[key]["size"].get<int>();

    if (buffer_id.empty()) {
      cl::Buffer cl_buffer(*context_.get(), CL_MEM_READ_WRITE, buffer_size);
      buffers_[key] = cl_buffer;
    } else {
      cl::Buffer cl_buffer(*context_.get(), CL_MEM_READ_ONLY, buffer_size);
      command_queue_->enqueueWriteBuffer(cl_buffer, CL_TRUE, 0, buffer_size,
                                         weights[buffer_id].data());
      buffers_[key] = cl_buffer;
    }
  }

  std::cout << "Building opencl program...\n";
  auto functions = arch["functions"];

  std::string kernel_source;

  for (const auto &func : functions) {
    kernel_source += func.get<std::string>() + '\n';
  }

  cl::Program::Sources sources;
  sources.push_back({kernel_source.c_str(), kernel_source.length()});
  program_ = std::make_unique<cl::Program>(*context_.get(), sources);

  auto err = program_->build(device_);
  if (err != CL_SUCCESS) {
    std::string build_log =
        program_->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);
    std::cerr << "Build Error: " << build_log << std::endl;
  }
}

void TinyModel::SetInputs(const std::vector<void *> &inputs) {
  assert((inputs.size() == inputs_.size()) &&
         "inputs.size() != inputs_.size()");

  auto size = inputs.size();

  for (int i = 0; i < size; ++i) {
    std::string input_name = "input" + std::to_string(i);
    int buffer_size = inputs_[i]["size"].get<int>();

    auto &cl_input_buffer = buffers_[input_name];
    command_queue_->enqueueWriteBuffer(cl_input_buffer, CL_TRUE, 0, buffer_size,
                                       inputs[i]);
  }
}

void TinyModel::Inference() {
  for (const auto &statement : statements_) {
    auto args = statement["args"].get<std::vector<std::string>>();
    auto kernel_name = statement["kernel"].get<std::string>();
    auto global_size = statement["global_size"].get<std::vector<int>>();
    auto local_size = statement["local_size"].get<std::vector<int>>();

    cl::Kernel kernel(*program_.get(), kernel_name.c_str());
    for (int i = 0; i < args.size(); ++i) {
      auto &buffer = buffers_[args[i]];
      kernel.setArg(i, buffer);
    }

    cl::NDRange global_size_cl(global_size[0] * local_size[0],
                               global_size[1] * local_size[1],
                               global_size[2] * local_size[2]);

    cl::NDRange local_size_cl(local_size[0], local_size[1], local_size[2]);

    command_queue_->enqueueNDRangeKernel(kernel, cl::NullRange, global_size_cl,
                                         local_size_cl);
  }
}

void TinyModel::GetOutputs(std::vector<void *> &outputs) {
  assert((outputs.size() == outputs_.size()) &&
         "outputs.size() != outputs_.size()");

  auto size = outputs.size();

  for (int i = 0; i < size; ++i) {
    std::string output_name = "output" + std::to_string(i);
    int buffer_size = outputs_[i]["size"].get<int>();

    auto &cl_output_buffer = buffers_[output_name];
    command_queue_->enqueueReadBuffer(cl_output_buffer, CL_TRUE, 0, buffer_size,
                                      outputs[i]);
  }
}

} // namespace tinygrad
