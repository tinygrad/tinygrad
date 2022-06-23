#include "thneed.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <map>
#include <string>

#include "common/clutil.h"
#include "common/timing.h"

Thneed *g_thneed = NULL;
int g_fd = -1;
map<pair<cl_kernel, int>, string> g_args;
map<pair<cl_kernel, int>, int> g_args_size;
map<cl_program, string> g_program_source;

void hexdump(uint8_t *d, int len) {
  assert((len%4) == 0);
  printf("  dumping %p len 0x%x\n", d, len);
  for (int i = 0; i < len/4; i++) {
    if (i != 0 && (i%0x10) == 0) printf("\n");
    printf("%8x ", d[i]);
  }
  printf("\n");
}

// *********** Thneed ***********

Thneed::Thneed(bool do_clinit) {
  if (do_clinit) clinit();
#ifdef INTERCEPTOR
  assert(g_fd != -1);
  fd = g_fd;
  ram = make_unique<GPUMalloc>(0x80000, fd);
#endif
  timestamp = -1;
  g_thneed = this;
  char *thneed_debug_env = getenv("THNEED_DEBUG");
  debug = (thneed_debug_env != NULL) ? atoi(thneed_debug_env) : 0;
}

void Thneed::find_inputs_outputs() {
  cl_int err;
  if (inputs.size() > 0) return;

  // save the global inputs/outputs
  for (auto &k : kq) {
    for (int i = 0; i < k->num_args; i++) {
      if (k->name == "zero_pad_image_float" && k->arg_names[i] == "input") {
        cl_mem aa = *(cl_mem*)(k->args[i].data());
        input_clmem.push_back(aa);

        size_t sz;
        clGetMemObjectInfo(aa, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
        input_sizes.push_back(sz);

        void *ret = clEnqueueMapBuffer(command_queue, aa, CL_TRUE, CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        inputs.push_back(ret);
      }

      if (k->name == "image2d_to_buffer_float" && k->arg_names[i] == "output") {
        output = *(cl_mem*)(k->args[i].data());
      }
    }
  }
}

void Thneed::copy_inputs(float **finputs) {
  //cl_int ret;
  for (int idx = 0; idx < inputs.size(); ++idx) {
    if (debug >= 1) printf("copying %lu -- %p -> %p\n", input_sizes[idx], finputs[idx], inputs[idx]);
    if (finputs[idx] != NULL) memcpy(inputs[idx], finputs[idx], input_sizes[idx]);
  }
}

void Thneed::copy_output(float *foutput) {
  if (output != NULL) {
    size_t sz;
    clGetMemObjectInfo(output, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
    if (debug >= 1) printf("copying %lu for output %p -> %p\n", sz, output, foutput);
    clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sz, foutput, 0, NULL, NULL);
  } else {
    printf("CAUTION: model output is NULL, does it have no outputs?\n");
  }
}

void Thneed::clinit() {
  device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
  //cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
#ifdef clCreateCommandQueueWithProperties
  cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, 0, 0};
  command_queue = CL_CHECK_ERR(clCreateCommandQueueWithProperties(context, device_id, props, &err));
#else
  command_queue = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &err));
#endif
  printf("Thneed::clinit done\n");
}

cl_int Thneed::clexec() {
  printf("Thneed::clexec: running %lu queued kernels\n", kq.size());
  for (auto &k : kq) {
    if (record) ckq.push_back(k);
    cl_int ret = k->exec();
    assert(ret == CL_SUCCESS);
  }
  return clFinish(command_queue);
}

// *********** CLQueuedKernel ***********

CLQueuedKernel::CLQueuedKernel(Thneed *lthneed,
                               cl_kernel _kernel,
                               cl_uint _work_dim,
                               const size_t *_global_work_size,
                               const size_t *_local_work_size) {
  thneed = lthneed;
  kernel = _kernel;
  work_dim = _work_dim;
  assert(work_dim <= 3);
  for (int i = 0; i < work_dim; i++) {
    global_work_size[i] = _global_work_size[i];
    local_work_size[i] = _local_work_size[i];
  }

  char _name[0x100];
  clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(_name), _name, NULL);
  name = string(_name);
  clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);

  // get args
  for (int i = 0; i < num_args; i++) {
    char arg_name[0x100];
    clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
    arg_names.push_back(string(arg_name));
    clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_name), arg_name, NULL);
    arg_types.push_back(string(arg_name));

    args.push_back(g_args[make_pair(kernel, i)]);
    args_size.push_back(g_args_size[make_pair(kernel, i)]);
  }

  // get program
  clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, NULL);
}

int CLQueuedKernel::get_arg_num(const char *search_arg_name) {
  for (int i = 0; i < num_args; i++) {
    if (arg_names[i] == search_arg_name) return i;
  }
  printf("failed to find %s in %s\n", search_arg_name, name.c_str());
  assert(false);
}

#ifndef INTERCEPTOR
#define thneed_clSetKernelArg clSetKernelArg
#endif

cl_int CLQueuedKernel::exec() {
  if (kernel == NULL) {
    cl_int err;
    kernel = clCreateKernel(program, name.c_str(), &err);
    assert(err == CL_SUCCESS);
    arg_names.clear();
    arg_types.clear();

    for (int j = 0; j < num_args; j++) {
      char arg_name[0x100];
      clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
      arg_names.push_back(string(arg_name));
      clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_name), arg_name, NULL);
      arg_types.push_back(string(arg_name));

      cl_int ret;
      if (args[j].size() != 0) {
        assert(args[j].size() == args_size[j]);
        ret = thneed_clSetKernelArg(kernel, j, args[j].size(), args[j].data());
      } else {
        ret = thneed_clSetKernelArg(kernel, j, args_size[j], NULL);
      }
      if (ret != CL_SUCCESS) {
        printf("CL error num_args:%d setting arg %d(%ld) arg %s = %s : %d %s\n", num_args, j, args[j].size(),
          arg_types[j].c_str(), arg_names[j].c_str(), ret, cl_get_error_string(ret));
        hexdump((uint8_t *)args[j].data(), args[j].size());
        debug_print(true);
      }
      assert(ret == CL_SUCCESS);
    }
  }

  if (thneed->debug >= 1) {
    debug_print(thneed->debug >= 2);
  }

  return clEnqueueNDRangeKernel(thneed->command_queue,
    kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}

uint64_t CLQueuedKernel::benchmark() {
  uint64_t ret = 0;
  int old_record = thneed->record;
  thneed->record = 0;
  clFinish(thneed->command_queue);
  // TODO: benchmarking at a lower level will make this more accurate
  for (int i = 0; i < 10; i++) {
    uint64_t sb = nanos_since_boot();
    exec();
    clFinish(thneed->command_queue);
    uint64_t et = nanos_since_boot() - sb;
    if (ret == 0 || et < ret) ret = et;
  }
  thneed->record = old_record;
  return ret;
}

void CLQueuedKernel::debug_print(bool verbose) {
  printf("%p %56s -- ", kernel, name.c_str());
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", global_work_size[i]);
  }
  printf(" -- ");
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", local_work_size[i]);
  }
  printf("\n");

  if (verbose) {
    for (int i = 0; i < num_args; i++) {
      string arg = args[i];
      printf("  %s %s", arg_types[i].c_str(), arg_names[i].c_str());
      void *arg_value = (void*)arg.data();
      int arg_size = arg.size();
      if (arg_size == 0) {
        printf(" (size) %d", args_size[i]);
      } else if (arg_size == 1) {
        printf(" = %d", *((char*)arg_value));
      } else if (arg_size == 2) {
        printf(" = %d", *((short*)arg_value));
      } else if (arg_size == 4) {
        if (arg_types[i] == "float") {
          printf(" = %f", *((float*)arg_value));
        } else {
          printf(" = %d", *((int*)arg_value));
        }
      } else if (arg_size == 8) {
        cl_mem val = (cl_mem)(*((uintptr_t*)arg_value));
        printf(" = %p", val);
        if (val != NULL) {
          if (arg_types[i] == "image2d_t" || arg_types[i] == "image1d_t") {
            cl_image_format format;
            size_t width, height, depth, array_size, row_pitch, slice_pitch;
            cl_mem buf;
            clGetImageInfo(val, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);
            assert(format.image_channel_order == CL_RGBA);
            assert(format.image_channel_data_type == CL_HALF_FLOAT);
            clGetImageInfo(val, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
            clGetImageInfo(val, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
            clGetImageInfo(val, CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch, NULL);
            clGetImageInfo(val, CL_IMAGE_DEPTH, sizeof(depth), &depth, NULL);
            clGetImageInfo(val, CL_IMAGE_ARRAY_SIZE, sizeof(array_size), &array_size, NULL);
            clGetImageInfo(val, CL_IMAGE_SLICE_PITCH, sizeof(slice_pitch), &slice_pitch, NULL);
            assert(depth == 0);
            assert(array_size == 0);
            assert(slice_pitch == 0);

            clGetImageInfo(val, CL_IMAGE_BUFFER, sizeof(buf), &buf, NULL);
            size_t sz;
            clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
            printf(" image %zu x %zu rp %zu @ %p buffer %zu", width, height, row_pitch, buf, sz);
          } else {
            size_t sz;
            clGetMemObjectInfo(val, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
            printf(" buffer %zu", sz);
          }
        }
      }
      printf("\n");
    }
  }
}
