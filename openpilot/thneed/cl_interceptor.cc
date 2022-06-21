// *********** OpenCL interceptor ***********

cl_int thneed_clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  g_args_size[make_pair(kernel, arg_index)] = arg_size;
  if (arg_value != NULL) {
    g_args[make_pair(kernel, arg_index)] = string((char*)arg_value, arg_size);
  } else {
    g_args[make_pair(kernel, arg_index)] = string("");
  }
  cl_int ret = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
  return ret;
}

cl_int thneed_clEnqueueNDRangeKernel(cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event) {

  Thneed *thneed = g_thneed;

  // SNPE doesn't use these
  assert(num_events_in_wait_list == 0);
  assert(global_work_offset == NULL);
  assert(event_wait_list == NULL);

  cl_int ret = 0;
  if (thneed != NULL && thneed->record) {
    if (thneed->context == NULL) {
      thneed->command_queue = command_queue;
      clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(thneed->context), &thneed->context, NULL);
      clGetContextInfo(thneed->context, CL_CONTEXT_DEVICES, sizeof(thneed->device_id), &thneed->device_id, NULL);
    }

    // if we are recording, we don't actually enqueue the kernel
    thneed->kq.push_back(unique_ptr<CLQueuedKernel>(new CLQueuedKernel(thneed, kernel, work_dim, global_work_size, local_work_size)));
    *event = NULL;
  } else {
    ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
      global_work_offset, global_work_size, local_work_size,
      num_events_in_wait_list, event_wait_list, event);
  }

  return ret;
}

cl_int thneed_clFinish(cl_command_queue command_queue) {
  Thneed *thneed = g_thneed;

  if (thneed != NULL && thneed->record) {
    if (thneed->run_optimizer) thneed->optimize();
    return thneed->clexec();
  } else {
    return clFinish(command_queue);
  }
}

cl_program thneed_clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) {
  assert(count == 1);
  cl_program ret = clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
  g_program_source[ret] = strings[0];
  return ret;
}

void *dlsym(void *handle, const char *symbol) {
#ifdef QCOM2
  void *(*my_dlsym)(void *handle, const char *symbol) = (void *(*)(void *handle, const char *symbol))((uintptr_t)dlopen + DLSYM_OFFSET);
#else
  #error "Unsupported platform for thneed"
#endif
  if (memcmp("REAL_", symbol, 5) == 0) {
    return my_dlsym(handle, symbol+5);
  } else if (strcmp("clFinish", symbol) == 0) {
    return (void*)thneed_clFinish;
  } else if (strcmp("clEnqueueNDRangeKernel", symbol) == 0) {
    return (void*)thneed_clEnqueueNDRangeKernel;
  } else if (strcmp("clSetKernelArg", symbol) == 0) {
    return (void*)thneed_clSetKernelArg;
  } else if (strcmp("clCreateProgramWithSource", symbol) == 0) {
    return (void*)thneed_clCreateProgramWithSource;
  } else {
    return my_dlsym(handle, symbol);
  }
}
