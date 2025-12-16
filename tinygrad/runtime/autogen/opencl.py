# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('opencl', 'OpenCL')
class struct__cl_platform_id(Struct): pass
cl_platform_id = Pointer(struct__cl_platform_id)
class struct__cl_device_id(Struct): pass
cl_device_id = Pointer(struct__cl_device_id)
class struct__cl_context(Struct): pass
cl_context = Pointer(struct__cl_context)
class struct__cl_command_queue(Struct): pass
cl_command_queue = Pointer(struct__cl_command_queue)
class struct__cl_mem(Struct): pass
cl_mem = Pointer(struct__cl_mem)
class struct__cl_program(Struct): pass
cl_program = Pointer(struct__cl_program)
class struct__cl_kernel(Struct): pass
cl_kernel = Pointer(struct__cl_kernel)
class struct__cl_event(Struct): pass
cl_event = Pointer(struct__cl_event)
class struct__cl_sampler(Struct): pass
cl_sampler = Pointer(struct__cl_sampler)
cl_bool = ctypes.c_uint32
cl_bitfield = ctypes.c_uint64
cl_properties = ctypes.c_uint64
cl_device_type = ctypes.c_uint64
cl_platform_info = ctypes.c_uint32
cl_device_info = ctypes.c_uint32
cl_device_fp_config = ctypes.c_uint64
cl_device_mem_cache_type = ctypes.c_uint32
cl_device_local_mem_type = ctypes.c_uint32
cl_device_exec_capabilities = ctypes.c_uint64
cl_device_svm_capabilities = ctypes.c_uint64
cl_command_queue_properties = ctypes.c_uint64
cl_device_partition_property = ctypes.c_int64
cl_device_affinity_domain = ctypes.c_uint64
cl_context_properties = ctypes.c_int64
cl_context_info = ctypes.c_uint32
cl_queue_properties = ctypes.c_uint64
cl_command_queue_info = ctypes.c_uint32
cl_channel_order = ctypes.c_uint32
cl_channel_type = ctypes.c_uint32
cl_mem_flags = ctypes.c_uint64
cl_svm_mem_flags = ctypes.c_uint64
cl_mem_object_type = ctypes.c_uint32
cl_mem_info = ctypes.c_uint32
cl_mem_migration_flags = ctypes.c_uint64
cl_image_info = ctypes.c_uint32
cl_buffer_create_type = ctypes.c_uint32
cl_addressing_mode = ctypes.c_uint32
cl_filter_mode = ctypes.c_uint32
cl_sampler_info = ctypes.c_uint32
cl_map_flags = ctypes.c_uint64
cl_pipe_properties = ctypes.c_int64
cl_pipe_info = ctypes.c_uint32
cl_program_info = ctypes.c_uint32
cl_program_build_info = ctypes.c_uint32
cl_program_binary_type = ctypes.c_uint32
cl_build_status = ctypes.c_int32
cl_kernel_info = ctypes.c_uint32
cl_kernel_arg_info = ctypes.c_uint32
cl_kernel_arg_address_qualifier = ctypes.c_uint32
cl_kernel_arg_access_qualifier = ctypes.c_uint32
cl_kernel_arg_type_qualifier = ctypes.c_uint64
cl_kernel_work_group_info = ctypes.c_uint32
cl_kernel_sub_group_info = ctypes.c_uint32
cl_event_info = ctypes.c_uint32
cl_command_type = ctypes.c_uint32
cl_profiling_info = ctypes.c_uint32
cl_sampler_properties = ctypes.c_uint64
cl_kernel_exec_info = ctypes.c_uint32
cl_device_atomic_capabilities = ctypes.c_uint64
cl_device_device_enqueue_capabilities = ctypes.c_uint64
cl_khronos_vendor_id = ctypes.c_uint32
cl_mem_properties = ctypes.c_uint64
cl_version = ctypes.c_uint32
class struct__cl_image_format(Struct): pass
struct__cl_image_format.SIZE = 8
struct__cl_image_format._fields_ = ['image_channel_order', 'image_channel_data_type']
setattr(struct__cl_image_format, 'image_channel_order', field(0, cl_channel_order))
setattr(struct__cl_image_format, 'image_channel_data_type', field(4, cl_channel_type))
cl_image_format = struct__cl_image_format
class struct__cl_image_desc(Struct): pass
size_t = ctypes.c_uint64
cl_uint = ctypes.c_uint32
struct__cl_image_desc.SIZE = 72
struct__cl_image_desc._fields_ = ['image_type', 'image_width', 'image_height', 'image_depth', 'image_array_size', 'image_row_pitch', 'image_slice_pitch', 'num_mip_levels', 'num_samples', 'buffer', 'mem_object']
setattr(struct__cl_image_desc, 'image_type', field(0, cl_mem_object_type))
setattr(struct__cl_image_desc, 'image_width', field(8, size_t))
setattr(struct__cl_image_desc, 'image_height', field(16, size_t))
setattr(struct__cl_image_desc, 'image_depth', field(24, size_t))
setattr(struct__cl_image_desc, 'image_array_size', field(32, size_t))
setattr(struct__cl_image_desc, 'image_row_pitch', field(40, size_t))
setattr(struct__cl_image_desc, 'image_slice_pitch', field(48, size_t))
setattr(struct__cl_image_desc, 'num_mip_levels', field(56, cl_uint))
setattr(struct__cl_image_desc, 'num_samples', field(60, cl_uint))
setattr(struct__cl_image_desc, 'buffer', field(64, cl_mem))
setattr(struct__cl_image_desc, 'mem_object', field(64, cl_mem))
cl_image_desc = struct__cl_image_desc
class struct__cl_buffer_region(Struct): pass
struct__cl_buffer_region.SIZE = 16
struct__cl_buffer_region._fields_ = ['origin', 'size']
setattr(struct__cl_buffer_region, 'origin', field(0, size_t))
setattr(struct__cl_buffer_region, 'size', field(8, size_t))
cl_buffer_region = struct__cl_buffer_region
class struct__cl_name_version(Struct): pass
struct__cl_name_version.SIZE = 68
struct__cl_name_version._fields_ = ['version', 'name']
setattr(struct__cl_name_version, 'version', field(0, cl_version))
setattr(struct__cl_name_version, 'name', field(4, Array(ctypes.c_char, 64)))
cl_name_version = struct__cl_name_version
cl_int = ctypes.c_int32
@dll.bind((cl_uint, Pointer(cl_platform_id), Pointer(cl_uint),), cl_int)
def clGetPlatformIDs(num_entries, platforms, num_platforms): ...
@dll.bind((cl_platform_id, cl_platform_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_platform_id, cl_device_type, cl_uint, Pointer(cl_device_id), Pointer(cl_uint),), cl_int)
def clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices): ...
@dll.bind((cl_device_id, cl_device_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_device_id, Pointer(cl_device_partition_property), cl_uint, Pointer(cl_device_id), Pointer(cl_uint),), cl_int)
def clCreateSubDevices(in_device, properties, num_devices, out_devices, num_devices_ret): ...
@dll.bind((cl_device_id,), cl_int)
def clRetainDevice(device): ...
@dll.bind((cl_device_id,), cl_int)
def clReleaseDevice(device): ...
@dll.bind((cl_context, cl_device_id, cl_command_queue,), cl_int)
def clSetDefaultDeviceCommandQueue(context, device, command_queue): ...
cl_ulong = ctypes.c_uint64
@dll.bind((cl_device_id, Pointer(cl_ulong), Pointer(cl_ulong),), cl_int)
def clGetDeviceAndHostTimer(device, device_timestamp, host_timestamp): ...
@dll.bind((cl_device_id, Pointer(cl_ulong),), cl_int)
def clGetHostTimer(device, host_timestamp): ...
@dll.bind((Pointer(cl_context_properties), cl_uint, Pointer(cl_device_id), ctypes.CFUNCTYPE(None, Pointer(ctypes.c_char), ctypes.c_void_p, size_t, ctypes.c_void_p), ctypes.c_void_p, Pointer(cl_int),), cl_context)
def clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret): ...
@dll.bind((Pointer(cl_context_properties), cl_device_type, ctypes.CFUNCTYPE(None, Pointer(ctypes.c_char), ctypes.c_void_p, size_t, ctypes.c_void_p), ctypes.c_void_p, Pointer(cl_int),), cl_context)
def clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret): ...
@dll.bind((cl_context,), cl_int)
def clRetainContext(context): ...
@dll.bind((cl_context,), cl_int)
def clReleaseContext(context): ...
@dll.bind((cl_context, cl_context_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetContextInfo(context, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_context, ctypes.CFUNCTYPE(None, cl_context, ctypes.c_void_p), ctypes.c_void_p,), cl_int)
def clSetContextDestructorCallback(context, pfn_notify, user_data): ...
@dll.bind((cl_context, cl_device_id, Pointer(cl_queue_properties), Pointer(cl_int),), cl_command_queue)
def clCreateCommandQueueWithProperties(context, device, properties, errcode_ret): ...
@dll.bind((cl_command_queue,), cl_int)
def clRetainCommandQueue(command_queue): ...
@dll.bind((cl_command_queue,), cl_int)
def clReleaseCommandQueue(command_queue): ...
@dll.bind((cl_command_queue, cl_command_queue_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetCommandQueueInfo(command_queue, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_context, cl_mem_flags, size_t, ctypes.c_void_p, Pointer(cl_int),), cl_mem)
def clCreateBuffer(context, flags, size, host_ptr, errcode_ret): ...
@dll.bind((cl_mem, cl_mem_flags, cl_buffer_create_type, ctypes.c_void_p, Pointer(cl_int),), cl_mem)
def clCreateSubBuffer(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret): ...
@dll.bind((cl_context, cl_mem_flags, Pointer(cl_image_format), Pointer(cl_image_desc), ctypes.c_void_p, Pointer(cl_int),), cl_mem)
def clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret): ...
@dll.bind((cl_context, cl_mem_flags, cl_uint, cl_uint, Pointer(cl_pipe_properties), Pointer(cl_int),), cl_mem)
def clCreatePipe(context, flags, pipe_packet_size, pipe_max_packets, properties, errcode_ret): ...
@dll.bind((cl_context, Pointer(cl_mem_properties), cl_mem_flags, size_t, ctypes.c_void_p, Pointer(cl_int),), cl_mem)
def clCreateBufferWithProperties(context, properties, flags, size, host_ptr, errcode_ret): ...
@dll.bind((cl_context, Pointer(cl_mem_properties), cl_mem_flags, Pointer(cl_image_format), Pointer(cl_image_desc), ctypes.c_void_p, Pointer(cl_int),), cl_mem)
def clCreateImageWithProperties(context, properties, flags, image_format, image_desc, host_ptr, errcode_ret): ...
@dll.bind((cl_mem,), cl_int)
def clRetainMemObject(memobj): ...
@dll.bind((cl_mem,), cl_int)
def clReleaseMemObject(memobj): ...
@dll.bind((cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, Pointer(cl_image_format), Pointer(cl_uint),), cl_int)
def clGetSupportedImageFormats(context, flags, image_type, num_entries, image_formats, num_image_formats): ...
@dll.bind((cl_mem, cl_mem_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetMemObjectInfo(memobj, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_mem, cl_image_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetImageInfo(image, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_mem, cl_pipe_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetPipeInfo(pipe, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_mem, ctypes.CFUNCTYPE(None, cl_mem, ctypes.c_void_p), ctypes.c_void_p,), cl_int)
def clSetMemObjectDestructorCallback(memobj, pfn_notify, user_data): ...
@dll.bind((cl_context, cl_svm_mem_flags, size_t, cl_uint,), ctypes.c_void_p)
def clSVMAlloc(context, flags, size, alignment): ...
@dll.bind((cl_context, ctypes.c_void_p,), None)
def clSVMFree(context, svm_pointer): ...
@dll.bind((cl_context, Pointer(cl_sampler_properties), Pointer(cl_int),), cl_sampler)
def clCreateSamplerWithProperties(context, sampler_properties, errcode_ret): ...
@dll.bind((cl_sampler,), cl_int)
def clRetainSampler(sampler): ...
@dll.bind((cl_sampler,), cl_int)
def clReleaseSampler(sampler): ...
@dll.bind((cl_sampler, cl_sampler_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetSamplerInfo(sampler, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_context, cl_uint, Pointer(Pointer(ctypes.c_char)), Pointer(size_t), Pointer(cl_int),), cl_program)
def clCreateProgramWithSource(context, count, strings, lengths, errcode_ret): ...
@dll.bind((cl_context, cl_uint, Pointer(cl_device_id), Pointer(size_t), Pointer(Pointer(ctypes.c_ubyte)), Pointer(cl_int), Pointer(cl_int),), cl_program)
def clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret): ...
@dll.bind((cl_context, cl_uint, Pointer(cl_device_id), Pointer(ctypes.c_char), Pointer(cl_int),), cl_program)
def clCreateProgramWithBuiltInKernels(context, num_devices, device_list, kernel_names, errcode_ret): ...
@dll.bind((cl_context, ctypes.c_void_p, size_t, Pointer(cl_int),), cl_program)
def clCreateProgramWithIL(context, il, length, errcode_ret): ...
@dll.bind((cl_program,), cl_int)
def clRetainProgram(program): ...
@dll.bind((cl_program,), cl_int)
def clReleaseProgram(program): ...
@dll.bind((cl_program, cl_uint, Pointer(cl_device_id), Pointer(ctypes.c_char), ctypes.CFUNCTYPE(None, cl_program, ctypes.c_void_p), ctypes.c_void_p,), cl_int)
def clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data): ...
@dll.bind((cl_program, cl_uint, Pointer(cl_device_id), Pointer(ctypes.c_char), cl_uint, Pointer(cl_program), Pointer(Pointer(ctypes.c_char)), ctypes.CFUNCTYPE(None, cl_program, ctypes.c_void_p), ctypes.c_void_p,), cl_int)
def clCompileProgram(program, num_devices, device_list, options, num_input_headers, input_headers, header_include_names, pfn_notify, user_data): ...
@dll.bind((cl_context, cl_uint, Pointer(cl_device_id), Pointer(ctypes.c_char), cl_uint, Pointer(cl_program), ctypes.CFUNCTYPE(None, cl_program, ctypes.c_void_p), ctypes.c_void_p, Pointer(cl_int),), cl_program)
def clLinkProgram(context, num_devices, device_list, options, num_input_programs, input_programs, pfn_notify, user_data, errcode_ret): ...
@dll.bind((cl_program, ctypes.CFUNCTYPE(None, cl_program, ctypes.c_void_p), ctypes.c_void_p,), cl_int)
def clSetProgramReleaseCallback(program, pfn_notify, user_data): ...
@dll.bind((cl_program, cl_uint, size_t, ctypes.c_void_p,), cl_int)
def clSetProgramSpecializationConstant(program, spec_id, spec_size, spec_value): ...
@dll.bind((cl_platform_id,), cl_int)
def clUnloadPlatformCompiler(platform): ...
@dll.bind((cl_program, cl_program_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_program, cl_device_id, cl_program_build_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetProgramBuildInfo(program, device, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_program, Pointer(ctypes.c_char), Pointer(cl_int),), cl_kernel)
def clCreateKernel(program, kernel_name, errcode_ret): ...
@dll.bind((cl_program, cl_uint, Pointer(cl_kernel), Pointer(cl_uint),), cl_int)
def clCreateKernelsInProgram(program, num_kernels, kernels, num_kernels_ret): ...
@dll.bind((cl_kernel, Pointer(cl_int),), cl_kernel)
def clCloneKernel(source_kernel, errcode_ret): ...
@dll.bind((cl_kernel,), cl_int)
def clRetainKernel(kernel): ...
@dll.bind((cl_kernel,), cl_int)
def clReleaseKernel(kernel): ...
@dll.bind((cl_kernel, cl_uint, size_t, ctypes.c_void_p,), cl_int)
def clSetKernelArg(kernel, arg_index, arg_size, arg_value): ...
@dll.bind((cl_kernel, cl_uint, ctypes.c_void_p,), cl_int)
def clSetKernelArgSVMPointer(kernel, arg_index, arg_value): ...
@dll.bind((cl_kernel, cl_kernel_exec_info, size_t, ctypes.c_void_p,), cl_int)
def clSetKernelExecInfo(kernel, param_name, param_value_size, param_value): ...
@dll.bind((cl_kernel, cl_kernel_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetKernelInfo(kernel, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_kernel, cl_uint, cl_kernel_arg_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetKernelArgInfo(kernel, arg_indx, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetKernelWorkGroupInfo(kernel, device, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_kernel, cl_device_id, cl_kernel_sub_group_info, size_t, ctypes.c_void_p, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetKernelSubGroupInfo(kernel, device, param_name, input_value_size, input_value, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_uint, Pointer(cl_event),), cl_int)
def clWaitForEvents(num_events, event_list): ...
@dll.bind((cl_event, cl_event_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetEventInfo(event, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_context, Pointer(cl_int),), cl_event)
def clCreateUserEvent(context, errcode_ret): ...
@dll.bind((cl_event,), cl_int)
def clRetainEvent(event): ...
@dll.bind((cl_event,), cl_int)
def clReleaseEvent(event): ...
@dll.bind((cl_event, cl_int,), cl_int)
def clSetUserEventStatus(event, execution_status): ...
@dll.bind((cl_event, cl_int, ctypes.CFUNCTYPE(None, cl_event, cl_int, ctypes.c_void_p), ctypes.c_void_p,), cl_int)
def clSetEventCallback(event, command_exec_callback_type, pfn_notify, user_data): ...
@dll.bind((cl_event, cl_profiling_info, size_t, ctypes.c_void_p, Pointer(size_t),), cl_int)
def clGetEventProfilingInfo(event, param_name, param_value_size, param_value, param_value_size_ret): ...
@dll.bind((cl_command_queue,), cl_int)
def clFlush(command_queue): ...
@dll.bind((cl_command_queue,), cl_int)
def clFinish(command_queue): ...
@dll.bind((cl_command_queue, cl_mem, cl_bool, size_t, size_t, ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_bool, Pointer(size_t), Pointer(size_t), Pointer(size_t), size_t, size_t, size_t, size_t, ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueReadBufferRect(command_queue, buffer, blocking_read, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_bool, size_t, size_t, ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_bool, Pointer(size_t), Pointer(size_t), Pointer(size_t), size_t, size_t, size_t, size_t, ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueWriteBufferRect(command_queue, buffer, blocking_write, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, ctypes.c_void_p, size_t, size_t, size_t, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueFillBuffer(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_mem, Pointer(size_t), Pointer(size_t), Pointer(size_t), size_t, size_t, size_t, size_t, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueCopyBufferRect(command_queue, src_buffer, dst_buffer, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_bool, Pointer(size_t), Pointer(size_t), size_t, size_t, ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueReadImage(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_bool, Pointer(size_t), Pointer(size_t), size_t, size_t, ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueWriteImage(command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, ctypes.c_void_p, Pointer(size_t), Pointer(size_t), cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueFillImage(command_queue, image, fill_color, origin, region, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_mem, Pointer(size_t), Pointer(size_t), Pointer(size_t), cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueCopyImage(command_queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_mem, Pointer(size_t), Pointer(size_t), size_t, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueCopyImageToBuffer(command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_mem, size_t, Pointer(size_t), Pointer(size_t), cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueCopyBufferToImage(command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, Pointer(cl_event), Pointer(cl_event), Pointer(cl_int),), ctypes.c_void_p)
def clEnqueueMapBuffer(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret): ...
@dll.bind((cl_command_queue, cl_mem, cl_bool, cl_map_flags, Pointer(size_t), Pointer(size_t), Pointer(size_t), Pointer(size_t), cl_uint, Pointer(cl_event), Pointer(cl_event), Pointer(cl_int),), ctypes.c_void_p)
def clEnqueueMapImage(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch, num_events_in_wait_list, event_wait_list, event, errcode_ret): ...
@dll.bind((cl_command_queue, cl_mem, ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueUnmapMemObject(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_uint, Pointer(cl_mem), cl_mem_migration_flags, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueMigrateMemObjects(command_queue, num_mem_objects, mem_objects, flags, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_kernel, cl_uint, Pointer(size_t), Pointer(size_t), Pointer(size_t), cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, ctypes.CFUNCTYPE(None, ctypes.c_void_p), ctypes.c_void_p, size_t, cl_uint, Pointer(cl_mem), Pointer(ctypes.c_void_p), cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueNativeKernel(command_queue, user_func, args, cb_args, num_mem_objects, mem_list, args_mem_loc, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueMarkerWithWaitList(command_queue, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueBarrierWithWaitList(command_queue, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_uint, Array(ctypes.c_void_p, 0), ctypes.CFUNCTYPE(None, cl_command_queue, cl_uint, Array(ctypes.c_void_p, 0), ctypes.c_void_p), ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueSVMFree(command_queue, num_svm_pointers, svm_pointers, pfn_free_func, user_data, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_bool, ctypes.c_void_p, ctypes.c_void_p, size_t, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueSVMMemcpy(command_queue, blocking_copy, dst_ptr, src_ptr, size, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueSVMMemFill(command_queue, svm_ptr, pattern, pattern_size, size, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_bool, cl_map_flags, ctypes.c_void_p, size_t, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueSVMMap(command_queue, blocking_map, flags, svm_ptr, size, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, ctypes.c_void_p, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueSVMUnmap(command_queue, svm_ptr, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_command_queue, cl_uint, Pointer(ctypes.c_void_p), Pointer(size_t), cl_mem_migration_flags, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueSVMMigrateMem(command_queue, num_svm_pointers, svm_pointers, sizes, flags, num_events_in_wait_list, event_wait_list, event): ...
@dll.bind((cl_platform_id, Pointer(ctypes.c_char),), ctypes.c_void_p)
def clGetExtensionFunctionAddressForPlatform(platform, func_name): ...
@dll.bind((cl_context, cl_mem_flags, Pointer(cl_image_format), size_t, size_t, size_t, ctypes.c_void_p, Pointer(cl_int),), cl_mem)
def clCreateImage2D(context, flags, image_format, image_width, image_height, image_row_pitch, host_ptr, errcode_ret): ...
@dll.bind((cl_context, cl_mem_flags, Pointer(cl_image_format), size_t, size_t, size_t, size_t, size_t, ctypes.c_void_p, Pointer(cl_int),), cl_mem)
def clCreateImage3D(context, flags, image_format, image_width, image_height, image_depth, image_row_pitch, image_slice_pitch, host_ptr, errcode_ret): ...
@dll.bind((cl_command_queue, Pointer(cl_event),), cl_int)
def clEnqueueMarker(command_queue, event): ...
@dll.bind((cl_command_queue, cl_uint, Pointer(cl_event),), cl_int)
def clEnqueueWaitForEvents(command_queue, num_events, event_list): ...
@dll.bind((cl_command_queue,), cl_int)
def clEnqueueBarrier(command_queue): ...
@dll.bind((), cl_int)
def clUnloadCompiler(): ...
@dll.bind((Pointer(ctypes.c_char),), ctypes.c_void_p)
def clGetExtensionFunctionAddress(func_name): ...
@dll.bind((cl_context, cl_device_id, cl_command_queue_properties, Pointer(cl_int),), cl_command_queue)
def clCreateCommandQueue(context, device, properties, errcode_ret): ...
@dll.bind((cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, Pointer(cl_int),), cl_sampler)
def clCreateSampler(context, normalized_coords, addressing_mode, filter_mode, errcode_ret): ...
@dll.bind((cl_command_queue, cl_kernel, cl_uint, Pointer(cl_event), Pointer(cl_event),), cl_int)
def clEnqueueTask(command_queue, kernel, num_events_in_wait_list, event_wait_list, event): ...
CL_NAME_VERSION_MAX_NAME_SIZE = 64
CL_SUCCESS = 0
CL_DEVICE_NOT_FOUND = -1
CL_DEVICE_NOT_AVAILABLE = -2
CL_COMPILER_NOT_AVAILABLE = -3
CL_MEM_OBJECT_ALLOCATION_FAILURE = -4
CL_OUT_OF_RESOURCES = -5
CL_OUT_OF_HOST_MEMORY = -6
CL_PROFILING_INFO_NOT_AVAILABLE = -7
CL_MEM_COPY_OVERLAP = -8
CL_IMAGE_FORMAT_MISMATCH = -9
CL_IMAGE_FORMAT_NOT_SUPPORTED = -10
CL_BUILD_PROGRAM_FAILURE = -11
CL_MAP_FAILURE = -12
CL_MISALIGNED_SUB_BUFFER_OFFSET = -13
CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14
CL_COMPILE_PROGRAM_FAILURE = -15
CL_LINKER_NOT_AVAILABLE = -16
CL_LINK_PROGRAM_FAILURE = -17
CL_DEVICE_PARTITION_FAILED = -18
CL_KERNEL_ARG_INFO_NOT_AVAILABLE = -19
CL_INVALID_VALUE = -30
CL_INVALID_DEVICE_TYPE = -31
CL_INVALID_PLATFORM = -32
CL_INVALID_DEVICE = -33
CL_INVALID_CONTEXT = -34
CL_INVALID_QUEUE_PROPERTIES = -35
CL_INVALID_COMMAND_QUEUE = -36
CL_INVALID_HOST_PTR = -37
CL_INVALID_MEM_OBJECT = -38
CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39
CL_INVALID_IMAGE_SIZE = -40
CL_INVALID_SAMPLER = -41
CL_INVALID_BINARY = -42
CL_INVALID_BUILD_OPTIONS = -43
CL_INVALID_PROGRAM = -44
CL_INVALID_PROGRAM_EXECUTABLE = -45
CL_INVALID_KERNEL_NAME = -46
CL_INVALID_KERNEL_DEFINITION = -47
CL_INVALID_KERNEL = -48
CL_INVALID_ARG_INDEX = -49
CL_INVALID_ARG_VALUE = -50
CL_INVALID_ARG_SIZE = -51
CL_INVALID_KERNEL_ARGS = -52
CL_INVALID_WORK_DIMENSION = -53
CL_INVALID_WORK_GROUP_SIZE = -54
CL_INVALID_WORK_ITEM_SIZE = -55
CL_INVALID_GLOBAL_OFFSET = -56
CL_INVALID_EVENT_WAIT_LIST = -57
CL_INVALID_EVENT = -58
CL_INVALID_OPERATION = -59
CL_INVALID_GL_OBJECT = -60
CL_INVALID_BUFFER_SIZE = -61
CL_INVALID_MIP_LEVEL = -62
CL_INVALID_GLOBAL_WORK_SIZE = -63
CL_INVALID_PROPERTY = -64
CL_INVALID_IMAGE_DESCRIPTOR = -65
CL_INVALID_COMPILER_OPTIONS = -66
CL_INVALID_LINKER_OPTIONS = -67
CL_INVALID_DEVICE_PARTITION_COUNT = -68
CL_INVALID_PIPE_SIZE = -69
CL_INVALID_DEVICE_QUEUE = -70
CL_INVALID_SPEC_ID = -71
CL_MAX_SIZE_RESTRICTION_EXCEEDED = -72
CL_FALSE = 0
CL_TRUE = 1
CL_BLOCKING = CL_TRUE
CL_NON_BLOCKING = CL_FALSE
CL_PLATFORM_PROFILE = 0x0900
CL_PLATFORM_VERSION = 0x0901
CL_PLATFORM_NAME = 0x0902
CL_PLATFORM_VENDOR = 0x0903
CL_PLATFORM_EXTENSIONS = 0x0904
CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905
CL_PLATFORM_NUMERIC_VERSION = 0x0906
CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907
CL_DEVICE_TYPE_DEFAULT = (1 << 0)
CL_DEVICE_TYPE_CPU = (1 << 1)
CL_DEVICE_TYPE_GPU = (1 << 2)
CL_DEVICE_TYPE_ACCELERATOR = (1 << 3)
CL_DEVICE_TYPE_CUSTOM = (1 << 4)
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_DEVICE_TYPE = 0x1000
CL_DEVICE_VENDOR_ID = 0x1001
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005
CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006
CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007
CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008
CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009
CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A
CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100B
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
CL_DEVICE_ADDRESS_BITS = 0x100D
CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100E
CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100F
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011
CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012
CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013
CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014
CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015
CL_DEVICE_IMAGE_SUPPORT = 0x1016
CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017
CL_DEVICE_MAX_SAMPLERS = 0x1018
CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019
CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A
CL_DEVICE_SINGLE_FP_CONFIG = 0x101B
CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C
CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020
CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021
CL_DEVICE_LOCAL_MEM_TYPE = 0x1022
CL_DEVICE_LOCAL_MEM_SIZE = 0x1023
CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024
CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025
CL_DEVICE_ENDIAN_LITTLE = 0x1026
CL_DEVICE_AVAILABLE = 0x1027
CL_DEVICE_COMPILER_AVAILABLE = 0x1028
CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029
CL_DEVICE_QUEUE_PROPERTIES = 0x102A
CL_DEVICE_QUEUE_ON_HOST_PROPERTIES = 0x102A
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_PROFILE = 0x102E
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_PLATFORM = 0x1031
CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032
CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034
CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035
CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036
CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037
CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038
CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039
CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A
CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B
CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C
CL_DEVICE_OPENCL_C_VERSION = 0x103D
CL_DEVICE_LINKER_AVAILABLE = 0x103E
CL_DEVICE_BUILT_IN_KERNELS = 0x103F
CL_DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040
CL_DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041
CL_DEVICE_PARENT_DEVICE = 0x1042
CL_DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043
CL_DEVICE_PARTITION_PROPERTIES = 0x1044
CL_DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045
CL_DEVICE_PARTITION_TYPE = 0x1046
CL_DEVICE_REFERENCE_COUNT = 0x1047
CL_DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048
CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049
CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104A
CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104B
CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS = 0x104C
CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE = 0x104D
CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES = 0x104E
CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE = 0x104F
CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = 0x1050
CL_DEVICE_MAX_ON_DEVICE_QUEUES = 0x1051
CL_DEVICE_MAX_ON_DEVICE_EVENTS = 0x1052
CL_DEVICE_SVM_CAPABILITIES = 0x1053
CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = 0x1054
CL_DEVICE_MAX_PIPE_ARGS = 0x1055
CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056
CL_DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057
CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058
CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059
CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105A
CL_DEVICE_IL_VERSION = 0x105B
CL_DEVICE_MAX_NUM_SUB_GROUPS = 0x105C
CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105D
CL_DEVICE_NUMERIC_VERSION = 0x105E
CL_DEVICE_EXTENSIONS_WITH_VERSION = 0x1060
CL_DEVICE_ILS_WITH_VERSION = 0x1061
CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION = 0x1062
CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES = 0x1063
CL_DEVICE_ATOMIC_FENCE_CAPABILITIES = 0x1064
CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT = 0x1065
CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066
CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x1067
CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = 0x1068
CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT = 0x1069
CL_DEVICE_OPENCL_C_FEATURES = 0x106F
CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES = 0x1070
CL_DEVICE_PIPE_SUPPORT = 0x1071
CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED = 0x1072
CL_FP_DENORM = (1 << 0)
CL_FP_INF_NAN = (1 << 1)
CL_FP_ROUND_TO_NEAREST = (1 << 2)
CL_FP_ROUND_TO_ZERO = (1 << 3)
CL_FP_ROUND_TO_INF = (1 << 4)
CL_FP_FMA = (1 << 5)
CL_FP_SOFT_FLOAT = (1 << 6)
CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = (1 << 7)
CL_NONE = 0x0
CL_READ_ONLY_CACHE = 0x1
CL_READ_WRITE_CACHE = 0x2
CL_LOCAL = 0x1
CL_GLOBAL = 0x2
CL_EXEC_KERNEL = (1 << 0)
CL_EXEC_NATIVE_KERNEL = (1 << 1)
CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = (1 << 0)
CL_QUEUE_PROFILING_ENABLE = (1 << 1)
CL_QUEUE_ON_DEVICE = (1 << 2)
CL_QUEUE_ON_DEVICE_DEFAULT = (1 << 3)
CL_CONTEXT_REFERENCE_COUNT = 0x1080
CL_CONTEXT_DEVICES = 0x1081
CL_CONTEXT_PROPERTIES = 0x1082
CL_CONTEXT_NUM_DEVICES = 0x1083
CL_CONTEXT_PLATFORM = 0x1084
CL_CONTEXT_INTEROP_USER_SYNC = 0x1085
CL_DEVICE_PARTITION_EQUALLY = 0x1086
CL_DEVICE_PARTITION_BY_COUNTS = 0x1087
CL_DEVICE_PARTITION_BY_COUNTS_LIST_END = 0x0
CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN = 0x1088
CL_DEVICE_AFFINITY_DOMAIN_NUMA = (1 << 0)
CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE = (1 << 1)
CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE = (1 << 2)
CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE = (1 << 3)
CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE = (1 << 4)
CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE = (1 << 5)
CL_DEVICE_SVM_COARSE_GRAIN_BUFFER = (1 << 0)
CL_DEVICE_SVM_FINE_GRAIN_BUFFER = (1 << 1)
CL_DEVICE_SVM_FINE_GRAIN_SYSTEM = (1 << 2)
CL_DEVICE_SVM_ATOMICS = (1 << 3)
CL_QUEUE_CONTEXT = 0x1090
CL_QUEUE_DEVICE = 0x1091
CL_QUEUE_REFERENCE_COUNT = 0x1092
CL_QUEUE_PROPERTIES = 0x1093
CL_QUEUE_SIZE = 0x1094
CL_QUEUE_DEVICE_DEFAULT = 0x1095
CL_QUEUE_PROPERTIES_ARRAY = 0x1098
CL_MEM_READ_WRITE = (1 << 0)
CL_MEM_WRITE_ONLY = (1 << 1)
CL_MEM_READ_ONLY = (1 << 2)
CL_MEM_USE_HOST_PTR = (1 << 3)
CL_MEM_ALLOC_HOST_PTR = (1 << 4)
CL_MEM_COPY_HOST_PTR = (1 << 5)
CL_MEM_HOST_WRITE_ONLY = (1 << 7)
CL_MEM_HOST_READ_ONLY = (1 << 8)
CL_MEM_HOST_NO_ACCESS = (1 << 9)
CL_MEM_SVM_FINE_GRAIN_BUFFER = (1 << 10)
CL_MEM_SVM_ATOMICS = (1 << 11)
CL_MEM_KERNEL_READ_AND_WRITE = (1 << 12)
CL_MIGRATE_MEM_OBJECT_HOST = (1 << 0)
CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED = (1 << 1)
CL_R = 0x10B0
CL_A = 0x10B1
CL_RG = 0x10B2
CL_RA = 0x10B3
CL_RGB = 0x10B4
CL_RGBA = 0x10B5
CL_BGRA = 0x10B6
CL_ARGB = 0x10B7
CL_INTENSITY = 0x10B8
CL_LUMINANCE = 0x10B9
CL_Rx = 0x10BA
CL_RGx = 0x10BB
CL_RGBx = 0x10BC
CL_DEPTH = 0x10BD
CL_sRGB = 0x10BF
CL_sRGBx = 0x10C0
CL_sRGBA = 0x10C1
CL_sBGRA = 0x10C2
CL_ABGR = 0x10C3
CL_SNORM_INT8 = 0x10D0
CL_SNORM_INT16 = 0x10D1
CL_UNORM_INT8 = 0x10D2
CL_UNORM_INT16 = 0x10D3
CL_UNORM_SHORT_565 = 0x10D4
CL_UNORM_SHORT_555 = 0x10D5
CL_UNORM_INT_101010 = 0x10D6
CL_SIGNED_INT8 = 0x10D7
CL_SIGNED_INT16 = 0x10D8
CL_SIGNED_INT32 = 0x10D9
CL_UNSIGNED_INT8 = 0x10DA
CL_UNSIGNED_INT16 = 0x10DB
CL_UNSIGNED_INT32 = 0x10DC
CL_HALF_FLOAT = 0x10DD
CL_FLOAT = 0x10DE
CL_UNORM_INT_101010_2 = 0x10E0
CL_MEM_OBJECT_BUFFER = 0x10F0
CL_MEM_OBJECT_IMAGE2D = 0x10F1
CL_MEM_OBJECT_IMAGE3D = 0x10F2
CL_MEM_OBJECT_IMAGE2D_ARRAY = 0x10F3
CL_MEM_OBJECT_IMAGE1D = 0x10F4
CL_MEM_OBJECT_IMAGE1D_ARRAY = 0x10F5
CL_MEM_OBJECT_IMAGE1D_BUFFER = 0x10F6
CL_MEM_OBJECT_PIPE = 0x10F7
CL_MEM_TYPE = 0x1100
CL_MEM_FLAGS = 0x1101
CL_MEM_SIZE = 0x1102
CL_MEM_HOST_PTR = 0x1103
CL_MEM_MAP_COUNT = 0x1104
CL_MEM_REFERENCE_COUNT = 0x1105
CL_MEM_CONTEXT = 0x1106
CL_MEM_ASSOCIATED_MEMOBJECT = 0x1107
CL_MEM_OFFSET = 0x1108
CL_MEM_USES_SVM_POINTER = 0x1109
CL_MEM_PROPERTIES = 0x110A
CL_IMAGE_FORMAT = 0x1110
CL_IMAGE_ELEMENT_SIZE = 0x1111
CL_IMAGE_ROW_PITCH = 0x1112
CL_IMAGE_SLICE_PITCH = 0x1113
CL_IMAGE_WIDTH = 0x1114
CL_IMAGE_HEIGHT = 0x1115
CL_IMAGE_DEPTH = 0x1116
CL_IMAGE_ARRAY_SIZE = 0x1117
CL_IMAGE_BUFFER = 0x1118
CL_IMAGE_NUM_MIP_LEVELS = 0x1119
CL_IMAGE_NUM_SAMPLES = 0x111A
CL_PIPE_PACKET_SIZE = 0x1120
CL_PIPE_MAX_PACKETS = 0x1121
CL_PIPE_PROPERTIES = 0x1122
CL_ADDRESS_NONE = 0x1130
CL_ADDRESS_CLAMP_TO_EDGE = 0x1131
CL_ADDRESS_CLAMP = 0x1132
CL_ADDRESS_REPEAT = 0x1133
CL_ADDRESS_MIRRORED_REPEAT = 0x1134
CL_FILTER_NEAREST = 0x1140
CL_FILTER_LINEAR = 0x1141
CL_SAMPLER_REFERENCE_COUNT = 0x1150
CL_SAMPLER_CONTEXT = 0x1151
CL_SAMPLER_NORMALIZED_COORDS = 0x1152
CL_SAMPLER_ADDRESSING_MODE = 0x1153
CL_SAMPLER_FILTER_MODE = 0x1154
CL_SAMPLER_MIP_FILTER_MODE = 0x1155
CL_SAMPLER_LOD_MIN = 0x1156
CL_SAMPLER_LOD_MAX = 0x1157
CL_SAMPLER_PROPERTIES = 0x1158
CL_MAP_READ = (1 << 0)
CL_MAP_WRITE = (1 << 1)
CL_MAP_WRITE_INVALIDATE_REGION = (1 << 2)
CL_PROGRAM_REFERENCE_COUNT = 0x1160
CL_PROGRAM_CONTEXT = 0x1161
CL_PROGRAM_NUM_DEVICES = 0x1162
CL_PROGRAM_DEVICES = 0x1163
CL_PROGRAM_SOURCE = 0x1164
CL_PROGRAM_BINARY_SIZES = 0x1165
CL_PROGRAM_BINARIES = 0x1166
CL_PROGRAM_NUM_KERNELS = 0x1167
CL_PROGRAM_KERNEL_NAMES = 0x1168
CL_PROGRAM_IL = 0x1169
CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT = 0x116A
CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT = 0x116B
CL_PROGRAM_BUILD_STATUS = 0x1181
CL_PROGRAM_BUILD_OPTIONS = 0x1182
CL_PROGRAM_BUILD_LOG = 0x1183
CL_PROGRAM_BINARY_TYPE = 0x1184
CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE = 0x1185
CL_PROGRAM_BINARY_TYPE_NONE = 0x0
CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 0x1
CL_PROGRAM_BINARY_TYPE_LIBRARY = 0x2
CL_PROGRAM_BINARY_TYPE_EXECUTABLE = 0x4
CL_BUILD_SUCCESS = 0
CL_BUILD_NONE = -1
CL_BUILD_ERROR = -2
CL_BUILD_IN_PROGRESS = -3
CL_KERNEL_FUNCTION_NAME = 0x1190
CL_KERNEL_NUM_ARGS = 0x1191
CL_KERNEL_REFERENCE_COUNT = 0x1192
CL_KERNEL_CONTEXT = 0x1193
CL_KERNEL_PROGRAM = 0x1194
CL_KERNEL_ATTRIBUTES = 0x1195
CL_KERNEL_ARG_ADDRESS_QUALIFIER = 0x1196
CL_KERNEL_ARG_ACCESS_QUALIFIER = 0x1197
CL_KERNEL_ARG_TYPE_NAME = 0x1198
CL_KERNEL_ARG_TYPE_QUALIFIER = 0x1199
CL_KERNEL_ARG_NAME = 0x119A
CL_KERNEL_ARG_ADDRESS_GLOBAL = 0x119B
CL_KERNEL_ARG_ADDRESS_LOCAL = 0x119C
CL_KERNEL_ARG_ADDRESS_CONSTANT = 0x119D
CL_KERNEL_ARG_ADDRESS_PRIVATE = 0x119E
CL_KERNEL_ARG_ACCESS_READ_ONLY = 0x11A0
CL_KERNEL_ARG_ACCESS_WRITE_ONLY = 0x11A1
CL_KERNEL_ARG_ACCESS_READ_WRITE = 0x11A2
CL_KERNEL_ARG_ACCESS_NONE = 0x11A3
CL_KERNEL_ARG_TYPE_NONE = 0
CL_KERNEL_ARG_TYPE_CONST = (1 << 0)
CL_KERNEL_ARG_TYPE_RESTRICT = (1 << 1)
CL_KERNEL_ARG_TYPE_VOLATILE = (1 << 2)
CL_KERNEL_ARG_TYPE_PIPE = (1 << 3)
CL_KERNEL_WORK_GROUP_SIZE = 0x11B0
CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1
CL_KERNEL_LOCAL_MEM_SIZE = 0x11B2
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3
CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4
CL_KERNEL_GLOBAL_WORK_SIZE = 0x11B5
CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE = 0x2033
CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE = 0x2034
CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT = 0x11B8
CL_KERNEL_MAX_NUM_SUB_GROUPS = 0x11B9
CL_KERNEL_COMPILE_NUM_SUB_GROUPS = 0x11BA
CL_KERNEL_EXEC_INFO_SVM_PTRS = 0x11B6
CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM = 0x11B7
CL_EVENT_COMMAND_QUEUE = 0x11D0
CL_EVENT_COMMAND_TYPE = 0x11D1
CL_EVENT_REFERENCE_COUNT = 0x11D2
CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11D3
CL_EVENT_CONTEXT = 0x11D4
CL_COMMAND_NDRANGE_KERNEL = 0x11F0
CL_COMMAND_TASK = 0x11F1
CL_COMMAND_NATIVE_KERNEL = 0x11F2
CL_COMMAND_READ_BUFFER = 0x11F3
CL_COMMAND_WRITE_BUFFER = 0x11F4
CL_COMMAND_COPY_BUFFER = 0x11F5
CL_COMMAND_READ_IMAGE = 0x11F6
CL_COMMAND_WRITE_IMAGE = 0x11F7
CL_COMMAND_COPY_IMAGE = 0x11F8
CL_COMMAND_COPY_IMAGE_TO_BUFFER = 0x11F9
CL_COMMAND_COPY_BUFFER_TO_IMAGE = 0x11FA
CL_COMMAND_MAP_BUFFER = 0x11FB
CL_COMMAND_MAP_IMAGE = 0x11FC
CL_COMMAND_UNMAP_MEM_OBJECT = 0x11FD
CL_COMMAND_MARKER = 0x11FE
CL_COMMAND_ACQUIRE_GL_OBJECTS = 0x11FF
CL_COMMAND_RELEASE_GL_OBJECTS = 0x1200
CL_COMMAND_READ_BUFFER_RECT = 0x1201
CL_COMMAND_WRITE_BUFFER_RECT = 0x1202
CL_COMMAND_COPY_BUFFER_RECT = 0x1203
CL_COMMAND_USER = 0x1204
CL_COMMAND_BARRIER = 0x1205
CL_COMMAND_MIGRATE_MEM_OBJECTS = 0x1206
CL_COMMAND_FILL_BUFFER = 0x1207
CL_COMMAND_FILL_IMAGE = 0x1208
CL_COMMAND_SVM_FREE = 0x1209
CL_COMMAND_SVM_MEMCPY = 0x120A
CL_COMMAND_SVM_MEMFILL = 0x120B
CL_COMMAND_SVM_MAP = 0x120C
CL_COMMAND_SVM_UNMAP = 0x120D
CL_COMMAND_SVM_MIGRATE_MEM = 0x120E
CL_COMPLETE = 0x0
CL_RUNNING = 0x1
CL_SUBMITTED = 0x2
CL_QUEUED = 0x3
CL_BUFFER_CREATE_TYPE_REGION = 0x1220
CL_PROFILING_COMMAND_QUEUED = 0x1280
CL_PROFILING_COMMAND_SUBMIT = 0x1281
CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283
CL_PROFILING_COMMAND_COMPLETE = 0x1284
CL_DEVICE_ATOMIC_ORDER_RELAXED = (1 << 0)
CL_DEVICE_ATOMIC_ORDER_ACQ_REL = (1 << 1)
CL_DEVICE_ATOMIC_ORDER_SEQ_CST = (1 << 2)
CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM = (1 << 3)
CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP = (1 << 4)
CL_DEVICE_ATOMIC_SCOPE_DEVICE = (1 << 5)
CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES = (1 << 6)
CL_DEVICE_QUEUE_SUPPORTED = (1 << 0)
CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT = (1 << 1)
CL_KHRONOS_VENDOR_ID_CODEPLAY = 0x10004
CL_VERSION_MAJOR_BITS = (10)
CL_VERSION_MINOR_BITS = (10)
CL_VERSION_PATCH_BITS = (12)
CL_VERSION_MAJOR_MASK = ((1 << CL_VERSION_MAJOR_BITS) - 1)
CL_VERSION_MINOR_MASK = ((1 << CL_VERSION_MINOR_BITS) - 1)
CL_VERSION_PATCH_MASK = ((1 << CL_VERSION_PATCH_BITS) - 1)
CL_VERSION_MAJOR = lambda version: ((version) >> (CL_VERSION_MINOR_BITS + CL_VERSION_PATCH_BITS))
CL_VERSION_MINOR = lambda version: (((version) >> CL_VERSION_PATCH_BITS) & CL_VERSION_MINOR_MASK)
CL_VERSION_PATCH = lambda version: ((version) & CL_VERSION_PATCH_MASK)
CL_MAKE_VERSION = lambda major,minor,patch: ((((major) & CL_VERSION_MAJOR_MASK) << (CL_VERSION_MINOR_BITS + CL_VERSION_PATCH_BITS)) | (((minor) & CL_VERSION_MINOR_MASK) << CL_VERSION_PATCH_BITS) | ((patch) & CL_VERSION_PATCH_MASK))