import subprocess
import ctypes
from tinygrad.helpers import DEBUG
import sys
import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

try:
  def cu_get_include_paths(compiler):
    result = subprocess.check_output([compiler, "-E", "-x", "c", "-", "-v"], input="", stderr=subprocess.STDOUT, universal_newlines=True)
    lines = result.splitlines()

    includes = []
    for line in lines:
      if line.startswith("#$ INCLUDES="):
        line = line.strip().rstrip()[len("#$ INCLUDES=\""):-1]
        includes = line.split()
    return includes

  _libcuda = ctypes.cdll.LoadLibrary("libcuda.so.1")
  _libnvrtc = ctypes.cdll.LoadLibrary("libnvrtc.so")
  _nvrtc_includes = cu_get_include_paths(compiler="nvcc")

  _libcuda.cuGetErrorString.restype = ctypes.c_int
  _libcuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.c_void_p]
  def cuGetErrorString(status):
    ptr = ctypes.c_char_p()
    status = _libcuda.cuGetErrorString(status, ctypes.byref(ptr))
    if status != 0: raise RuntimeError("CUDA error: cuGetErrorString failed")
    return ptr.value.decode("utf-8")

  def cuCheckStatus(status):
    if status != 0: raise RuntimeError("CUDA error %s: %s" % (status, cuGetErrorString(status)))

  _libcuda.cuInit.restype = int
  _libcuda.cuInit.argtypes = [ctypes.c_uint]
  def cuInit(flags):
    status = _libcuda.cuInit(flags)
    cuCheckStatus(status)

  _libcuda.cuCtxCreate_v2.restype = int
  _libcuda.cuCtxCreate_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
  def cuCtxCreate(flags, device):
    ptr = ctypes.c_void_p()
    status = _libcuda.cuCtxCreate_v2(ctypes.byref(ptr), flags, device)
    cuCheckStatus(status)
    return ptr

  _libcuda.cuCtxSynchronize.restype = int
  _libcuda.cuCtxSynchronize.argtypes = []
  def cuCtxSynchronize():
    status = _libcuda.cuCtxSynchronize()
    cuCheckStatus(status)

  _libcuda.cuStreamSynchronize.restype = int
  _libcuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
  def cuStreamSynchronize(stream):
    status = _libcuda.cuStreamSynchronize(stream)
    cuCheckStatus(status)

  _libcuda.cuEventCreate.restype = int
  _libcuda.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
  def cuEventCreate(flags=0):
    ptr = ctypes.c_void_p()
    status = _libcuda.cuEventCreate(ctypes.byref(ptr), flags)
    cuCheckStatus(status)
    return ptr

  _libcuda.cuEventRecord.restype = int
  _libcuda.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  def cuEventRecord(event, stream=None):
    status = _libcuda.cuEventRecord(event, stream)
    cuCheckStatus(status)

  _libcuda.cuEventDestroy.restype = int
  _libcuda.cuEventDestroy.argtypes = [ctypes.c_void_p]
  def cuEventDestroy(event):
    status = _libcuda.cuEventDestroy(event)
    cuCheckStatus(status)

  _libcuda.cuEventSynchronize.restype = int
  _libcuda.cuEventSynchronize.argtypes = [ctypes.c_void_p]
  def cuEventSynchronize(event):
    status = _libcuda.cuEventSynchronize(event)
    cuCheckStatus(status)

  _libcuda.cuEventElapsedTime.restype = int
  _libcuda.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
  def cuEventElapsedTime(start, stop):
    t = ctypes.c_float()
    status = _libcuda.cuEventElapsedTime(ctypes.byref(t), start, stop)
    cuCheckStatus(status)
    return t.value

  ## Stream Management

  _libcuda.cuStreamCreate.restype = int
  _libcuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
  def cuStreamCreate():
    ptr = ctypes.c_void_p()
    status = _libcuda.cuStreamCreate(ctypes.byref(ptr))
    cuCheckStatus(status)
    return ptr

  _libcuda.cuStreamDestroy.restype = int
  _libcuda.cuStreamDestroy.argtypes = [ctypes.c_void_p]
  def cuStreamDestroy(stream):
    status = _libcuda.cuStreamDestroy(stream)
    cuCheckStatus(status)

  ## Graph Management

  _libcuda.cuGraphCreate.restype = int
  _libcuda.cuGraphCreate.argtypes = [ctypes.c_void_p, ctypes.c_uint]
  def cuGraphCreate():
    ptr = ctypes.c_void_p()
    status = _libcuda.cuGraphCreate(ctypes.byref(ptr), 0)
    cuCheckStatus(status)
    return ptr

  _libcuda.cuGraphInstantiate.restype = int
  _libcuda.cuGraphInstantiate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  def cuGraphInstantiate(graph):
    ptr = ctypes.c_void_p()
    status = _libcuda.cuGraphInstantiate(ctypes.byref(ptr), graph, 0, 0, 0)
    cuCheckStatus(status)
    return ptr

  _libcuda.cuGraphDestroy.restype = int
  _libcuda.cuGraphDestroy.argtypes = [ctypes.c_void_p]
  def cuGraphDestroy(graph):
    status = _libcuda.cuGraphDestroy(graph)
    cuCheckStatus(status)

  _libcuda.cuGraphExecDestroy.restype = int
  _libcuda.cuGraphExecDestroy.argtypes = [ctypes.c_void_p]
  def cuGraphExecDestroy(gexec):
    status = _libcuda.cuGraphExecDestroy(gexec)
    cuCheckStatus(status)

  _libcuda.cuGraphLaunch.restype = int
  _libcuda.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  def cuGraphLaunch(graph_exec, stream=0):
    status = _libcuda.cuGraphLaunch(graph_exec, stream)
    cuCheckStatus(status)

  class hipKernelNodeParams(ctypes.Structure):
    _fields_ = [("blockDimX", ctypes.c_uint32), ("blockDimY", ctypes.c_uint32), ("blockDimZ", ctypes.c_uint32),
                ("extra", ctypes.POINTER(ctypes.c_void_p)),
                ("func", ctypes.c_void_p),
                ("gridDimX", ctypes.c_uint32), ("gridDimY", ctypes.c_uint32), ("gridDimZ", ctypes.c_uint32),
                ("kernelParams", ctypes.POINTER(ctypes.c_void_p)),
                ("sharedMemBytes", ctypes.c_uint32)]

  @dataclass
  class kernelNodeParamsWrapper():
    c_struct: Any
    context: Any = None

  def getCStructForType(argtypes):
    fields = []
    for j,typ in enumerate(argtypes):
      fields.append((f'field{j}', typ))

    class CStructure(ctypes.Structure):
      _pack_ = 1
      _fields_ = fields
    return CStructure

  def setKernelNodeLaunchDims(npwrapper:kernelNodeParamsWrapper, grid, block):
    npwrapper.c_struct.blockDimX = block[0]
    npwrapper.c_struct.blockDimY = block[1]
    npwrapper.c_struct.blockDimZ = block[2]
    npwrapper.c_struct.gridDimX = grid[0]
    npwrapper.c_struct.gridDimY = grid[1]
    npwrapper.c_struct.gridDimZ = grid[2]

  def setKernelNodeParams(npwrapper:kernelNodeParamsWrapper, args, ids):
    for j,i in enumerate(ids):
      setattr(npwrapper.context[1], f'field{i}', args[j])

  def buildKernelNodeParams(args, argtypes, func, grid, block, sharedMemBytes=0):
    c_struct_t = getCStructForType(argtypes)
    struct = c_struct_t(*args)
    size = ctypes.c_size_t(ctypes.sizeof(struct))
    p_size = ctypes.c_void_p(ctypes.addressof(size))
    p_struct = ctypes.c_void_p(ctypes.addressof(struct))
    config = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), p_struct,
                                  ctypes.c_void_p(2), p_size, ctypes.c_void_p(3))
    params = hipKernelNodeParams(block[0], block[1], block[2], config, func, grid[0], grid[1], grid[2], None, sharedMemBytes)
    return kernelNodeParamsWrapper(c_struct=params, context=(size, struct, config))

  _libcuda.cuGraphAddKernelNode.restype = int
  _libcuda.cuGraphAddKernelNode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
  def cuGraphAddKernelNode(graph, deps, params:kernelNodeParamsWrapper):
    graph_node = ctypes.c_void_p()
    deps_in = (ctypes.c_void_p * len(deps))()
    deps_in[:] = deps
    num_deps = ctypes.c_size_t(len(deps))
    status = _libcuda.cuGraphAddKernelNode(ctypes.byref(graph_node), graph, deps_in, num_deps, ctypes.byref(params.c_struct))
    cuCheckStatus(status)
    return graph_node

  _libcuda.cuGraphExecKernelNodeSetParams.restype = int
  _libcuda.cuGraphExecKernelNodeSetParams.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
  def cuGraphExecKernelNodeSetParams(gexec, node, params:kernelNodeParamsWrapper):
    status = _libcuda.cuGraphExecKernelNodeSetParams(gexec, node, ctypes.byref(params.c_struct))
    cuCheckStatus(status)

  _libcuda.cuMemAlloc_v2.restype = int
  _libcuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
  def cuMemAlloc(count):
    ptr = ctypes.c_void_p()
    status = _libcuda.cuMemAlloc_v2(ctypes.byref(ptr), count)
    cuCheckStatus(status)
    return ptr.value

  _libcuda.cuMemFree_v2.restype = int
  _libcuda.cuMemFree_v2.argtypes = [ctypes.c_void_p]
  def cuMemFree(ptr):
    status = _libcuda.cuMemFree_v2(ptr)
    cuCheckStatus(status)

  _libcuda.cuMemcpyHtoDAsync_v2.restype = int
  _libcuda.cuMemcpyHtoDAsync_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
  def cuMemcpyHtoDAsync(dst, src, count, stream):
    status = _libcuda.cuMemcpyHtoDAsync_v2(dst, src, ctypes.c_size_t(count), stream)
    cuCheckStatus(status)

  _libcuda.cuMemcpyDtoH_v2.restype = int
  _libcuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
  def cuMemcpyDtoH(dst, src, count):
    status = _libcuda.cuMemcpyDtoH_v2(dst, src, ctypes.c_size_t(count))
    cuCheckStatus(status)

  _libcuda.cuMemGetInfo_v2.restype = int
  _libcuda.cuMemGetInfo_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
  def cuMemGetInfo():
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _libcuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
    cuCheckStatus(status)
    return free.value, total.value


  ## Device Management

  _libcuda.cuDeviceGet.restype = int
  _libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
  def cuDeviceGet(id):
    dev = ctypes.c_int()
    status = _libcuda.cuDeviceGet(ctypes.byref(dev), id)
    cuCheckStatus(status)
    return dev.value

  _libcuda.cuDeviceGetCount.restype = int
  _libcuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
  def cuDeviceGetCount():
    count = ctypes.c_int()
    status = _libcuda.cuDeviceGetCount(ctypes.byref(count))
    cuCheckStatus(status)
    return count.value

  _libcuda.cuDeviceComputeCapability.restype = int
  _libcuda.cuDeviceComputeCapability.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
  def cuDeviceComputeCapability(device_id):
    major = ctypes.c_int()
    minor = ctypes.c_int()
    status = _libcuda.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device_id)
    cuCheckStatus(status)
    return (major.value, minor.value)

  _libcuda.cuModuleLoadData.restype = int
  _libcuda.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
  def cuModuleLoadData(data):
    module = ctypes.c_void_p()
    data_ptr = np.char.array(data).ctypes.data
    status = _libcuda.cuModuleLoadData(ctypes.byref(module), data_ptr)
    cuCheckStatus(status)
    return module

  _libcuda.cuModuleGetFunction.restype = int
  _libcuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
  def cuModuleGetFunction(module, func_name):
    kernel = ctypes.c_void_p()
    status = _libcuda.cuModuleGetFunction(ctypes.byref(kernel), module, func_name.encode("utf-8"))
    cuCheckStatus(status)
    return kernel

  _libcuda.cuModuleUnload.restype = int
  _libcuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
  def cuModuleUnload(module):
    status = _libcuda.cuModuleUnload(module)
    cuCheckStatus(status)

  _libcuda.cuLaunchKernel.restype = int
  _libcuda.cuLaunchKernel.argtypes = [ctypes.c_void_p,
                                            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # block dim
                                            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, # thread dim
                                            ctypes.c_uint, ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
  def cuLaunchKernel(kernel, bx, by, bz, tx, ty, tz, shared, stream, struct):
    c_bx, c_by, c_bz = ctypes.c_uint(bx), ctypes.c_uint(by), ctypes.c_uint(bz)
    c_tx, c_ty, c_tz = ctypes.c_uint(tx), ctypes.c_uint(ty), ctypes.c_uint(tz)
    c_shared = ctypes.c_uint(shared)

    param_buffer_ptr, param_buffer_size, param_buffer_end = ctypes.c_void_p(1), ctypes.c_void_p(2), ctypes.c_void_p(0)
    size = ctypes.c_size_t(ctypes.sizeof(struct))
    p_size, p_struct = ctypes.c_void_p(ctypes.addressof(size)), ctypes.c_void_p(ctypes.addressof(struct))
    config = (ctypes.c_void_p * 5)(param_buffer_ptr, p_struct, param_buffer_size, p_size, param_buffer_end)

    status = _libcuda.cuLaunchKernel(kernel, c_bx, c_by, c_bz, c_tx, c_ty, c_tz, c_shared, stream, None, config)
    cuCheckStatus(status)

  _libnvrtc.nvrtcCreateProgram.restype = int
  _libnvrtc.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char),
                                              ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]
  def nvrtcCreateProgram(source, name, header_names, header_sources):
    c_header_names, c_header_sources = (ctypes.c_char_p * len(header_names))(), (ctypes.c_char_p * len(header_sources))()
    c_header_names[:], c_header_sources[:] = [h.encode("utf-8") for h in header_names], [h.encode("utf-8") for h in header_sources]

    prog = ctypes.c_void_p()
    status = _libnvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode("utf-8"), name.encode("utf-8"), len(header_names), c_header_sources, c_header_names)
    cuCheckStatus(status)
    return prog

  _libnvrtc.nvrtcDestroyProgram.restype = int
  _libnvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
  def nvrtcDestroyProgram(prog):
    status = _libnvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    cuCheckStatus(status)

  _libnvrtc.nvrtcGetProgramLogSize.restype = int
  _libnvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
  _libnvrtc.nvrtcGetProgramLog.restype = int
  _libnvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
  def nvrtcGetProgramLog(prog):
    logsz = ctypes.c_size_t()
    status = _libnvrtc.nvrtcGetProgramLogSize(prog, logsz)
    cuCheckStatus(status)
    logstr = ctypes.create_string_buffer(logsz.value)
    status = _libnvrtc.nvrtcGetProgramLog(prog, logstr)
    cuCheckStatus(status)
    return logstr.value.decode()

  _libnvrtc.nvrtcCompileProgram.restype = int
  _libnvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
  def nvrtcCompileProgram(prog, options):
    options += _nvrtc_includes
    c_options = (ctypes.c_char_p * len(options))()
    c_options[:] = [o.encode("utf-8") for o in options]

    status = _libnvrtc.nvrtcCompileProgram(prog, len(options), c_options)
    if status == 6: print(nvrtcGetProgramLog(prog))
    cuCheckStatus(status)

  _libnvrtc.nvrtcGetPTXSize.restype = int
  _libnvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
  _libnvrtc.nvrtcGetPTX.restype = int
  _libnvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
  def nvrtcGetPTX(prog):
    code_size = ctypes.c_size_t()
    status = _libnvrtc.nvrtcGetPTXSize(prog, ctypes.byref(code_size))
    cuCheckStatus(status)
    e_code = ("0" * code_size.value).encode("utf-8")
    status = _libnvrtc.nvrtcGetPTX(prog, e_code)
    cuCheckStatus(status)
    return e_code
except:
  if DEBUG >= 1: print("WARNING: libcuda.so or libnvrtc.so not found. CUDA support will not work.")
