from tinygrad.device import CompileError
from tinygrad.runtime.autogen import intel_ocloc as ocloc
import ctypes

class IntelOfflineCompiler:
  def compile(self, cl_kernel:str, gpu_arch:str) -> bytes:
    # prepare ocoloc paramter and cast to proper ctype object for function call
    cl_kernel_bytes = (cl_kernel + "\0").encode('utf-8')
    ocloc_arguments = [b"ocloc", b"compile", b"-file", b"kernel.cl", b"-device", gpu_arch.encode('utf-8'), b"-o", b"kernel.bin"]
    ocloc_arguments_param = ctypes.cast((ctypes.c_char_p * (len(ocloc_arguments)))(*ocloc_arguments),(ctypes.POINTER(ctypes.POINTER(ctypes.c_char))))
    cl_kernel_param = (ctypes.cast((ctypes.c_ubyte * (len(cl_kernel_bytes)))(*cl_kernel_bytes), (ctypes.POINTER((ctypes.c_ubyte)))))
    cl_kernel_len_param = ctypes.byref(ctypes.c_uint64(len(cl_kernel_bytes)))
    provided_kernels_param = ctypes.cast((ctypes.c_char_p * 1)(b"kernel.cl"), (ctypes.POINTER(ctypes.POINTER(ctypes.c_char))))
    # create output paramters
    num_outputs_param = ctypes.c_uint32(0)
    data_outputs_param = ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))()
    len_outputs_param = ctypes.POINTER(ctypes.c_uint64)()
    name_outputs_param = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))()
    # compile and check result/output
    ocloc_retcode = ocloc.oclocInvoke(ctypes.c_uint32(len(ocloc_arguments)), ocloc_arguments_param, ctypes.c_uint32(1),
                            cl_kernel_param, cl_kernel_len_param, provided_kernels_param,
                            0, None, None, None, ctypes.byref(num_outputs_param), ctypes.byref(data_outputs_param),
                            ctypes.byref(len_outputs_param), ctypes.byref(name_outputs_param))
    if ocloc_retcode != ocloc.OCLOC_SUCCESS:
      raise CompileError(f"Intel OpenCL Offline Compiler (ocloc) Error\n\n{ocloc._ocloc_error_t__enumvalues[ocloc_retcode]}")
    binary = bytes(ctypes.string_at(data_outputs_param[0], len_outputs_param[0]))
    # free memory which was internally allocated for output buffers
    ocloc_retcode = ocloc.oclocFreeOutput(ctypes.byref(num_outputs_param), ctypes.byref(data_outputs_param),
                                          ctypes.byref(len_outputs_param), ctypes.byref(name_outputs_param))
    if ocloc_retcode != ocloc.OCLOC_SUCCESS:
      raise CompileError(f"Intel OpenCL Offline Compiler (ocloc) Error\n\n{ocloc._ocloc_error_t__enumvalues[ocloc_retcode]}")
    return binary