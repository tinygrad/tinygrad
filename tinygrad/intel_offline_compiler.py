from tinygrad.runtime.autogen import intel_ocloc as ocloc
import ctypes

"""
gpu_arch: xe-hpc(Max 1100), acm-g10(A770) (for more information see ocloc compile --help)

@brief: TODO
"""
class IntelOfflineCompiler:
  def __init__(self):
   pass
  def compile(self, cl_kernel:str, gpu_arch:str) -> bytes:
    # prepare ocoloc paramter and cast to proper ctype object for function call
    ocloc_arguments = [b"ocloc", b"compile", b"-file", b"kernel.cl", b"-device", gpu_arch.encode('utf-8'), b"-o", b"kernel.bin"]
    ocloc_arguments_carray = (ctypes.c_char_p * (len(ocloc_arguments)))(*ocloc_arguments)
    ocloc_arguments_len_param = ctypes.c_uint32(len(ocloc_arguments))
    ocloc_arguments_param = ctypes.cast(ocloc_arguments_carray, (ctypes.POINTER(ctypes.POINTER(ctypes.c_char))))
    # create byte array for (in-memory) kernel source code which is directly provided to ocloc lib
    cl_kernel_bytes = (cl_kernel + "\0").encode('utf-8')
    cl_kernel_carray = (ctypes.c_ubyte * (len(cl_kernel_bytes)))(*cl_kernel_bytes)
    cl_kernel_param = (ctypes.cast(cl_kernel_carray, (ctypes.POINTER((ctypes.c_ubyte)))))
    cl_kernel_len = ctypes.c_uint64(len(cl_kernel_bytes))
    cl_kernel_len_param = ctypes.byref(cl_kernel_len)
    # create array with all provided (in-memory) kernel files which are matched with in args provided name (see -file kernel.cl)
    # provided_kernels_param shall match kernel.cl which is provided in ocloc_arguments (it's because libocloc api)
    provided_kernels_carray = (ctypes.c_char_p * 1)(b"kernel.cl")
    provided_kernels_param = ctypes.cast(provided_kernels_carray, (ctypes.POINTER(ctypes.POINTER(ctypes.c_char))))
    # create output paramters
    num_outputs_param = ctypes.c_uint32(0)
    data_outputs_param = ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))()
    len_outputs_param = ctypes.POINTER(ctypes.c_uint64)()
    name_outputs_param = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))()
    # compile and check result/output
    ocloc_retcode = ocloc.oclocInvoke(ocloc_arguments_len_param, ocloc_arguments_param, ctypes.c_uint32(1),
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
      print("Error: ocloc freeing memory failed!")
    return binary