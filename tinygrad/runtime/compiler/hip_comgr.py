import ctypes
import gpuctypes.comgr as comgr
from tinygrad.helpers import init_c_var

def check(status):
  if status != 0:
    comgr.amd_comgr_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"comgr fail {status}, {ctypes.string_at(status_str).decode()}")

def _get_comgr_data(data_set, data_type):
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz := ctypes.c_uint64()), None))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz), (dat := ctypes.create_string_buffer(sz.value))))
  return bytes(dat)

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip(prg:str, arch="gfx1100") -> bytes:
  rprg = prg.encode()
  action_info = init_c_var(comgr.amd_comgr_action_info_t(), lambda x: check(comgr.amd_comgr_create_action_info(ctypes.byref(x))))
  data_src = init_c_var(comgr.amd_comgr_data_t(), lambda x: check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(x))))
  data_set_src = init_c_var(comgr.amd_comgr_data_set_t(), lambda x: check(comgr.amd_comgr_create_data_set(ctypes.byref(x))))
  data_set_bc = init_c_var(comgr.amd_comgr_data_set_t(), lambda x: check(comgr.amd_comgr_create_data_set(ctypes.byref(x))))
  data_set_reloc = init_c_var(comgr.amd_comgr_data_set_t(), lambda x: check(comgr.amd_comgr_create_data_set(ctypes.byref(x))))
  data_set_exec = init_c_var(comgr.amd_comgr_data_set_t(), lambda x: check(comgr.amd_comgr_create_data_set(ctypes.byref(x))))
  check(comgr.amd_comgr_set_data(data_src, len(rprg), rprg))
  check(comgr.amd_comgr_set_data_name(data_src, b"<null>"))
  check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
  check(comgr.amd_comgr_action_info_set_language(action_info, comgr.AMD_COMGR_LANGUAGE_HIP))
  check(comgr.amd_comgr_action_info_set_isa_name(action_info, b"amdgcn-amd-amdhsa--" + arch.encode()))
  check(comgr.amd_comgr_action_info_set_logging(action_info, True))
  # -include hiprtc_runtime.h was removed
  check(comgr.amd_comgr_action_info_set_options(action_info, b"-O3 -mcumode --hip-version=6.0.32830 -DHIP_VERSION_MAJOR=6 -DHIP_VERSION_MINOR=0 -DHIP_VERSION_PATCH=32830 -D__HIPCC_RTC__ -std=c++14 -nogpuinc -Wno-gnu-line-marker -Wno-missing-prototypes --offload-arch=gfx1100 -I/opt/rocm/include -Xclang -disable-llvm-passes")) # noqa: E501
  status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action_info, data_set_src, data_set_bc)
  if status != 0:
    print(_get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
    raise RuntimeError("compile failed")
  check(comgr.amd_comgr_action_info_set_options(action_info, b"-O3 -mllvm -amdgpu-internalize-symbols"))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action_info, data_set_bc, data_set_reloc))
  check(comgr.amd_comgr_action_info_set_options(action_info, b""))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  return _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
