import ctypes
import tinygrad.runtime.autogen.comgr as comgr
from tinygrad.helpers import DEBUG

def check(status):
  if status != 0:
    comgr.amd_comgr_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"comgr fail {status}, {ctypes.string_at(status_str).decode()}")

def _get_comgr_data(data_set, data_type):
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz := ctypes.c_uint64()), None))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz), (dat := ctypes.create_string_buffer(sz.value))))
  check(comgr.amd_comgr_release_data(data_exec))
  return bytes(dat)
def _get_comgr_data_metadata(data_set, data_type):
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data_metadata(data_exec, ctypes.byref(data_metadata := comgr.amd_comgr_metadata_node_t())))

  metadata_keys = []
  @ctypes.CFUNCTYPE(comgr.amd_comgr_status_t, comgr.amd_comgr_metadata_node_t, comgr.amd_comgr_metadata_node_t, ctypes.c_void_p)
  def _map_callback(k, v, _):
    check(comgr.amd_comgr_get_metadata_string(k, ctypes.byref(sz := ctypes.c_uint64()), None))
    check(comgr.amd_comgr_get_metadata_string(k, ctypes.byref(sz), (kstr := ctypes.create_string_buffer(sz.value))))
    metadata_keys.append(kstr)
    return comgr.AMD_COMGR_STATUS_SUCCESS

  def walk(node):
    check(comgr.amd_comgr_get_metadata_kind(node, ctypes.byref(kind := comgr.amd_comgr_metadata_kind_t())))
    if kind.value == comgr.AMD_COMGR_METADATA_KIND_MAP:
      metadata_keys.clear()
      check(comgr.amd_comgr_iterate_map_metadata(node, _map_callback, ctypes.c_void_p()))
      ret = {}
      # need to copy, since the global metadata_keys buffer can change in recursion
      for k in metadata_keys[:]:
        check(comgr.amd_comgr_metadata_lookup(node, k, ctypes.byref(map_v := comgr.amd_comgr_metadata_node_t())))
        ret[k.value.decode()] = walk(map_v)
        check(comgr.amd_comgr_destroy_metadata(map_v))
    elif kind.value == comgr.AMD_COMGR_METADATA_KIND_LIST:
      check(comgr.amd_comgr_get_metadata_list_size(node, ctypes.byref(list_sz := ctypes.c_uint64())))
      ret = []
      for i in range(list_sz.value):
        check(comgr.amd_comgr_index_list_metadata(node, ctypes.c_size_t(i), ctypes.byref(list_v := comgr.amd_comgr_metadata_node_t())))
        ret.append(walk(list_v))
        check(comgr.amd_comgr_destroy_metadata(list_v))
    elif kind.value == comgr.AMD_COMGR_METADATA_KIND_STRING:
      check(comgr.amd_comgr_get_metadata_string(node, ctypes.byref(sz := ctypes.c_uint64()), None))
      check(comgr.amd_comgr_get_metadata_string(node, ctypes.byref(sz), (rstr := ctypes.create_string_buffer(sz.value))))
      ret = rstr.value.decode()
    else:
      assert False
    return ret
  ret = walk(data_metadata)

  check(comgr.amd_comgr_destroy_metadata(data_metadata))
  check(comgr.amd_comgr_release_data(data_exec))
  return ret

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip(prg:str, arch="gfx1100") -> bytes:
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action_info := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_language(action_info, comgr.AMD_COMGR_LANGUAGE_HIP))
  check(comgr.amd_comgr_action_info_set_isa_name(action_info, b"amdgcn-amd-amdhsa--" + arch.encode()))
  check(comgr.amd_comgr_action_info_set_logging(action_info, True))

  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_src := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_bc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_asm := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_reloc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_exec := comgr.amd_comgr_data_set_t())))

  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(data_src := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_set_data(data_src, len(rprg := prg.encode()), rprg))
  check(comgr.amd_comgr_set_data_name(data_src, b"<null>"))

  check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
  # -include hiprtc_runtime.h was removed
  check(comgr.amd_comgr_action_info_set_options(action_info, f"-O3 -mcumode --hip-version=6.0.32830 -DHIP_VERSION_MAJOR=6 -DHIP_VERSION_MINOR=0 -DHIP_VERSION_PATCH=32830 -D__HIPCC_RTC__ -std=c++14 -nogpuinc -Wno-gnu-line-marker -Wno-missing-prototypes --offload-arch={arch} -I/opt/rocm/include -Xclang -disable-llvm-passes".encode())) # noqa: E501
  status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action_info, data_set_src, data_set_bc)
  if status != 0:
    print(_get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
    raise RuntimeError("compile failed")
  check(comgr.amd_comgr_action_info_set_options(action_info, b"-O3 -mllvm -amdgpu-internalize-symbols"))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY, action_info, data_set_bc, data_set_asm))
  if DEBUG >= 6:
    # there is more info in the assembly than in the metaata
    asm = _get_comgr_data(data_set_asm, comgr.AMD_COMGR_DATA_KIND_SOURCE)
    print(asm.decode())
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, action_info, data_set_asm, data_set_reloc))
  check(comgr.amd_comgr_action_info_set_options(action_info, b""))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  metadata = _get_comgr_data_metadata(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  if DEBUG >= 6:
    import json
    print(json.dumps(metadata, indent=2))
  sgpr_spill, vgpr_spill = metadata["amdhsa.kernels"][0][".sgpr_spill_count"], metadata["amdhsa.kernels"][0][".vgpr_spill_count"]
  assert sgpr_spill == "0" and vgpr_spill == "0", f"spilled registers: s{sgpr_spill} v{vgpr_spill}"
  ret = _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  check(comgr.amd_comgr_release_data(data_src))
  for x in [data_set_src, data_set_bc, data_set_asm, data_set_reloc, data_set_exec]: check(comgr.amd_comgr_destroy_data_set(x))
  check(comgr.amd_comgr_destroy_action_info(action_info))
  return ret
