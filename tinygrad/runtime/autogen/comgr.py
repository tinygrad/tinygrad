# mypy: ignore-errors
import ctypes, os
from tinygrad.helpers import unwrap
from tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR
def dll():
  try: return ctypes.CDLL(unwrap(os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libamd_comgr.so'))
  except: pass
  try: return ctypes.CDLL(unwrap('/usr/local/lib/libamd_comgr.dylib'))
  except: pass
  try: return ctypes.CDLL(unwrap('/opt/homebrew/lib/libamd_comgr.dylib'))
  except: pass
  return None
dll = dll()

amd_comgr_status_s = CEnum(ctypes.c_uint32)
AMD_COMGR_STATUS_SUCCESS = amd_comgr_status_s.define('AMD_COMGR_STATUS_SUCCESS', 0)
AMD_COMGR_STATUS_ERROR = amd_comgr_status_s.define('AMD_COMGR_STATUS_ERROR', 1)
AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT = amd_comgr_status_s.define('AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT', 2)
AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES = amd_comgr_status_s.define('AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES', 3)

amd_comgr_status_t = amd_comgr_status_s
amd_comgr_language_s = CEnum(ctypes.c_uint32)
AMD_COMGR_LANGUAGE_NONE = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_NONE', 0)
AMD_COMGR_LANGUAGE_OPENCL_1_2 = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_OPENCL_1_2', 1)
AMD_COMGR_LANGUAGE_OPENCL_2_0 = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_OPENCL_2_0', 2)
AMD_COMGR_LANGUAGE_HC = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_HC', 3)
AMD_COMGR_LANGUAGE_HIP = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_HIP', 4)
AMD_COMGR_LANGUAGE_LLVM_IR = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_LLVM_IR', 5)
AMD_COMGR_LANGUAGE_LAST = amd_comgr_language_s.define('AMD_COMGR_LANGUAGE_LAST', 5)

amd_comgr_language_t = amd_comgr_language_s
# amd_comgr_status_t amd_comgr_status_string(amd_comgr_status_t status, const char **status_string)
try: (amd_comgr_status_string:=dll.amd_comgr_status_string).restype, amd_comgr_status_string.argtypes = amd_comgr_status_t, [amd_comgr_status_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

size_t = ctypes.c_uint64
# void amd_comgr_get_version(size_t *major, size_t *minor)
try: (amd_comgr_get_version:=dll.amd_comgr_get_version).restype, amd_comgr_get_version.argtypes = None, [ctypes.POINTER(size_t), ctypes.POINTER(size_t)]
except AttributeError: pass

amd_comgr_data_kind_s = CEnum(ctypes.c_uint32)
AMD_COMGR_DATA_KIND_UNDEF = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_UNDEF', 0)
AMD_COMGR_DATA_KIND_SOURCE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_SOURCE', 1)
AMD_COMGR_DATA_KIND_INCLUDE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_INCLUDE', 2)
AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER', 3)
AMD_COMGR_DATA_KIND_DIAGNOSTIC = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_DIAGNOSTIC', 4)
AMD_COMGR_DATA_KIND_LOG = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_LOG', 5)
AMD_COMGR_DATA_KIND_BC = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_BC', 6)
AMD_COMGR_DATA_KIND_RELOCATABLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_RELOCATABLE', 7)
AMD_COMGR_DATA_KIND_EXECUTABLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_EXECUTABLE', 8)
AMD_COMGR_DATA_KIND_BYTES = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_BYTES', 9)
AMD_COMGR_DATA_KIND_FATBIN = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_FATBIN', 16)
AMD_COMGR_DATA_KIND_AR = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_AR', 17)
AMD_COMGR_DATA_KIND_BC_BUNDLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_BC_BUNDLE', 18)
AMD_COMGR_DATA_KIND_AR_BUNDLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_AR_BUNDLE', 19)
AMD_COMGR_DATA_KIND_OBJ_BUNDLE = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_OBJ_BUNDLE', 20)
AMD_COMGR_DATA_KIND_LAST = amd_comgr_data_kind_s.define('AMD_COMGR_DATA_KIND_LAST', 20)

amd_comgr_data_kind_t = amd_comgr_data_kind_s
class amd_comgr_data_s(Struct): pass
uint64_t = ctypes.c_uint64
amd_comgr_data_s._fields_ = [
  ('handle', uint64_t),
]
amd_comgr_data_t = amd_comgr_data_s
class amd_comgr_data_set_s(Struct): pass
amd_comgr_data_set_s._fields_ = [
  ('handle', uint64_t),
]
amd_comgr_data_set_t = amd_comgr_data_set_s
class amd_comgr_action_info_s(Struct): pass
amd_comgr_action_info_s._fields_ = [
  ('handle', uint64_t),
]
amd_comgr_action_info_t = amd_comgr_action_info_s
class amd_comgr_metadata_node_s(Struct): pass
amd_comgr_metadata_node_s._fields_ = [
  ('handle', uint64_t),
]
amd_comgr_metadata_node_t = amd_comgr_metadata_node_s
class amd_comgr_symbol_s(Struct): pass
amd_comgr_symbol_s._fields_ = [
  ('handle', uint64_t),
]
amd_comgr_symbol_t = amd_comgr_symbol_s
class amd_comgr_disassembly_info_s(Struct): pass
amd_comgr_disassembly_info_s._fields_ = [
  ('handle', uint64_t),
]
amd_comgr_disassembly_info_t = amd_comgr_disassembly_info_s
class amd_comgr_symbolizer_info_s(Struct): pass
amd_comgr_symbolizer_info_s._fields_ = [
  ('handle', uint64_t),
]
amd_comgr_symbolizer_info_t = amd_comgr_symbolizer_info_s
# amd_comgr_status_t amd_comgr_get_isa_count(size_t *count)
try: (amd_comgr_get_isa_count:=dll.amd_comgr_get_isa_count).restype, amd_comgr_get_isa_count.argtypes = amd_comgr_status_t, [ctypes.POINTER(size_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_isa_name(size_t index, const char **isa_name)
try: (amd_comgr_get_isa_name:=dll.amd_comgr_get_isa_name).restype, amd_comgr_get_isa_name.argtypes = amd_comgr_status_t, [size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_isa_metadata(const char *isa_name, amd_comgr_metadata_node_t *metadata)
try: (amd_comgr_get_isa_metadata:=dll.amd_comgr_get_isa_metadata).restype, amd_comgr_get_isa_metadata.argtypes = amd_comgr_status_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(amd_comgr_metadata_node_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_create_data(amd_comgr_data_kind_t kind, amd_comgr_data_t *data)
try: (amd_comgr_create_data:=dll.amd_comgr_create_data).restype, amd_comgr_create_data.argtypes = amd_comgr_status_t, [amd_comgr_data_kind_t, ctypes.POINTER(amd_comgr_data_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_release_data(amd_comgr_data_t data)
try: (amd_comgr_release_data:=dll.amd_comgr_release_data).restype, amd_comgr_release_data.argtypes = amd_comgr_status_t, [amd_comgr_data_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_data_kind(amd_comgr_data_t data, amd_comgr_data_kind_t *kind)
try: (amd_comgr_get_data_kind:=dll.amd_comgr_get_data_kind).restype, amd_comgr_get_data_kind.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(amd_comgr_data_kind_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_set_data(amd_comgr_data_t data, size_t size, const char *bytes)
try: (amd_comgr_set_data:=dll.amd_comgr_set_data).restype, amd_comgr_set_data.argtypes = amd_comgr_status_t, [amd_comgr_data_t, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_set_data_from_file_slice(amd_comgr_data_t data, int file_descriptor, uint64_t offset, uint64_t size)
try: (amd_comgr_set_data_from_file_slice:=dll.amd_comgr_set_data_from_file_slice).restype, amd_comgr_set_data_from_file_slice.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.c_int32, uint64_t, uint64_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_set_data_name(amd_comgr_data_t data, const char *name)
try: (amd_comgr_set_data_name:=dll.amd_comgr_set_data_name).restype, amd_comgr_set_data_name.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_data(amd_comgr_data_t data, size_t *size, char *bytes)
try: (amd_comgr_get_data:=dll.amd_comgr_get_data).restype, amd_comgr_get_data.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_data_name(amd_comgr_data_t data, size_t *size, char *name)
try: (amd_comgr_get_data_name:=dll.amd_comgr_get_data_name).restype, amd_comgr_get_data_name.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_data_isa_name(amd_comgr_data_t data, size_t *size, char *isa_name)
try: (amd_comgr_get_data_isa_name:=dll.amd_comgr_get_data_isa_name).restype, amd_comgr_get_data_isa_name.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_create_symbolizer_info(amd_comgr_data_t code_object, void (*print_symbol_callback)(const char *, void *), amd_comgr_symbolizer_info_t *symbolizer_info)
try: (amd_comgr_create_symbolizer_info:=dll.amd_comgr_create_symbolizer_info).restype, amd_comgr_create_symbolizer_info.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p), ctypes.POINTER(amd_comgr_symbolizer_info_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_destroy_symbolizer_info(amd_comgr_symbolizer_info_t symbolizer_info)
try: (amd_comgr_destroy_symbolizer_info:=dll.amd_comgr_destroy_symbolizer_info).restype, amd_comgr_destroy_symbolizer_info.argtypes = amd_comgr_status_t, [amd_comgr_symbolizer_info_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_symbolize(amd_comgr_symbolizer_info_t symbolizer_info, uint64_t address, bool is_code, void *user_data)
try: (amd_comgr_symbolize:=dll.amd_comgr_symbolize).restype, amd_comgr_symbolize.argtypes = amd_comgr_status_t, [amd_comgr_symbolizer_info_t, uint64_t, ctypes.c_bool, ctypes.c_void_p]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_data_metadata(amd_comgr_data_t data, amd_comgr_metadata_node_t *metadata)
try: (amd_comgr_get_data_metadata:=dll.amd_comgr_get_data_metadata).restype, amd_comgr_get_data_metadata.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(amd_comgr_metadata_node_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_destroy_metadata(amd_comgr_metadata_node_t metadata)
try: (amd_comgr_destroy_metadata:=dll.amd_comgr_destroy_metadata).restype, amd_comgr_destroy_metadata.argtypes = amd_comgr_status_t, [amd_comgr_metadata_node_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_create_data_set(amd_comgr_data_set_t *data_set)
try: (amd_comgr_create_data_set:=dll.amd_comgr_create_data_set).restype, amd_comgr_create_data_set.argtypes = amd_comgr_status_t, [ctypes.POINTER(amd_comgr_data_set_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_destroy_data_set(amd_comgr_data_set_t data_set)
try: (amd_comgr_destroy_data_set:=dll.amd_comgr_destroy_data_set).restype, amd_comgr_destroy_data_set.argtypes = amd_comgr_status_t, [amd_comgr_data_set_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_data_set_add(amd_comgr_data_set_t data_set, amd_comgr_data_t data)
try: (amd_comgr_data_set_add:=dll.amd_comgr_data_set_add).restype, amd_comgr_data_set_add.argtypes = amd_comgr_status_t, [amd_comgr_data_set_t, amd_comgr_data_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_data_set_remove(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind)
try: (amd_comgr_data_set_remove:=dll.amd_comgr_data_set_remove).restype, amd_comgr_data_set_remove.argtypes = amd_comgr_status_t, [amd_comgr_data_set_t, amd_comgr_data_kind_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_data_count(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind, size_t *count)
try: (amd_comgr_action_data_count:=dll.amd_comgr_action_data_count).restype, amd_comgr_action_data_count.argtypes = amd_comgr_status_t, [amd_comgr_data_set_t, amd_comgr_data_kind_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_data_get_data(amd_comgr_data_set_t data_set, amd_comgr_data_kind_t data_kind, size_t index, amd_comgr_data_t *data)
try: (amd_comgr_action_data_get_data:=dll.amd_comgr_action_data_get_data).restype, amd_comgr_action_data_get_data.argtypes = amd_comgr_status_t, [amd_comgr_data_set_t, amd_comgr_data_kind_t, size_t, ctypes.POINTER(amd_comgr_data_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_create_action_info(amd_comgr_action_info_t *action_info)
try: (amd_comgr_create_action_info:=dll.amd_comgr_create_action_info).restype, amd_comgr_create_action_info.argtypes = amd_comgr_status_t, [ctypes.POINTER(amd_comgr_action_info_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_destroy_action_info(amd_comgr_action_info_t action_info)
try: (amd_comgr_destroy_action_info:=dll.amd_comgr_destroy_action_info).restype, amd_comgr_destroy_action_info.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_set_isa_name(amd_comgr_action_info_t action_info, const char *isa_name)
try: (amd_comgr_action_info_set_isa_name:=dll.amd_comgr_action_info_set_isa_name).restype, amd_comgr_action_info_set_isa_name.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_get_isa_name(amd_comgr_action_info_t action_info, size_t *size, char *isa_name)
try: (amd_comgr_action_info_get_isa_name:=dll.amd_comgr_action_info_get_isa_name).restype, amd_comgr_action_info_get_isa_name.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_set_language(amd_comgr_action_info_t action_info, amd_comgr_language_t language)
try: (amd_comgr_action_info_set_language:=dll.amd_comgr_action_info_set_language).restype, amd_comgr_action_info_set_language.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, amd_comgr_language_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_get_language(amd_comgr_action_info_t action_info, amd_comgr_language_t *language)
try: (amd_comgr_action_info_get_language:=dll.amd_comgr_action_info_get_language).restype, amd_comgr_action_info_get_language.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(amd_comgr_language_t)]
except AttributeError: pass

# __attribute__((deprecated("Will be removed in Comgr v3.0 (Rocm v6.0). Use amd_comgr_action_info_set_option_list() instead"))) amd_comgr_status_t amd_comgr_action_info_set_options(amd_comgr_action_info_t action_info, const char *options)
try: (amd_comgr_action_info_set_options:=dll.amd_comgr_action_info_set_options).restype, amd_comgr_action_info_set_options.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# __attribute__((deprecated("Will be removed in Comgr v3.0 (Rocm v6.0). Use amd_comgr_action_info_get_option_list_count() and amd_comgr_action_info_get_option_list_item() instead"))) amd_comgr_status_t amd_comgr_action_info_get_options(amd_comgr_action_info_t action_info, size_t *size, char *options)
try: (amd_comgr_action_info_get_options:=dll.amd_comgr_action_info_get_options).restype, amd_comgr_action_info_get_options.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_set_option_list(amd_comgr_action_info_t action_info, const char *options[], size_t count)
try: (amd_comgr_action_info_set_option_list:=dll.amd_comgr_action_info_set_option_list).restype, amd_comgr_action_info_set_option_list.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, (ctypes.POINTER(ctypes.c_char) * 0), size_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_get_option_list_count(amd_comgr_action_info_t action_info, size_t *count)
try: (amd_comgr_action_info_get_option_list_count:=dll.amd_comgr_action_info_get_option_list_count).restype, amd_comgr_action_info_get_option_list_count.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_get_option_list_item(amd_comgr_action_info_t action_info, size_t index, size_t *size, char *option)
try: (amd_comgr_action_info_get_option_list_item:=dll.amd_comgr_action_info_get_option_list_item).restype, amd_comgr_action_info_get_option_list_item.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, size_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_set_bundle_entry_ids(amd_comgr_action_info_t action_info, const char *bundle_entry_ids[], size_t count)
try: (amd_comgr_action_info_set_bundle_entry_ids:=dll.amd_comgr_action_info_set_bundle_entry_ids).restype, amd_comgr_action_info_set_bundle_entry_ids.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, (ctypes.POINTER(ctypes.c_char) * 0), size_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_get_bundle_entry_id_count(amd_comgr_action_info_t action_info, size_t *count)
try: (amd_comgr_action_info_get_bundle_entry_id_count:=dll.amd_comgr_action_info_get_bundle_entry_id_count).restype, amd_comgr_action_info_get_bundle_entry_id_count.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_get_bundle_entry_id(amd_comgr_action_info_t action_info, size_t index, size_t *size, char *bundle_entry_id)
try: (amd_comgr_action_info_get_bundle_entry_id:=dll.amd_comgr_action_info_get_bundle_entry_id).restype, amd_comgr_action_info_get_bundle_entry_id.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, size_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_set_working_directory_path(amd_comgr_action_info_t action_info, const char *path)
try: (amd_comgr_action_info_set_working_directory_path:=dll.amd_comgr_action_info_set_working_directory_path).restype, amd_comgr_action_info_set_working_directory_path.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_get_working_directory_path(amd_comgr_action_info_t action_info, size_t *size, char *path)
try: (amd_comgr_action_info_get_working_directory_path:=dll.amd_comgr_action_info_get_working_directory_path).restype, amd_comgr_action_info_get_working_directory_path.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_set_logging(amd_comgr_action_info_t action_info, bool logging)
try: (amd_comgr_action_info_set_logging:=dll.amd_comgr_action_info_set_logging).restype, amd_comgr_action_info_set_logging.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.c_bool]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_action_info_get_logging(amd_comgr_action_info_t action_info, bool *logging)
try: (amd_comgr_action_info_get_logging:=dll.amd_comgr_action_info_get_logging).restype, amd_comgr_action_info_get_logging.argtypes = amd_comgr_status_t, [amd_comgr_action_info_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

amd_comgr_action_kind_s = CEnum(ctypes.c_uint32)
AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR', 0)
AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS', 1)
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC', 2)
AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES', 3)
AMD_COMGR_ACTION_LINK_BC_TO_BC = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_LINK_BC_TO_BC', 4)
AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC', 5)
AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE', 6)
AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY', 7)
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE', 8)
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE', 9)
AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE', 10)
AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE', 11)
AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE', 12)
AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE', 13)
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN', 14)
AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC', 15)
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE', 16)
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE', 17)
AMD_COMGR_ACTION_UNBUNDLE = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_UNBUNDLE', 18)
AMD_COMGR_ACTION_LAST = amd_comgr_action_kind_s.define('AMD_COMGR_ACTION_LAST', 18)

amd_comgr_action_kind_t = amd_comgr_action_kind_s
# amd_comgr_status_t amd_comgr_do_action(amd_comgr_action_kind_t kind, amd_comgr_action_info_t info, amd_comgr_data_set_t input, amd_comgr_data_set_t result)
try: (amd_comgr_do_action:=dll.amd_comgr_do_action).restype, amd_comgr_do_action.argtypes = amd_comgr_status_t, [amd_comgr_action_kind_t, amd_comgr_action_info_t, amd_comgr_data_set_t, amd_comgr_data_set_t]
except AttributeError: pass

amd_comgr_metadata_kind_s = CEnum(ctypes.c_uint32)
AMD_COMGR_METADATA_KIND_NULL = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_NULL', 0)
AMD_COMGR_METADATA_KIND_STRING = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_STRING', 1)
AMD_COMGR_METADATA_KIND_MAP = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_MAP', 2)
AMD_COMGR_METADATA_KIND_LIST = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_LIST', 3)
AMD_COMGR_METADATA_KIND_LAST = amd_comgr_metadata_kind_s.define('AMD_COMGR_METADATA_KIND_LAST', 3)

amd_comgr_metadata_kind_t = amd_comgr_metadata_kind_s
# amd_comgr_status_t amd_comgr_get_metadata_kind(amd_comgr_metadata_node_t metadata, amd_comgr_metadata_kind_t *kind)
try: (amd_comgr_get_metadata_kind:=dll.amd_comgr_get_metadata_kind).restype, amd_comgr_get_metadata_kind.argtypes = amd_comgr_status_t, [amd_comgr_metadata_node_t, ctypes.POINTER(amd_comgr_metadata_kind_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_metadata_string(amd_comgr_metadata_node_t metadata, size_t *size, char *string)
try: (amd_comgr_get_metadata_string:=dll.amd_comgr_get_metadata_string).restype, amd_comgr_get_metadata_string.argtypes = amd_comgr_status_t, [amd_comgr_metadata_node_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_metadata_map_size(amd_comgr_metadata_node_t metadata, size_t *size)
try: (amd_comgr_get_metadata_map_size:=dll.amd_comgr_get_metadata_map_size).restype, amd_comgr_get_metadata_map_size.argtypes = amd_comgr_status_t, [amd_comgr_metadata_node_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_iterate_map_metadata(amd_comgr_metadata_node_t metadata, amd_comgr_status_t (*callback)(amd_comgr_metadata_node_t, amd_comgr_metadata_node_t, void *), void *user_data)
try: (amd_comgr_iterate_map_metadata:=dll.amd_comgr_iterate_map_metadata).restype, amd_comgr_iterate_map_metadata.argtypes = amd_comgr_status_t, [amd_comgr_metadata_node_t, ctypes.CFUNCTYPE(amd_comgr_status_t, amd_comgr_metadata_node_t, amd_comgr_metadata_node_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_metadata_lookup(amd_comgr_metadata_node_t metadata, const char *key, amd_comgr_metadata_node_t *value)
try: (amd_comgr_metadata_lookup:=dll.amd_comgr_metadata_lookup).restype, amd_comgr_metadata_lookup.argtypes = amd_comgr_status_t, [amd_comgr_metadata_node_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(amd_comgr_metadata_node_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_metadata_list_size(amd_comgr_metadata_node_t metadata, size_t *size)
try: (amd_comgr_get_metadata_list_size:=dll.amd_comgr_get_metadata_list_size).restype, amd_comgr_get_metadata_list_size.argtypes = amd_comgr_status_t, [amd_comgr_metadata_node_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_index_list_metadata(amd_comgr_metadata_node_t metadata, size_t index, amd_comgr_metadata_node_t *value)
try: (amd_comgr_index_list_metadata:=dll.amd_comgr_index_list_metadata).restype, amd_comgr_index_list_metadata.argtypes = amd_comgr_status_t, [amd_comgr_metadata_node_t, size_t, ctypes.POINTER(amd_comgr_metadata_node_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_iterate_symbols(amd_comgr_data_t data, amd_comgr_status_t (*callback)(amd_comgr_symbol_t, void *), void *user_data)
try: (amd_comgr_iterate_symbols:=dll.amd_comgr_iterate_symbols).restype, amd_comgr_iterate_symbols.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.CFUNCTYPE(amd_comgr_status_t, amd_comgr_symbol_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_symbol_lookup(amd_comgr_data_t data, const char *name, amd_comgr_symbol_t *symbol)
try: (amd_comgr_symbol_lookup:=dll.amd_comgr_symbol_lookup).restype, amd_comgr_symbol_lookup.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(amd_comgr_symbol_t)]
except AttributeError: pass

amd_comgr_symbol_type_s = CEnum(ctypes.c_int32)
AMD_COMGR_SYMBOL_TYPE_UNKNOWN = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_UNKNOWN', -1)
AMD_COMGR_SYMBOL_TYPE_NOTYPE = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_NOTYPE', 0)
AMD_COMGR_SYMBOL_TYPE_OBJECT = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_OBJECT', 1)
AMD_COMGR_SYMBOL_TYPE_FUNC = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_FUNC', 2)
AMD_COMGR_SYMBOL_TYPE_SECTION = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_SECTION', 3)
AMD_COMGR_SYMBOL_TYPE_FILE = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_FILE', 4)
AMD_COMGR_SYMBOL_TYPE_COMMON = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_COMMON', 5)
AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL = amd_comgr_symbol_type_s.define('AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL', 10)

amd_comgr_symbol_type_t = amd_comgr_symbol_type_s
amd_comgr_symbol_info_s = CEnum(ctypes.c_uint32)
AMD_COMGR_SYMBOL_INFO_NAME_LENGTH = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_NAME_LENGTH', 0)
AMD_COMGR_SYMBOL_INFO_NAME = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_NAME', 1)
AMD_COMGR_SYMBOL_INFO_TYPE = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_TYPE', 2)
AMD_COMGR_SYMBOL_INFO_SIZE = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_SIZE', 3)
AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED', 4)
AMD_COMGR_SYMBOL_INFO_VALUE = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_VALUE', 5)
AMD_COMGR_SYMBOL_INFO_LAST = amd_comgr_symbol_info_s.define('AMD_COMGR_SYMBOL_INFO_LAST', 5)

amd_comgr_symbol_info_t = amd_comgr_symbol_info_s
# amd_comgr_status_t amd_comgr_symbol_get_info(amd_comgr_symbol_t symbol, amd_comgr_symbol_info_t attribute, void *value)
try: (amd_comgr_symbol_get_info:=dll.amd_comgr_symbol_get_info).restype, amd_comgr_symbol_get_info.argtypes = amd_comgr_status_t, [amd_comgr_symbol_t, amd_comgr_symbol_info_t, ctypes.c_void_p]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_create_disassembly_info(const char *isa_name, uint64_t (*read_memory_callback)(uint64_t, char *, uint64_t, void *), void (*print_instruction_callback)(const char *, void *), void (*print_address_annotation_callback)(uint64_t, void *), amd_comgr_disassembly_info_t *disassembly_info)
try: (amd_comgr_create_disassembly_info:=dll.amd_comgr_create_disassembly_info).restype, amd_comgr_create_disassembly_info.argtypes = amd_comgr_status_t, [ctypes.POINTER(ctypes.c_char), ctypes.CFUNCTYPE(uint64_t, uint64_t, ctypes.POINTER(ctypes.c_char), uint64_t, ctypes.c_void_p), ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p), ctypes.CFUNCTYPE(None, uint64_t, ctypes.c_void_p), ctypes.POINTER(amd_comgr_disassembly_info_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_destroy_disassembly_info(amd_comgr_disassembly_info_t disassembly_info)
try: (amd_comgr_destroy_disassembly_info:=dll.amd_comgr_destroy_disassembly_info).restype, amd_comgr_destroy_disassembly_info.argtypes = amd_comgr_status_t, [amd_comgr_disassembly_info_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_disassemble_instruction(amd_comgr_disassembly_info_t disassembly_info, uint64_t address, void *user_data, uint64_t *size)
try: (amd_comgr_disassemble_instruction:=dll.amd_comgr_disassemble_instruction).restype, amd_comgr_disassemble_instruction.argtypes = amd_comgr_status_t, [amd_comgr_disassembly_info_t, uint64_t, ctypes.c_void_p, ctypes.POINTER(uint64_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_demangle_symbol_name(amd_comgr_data_t mangled_symbol_name, amd_comgr_data_t *demangled_symbol_name)
try: (amd_comgr_demangle_symbol_name:=dll.amd_comgr_demangle_symbol_name).restype, amd_comgr_demangle_symbol_name.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(amd_comgr_data_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_populate_mangled_names(amd_comgr_data_t data, size_t *count)
try: (amd_comgr_populate_mangled_names:=dll.amd_comgr_populate_mangled_names).restype, amd_comgr_populate_mangled_names.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_get_mangled_name(amd_comgr_data_t data, size_t index, size_t *size, char *mangled_name)
try: (amd_comgr_get_mangled_name:=dll.amd_comgr_get_mangled_name).restype, amd_comgr_get_mangled_name.argtypes = amd_comgr_status_t, [amd_comgr_data_t, size_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_populate_name_expression_map(amd_comgr_data_t data, size_t *count)
try: (amd_comgr_populate_name_expression_map:=dll.amd_comgr_populate_name_expression_map).restype, amd_comgr_populate_name_expression_map.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(size_t)]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_map_name_expression_to_symbol_name(amd_comgr_data_t data, size_t *size, char *name_expression, char *symbol_name)
try: (amd_comgr_map_name_expression_to_symbol_name:=dll.amd_comgr_map_name_expression_to_symbol_name).restype, amd_comgr_map_name_expression_to_symbol_name.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class code_object_info_s(Struct): pass
code_object_info_s._fields_ = [
  ('isa', ctypes.POINTER(ctypes.c_char)),
  ('size', size_t),
  ('offset', uint64_t),
]
amd_comgr_code_object_info_t = code_object_info_s
# amd_comgr_status_t amd_comgr_lookup_code_object(amd_comgr_data_t data, amd_comgr_code_object_info_t *info_list, size_t info_list_size)
try: (amd_comgr_lookup_code_object:=dll.amd_comgr_lookup_code_object).restype, amd_comgr_lookup_code_object.argtypes = amd_comgr_status_t, [amd_comgr_data_t, ctypes.POINTER(amd_comgr_code_object_info_t), size_t]
except AttributeError: pass

# amd_comgr_status_t amd_comgr_map_elf_virtual_address_to_code_object_offset(amd_comgr_data_t data, uint64_t elf_virtual_address, uint64_t *code_object_offset, uint64_t *slice_size, bool *nobits)
try: (amd_comgr_map_elf_virtual_address_to_code_object_offset:=dll.amd_comgr_map_elf_virtual_address_to_code_object_offset).restype, amd_comgr_map_elf_virtual_address_to_code_object_offset.argtypes = amd_comgr_status_t, [amd_comgr_data_t, uint64_t, ctypes.POINTER(uint64_t), ctypes.POINTER(uint64_t), ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

AMD_COMGR_DEPRECATED = lambda msg: __attribute__((deprecated(msg)))
AMD_COMGR_INTERFACE_VERSION_MAJOR = 2
AMD_COMGR_INTERFACE_VERSION_MINOR = 8