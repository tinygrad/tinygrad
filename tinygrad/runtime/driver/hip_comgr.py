import ctypes

def check(status):
  import tinygrad.runtime.autogen.comgr as comgr
  if status != 0:
    comgr.amd_comgr_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"comgr fail {status}, {ctypes.string_at(status_str).decode()}")

def _get_comgr_data(data_set, data_type):
  import tinygrad.runtime.autogen.comgr as comgr
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz := ctypes.c_uint64()), None))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz), (dat := ctypes.create_string_buffer(sz.value))))
  check(comgr.amd_comgr_release_data(data_exec))
  return bytes(dat)

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip(prg:str, arch="gfx1100") -> bytes:
  import http.client, urllib.parse
  params = urllib.parse.urlencode({'code': prg})
  headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
  conn = http.client.HTTPConnection("temps-mbp.home", 80)
  conn.request("POST", "/", params, headers)
  response = conn.getresponse()
  asm = response.read().decode()
  conn.close()
  return asm.encode("utf-8")
