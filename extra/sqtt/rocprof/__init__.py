from tinygrad.runtime.autogen import load

rocprof_src = "https://github.com/ROCm/rocprof-trace-decoder/archive/dd0485100971522cc4cd8ae136bdda431061a04d.tar.gz"

def __getattr__(nm):
  if nm == "rocprof": return load("rocprof", ["find_library('rocprof-trace-decoder')"], [f"{{}}/include/{s}.h" for s in
    ["rocprof_trace_decoder","trace_decoder_instrument","trace_decoder_types"]], tarball=rocprof_src, path=__name__)
