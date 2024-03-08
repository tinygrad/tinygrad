from tinygrad import Device
from tinygrad.device import CompiledASTRunner
from tinygrad.helpers import to_function_name, getenv
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.features.search import bufs_from_lin
from tinygrad.runtime.ops_cuda import PTXCompiler

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  dev = Device["CUDA"]
  ptx = PTXCompiler(dev.arch)

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  for num,ast in enumerate(ast_strs):
    # cuda compile
    lin = ast_str_to_lin(ast, opts=dev.compiler.linearizer_opts)
    lin.hand_coded_optimizations()
    cuda_prg = dev.to_program(lin)

    bufs = bufs_from_lin(lin)

    # ptx compile
    lin = ast_str_to_lin(ast, opts=ptx.linearizer_opts)
    lin.hand_coded_optimizations()
    lin.linearize()
    ptx_src = ptx.render(to_function_name(lin.name), lin.uops)
    try:
      ptx_prg = CompiledASTRunner(lin.name, ptx_src, dev, lin.global_size, lin.local_size, lin.uops.vars(), precompiled=ptx.compile(ptx_src))
    except RuntimeError:
      print("PTX FAIL")
      continue
    # warmup
    cuda_prg(bufs, {}, wait=True)

    tm_cuda, tm_ptx = [], []
    for i in range(3):
      tm_cuda.append(cuda_prg(bufs, {}, wait=True))
      tm_ptx.append(ptx_prg(bufs, {}, wait=True))
    ratio = sum(tm_ptx)/sum(tm_cuda)
    print(f"{num:4d} {ratio:6.2f}x", lin.name)
    if ratio > 2:
      for l1, l2 in zip(cuda_prg.lib.decode(), ptx_src):
        print(f"{l1:40s} | {l2:40s}")



