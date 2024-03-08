import itertools
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

  # NUM=112 python3 test/external/speed_compare_cuda_ptx.py

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
    for i in range(5):
      tm_cuda.append(cuda_prg(bufs, {}, wait=True))
      tm_ptx.append(ptx_prg(bufs, {}, wait=True))
    ratio = min(tm_ptx)/min(tm_cuda)
    print(f"{num:4d} {ratio:6.2f}x", lin.name)
    if ratio > 2:
      def fix(x): return x.replace('\t', ' ').strip()
      ll1, ll2 = cuda_prg.lib.decode().split('\n'), ptx_src.split('\n')
      if single != -1:
        for ln, (l1, l2) in enumerate(itertools.zip_longest(ll1, ll2, fillvalue='')):
          print(f"{ln:5d} | {fix(l1):80s} | {fix(l2):80s}")
      print(len(ll1), len(ll2), "RATIO", ratio, "us", min(tm_ptx)*1e6)
