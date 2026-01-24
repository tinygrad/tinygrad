import itertools
from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.codegen import get_program
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_lin, ast_str_to_ast
from tinygrad.codegen.opt.postrange import bufs_from_ast
from tinygrad.runtime.ops_cuda import PTXCompiler, PTXRenderer, CUDACompiler, CUDARenderer

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # no bfloat16 for ptx at the moment, also skip malformed ASTs with Invalid
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x and "Invalid" not in x]
  dev = Device["CUDA"]

  cuda_renderer = CUDARenderer(dev.arch)
  cuda_renderer.compiler = CUDACompiler(dev.arch)

  ptx_renderer = PTXRenderer(dev.arch)
  ptx_renderer.compiler = PTXCompiler(dev.arch)

  # NUM=112 python3 test/external/speed_compare_cuda_ptx.py

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_cuda, average_tm_ptx = 0, 0
  for num, ast in enumerate(ast_strs):
    # cuda compile
    lin = hand_coded_optimizations(ast_str_to_lin(ast, opts=cuda_renderer))
    cuda_prg = CompiledRunner(get_program(lin.get_optimized_ast(), lin.ren))

    bufs = [b.allocate() for b in bufs_from_ast(ast_str_to_ast(ast), "CUDA")]

    # ptx compile
    lin = hand_coded_optimizations(ast_str_to_lin(ast, opts=ptx_renderer))
    ptx_prg = CompiledRunner(get_program(lin.get_optimized_ast(), lin.ren))

    # warmup
    try:
      cuda_prg(bufs, {}, wait=True)
    except RuntimeError:
      print("cuda failed ast:", num)
      continue
    ptx_prg(bufs, {}, wait=True)

    tm_cuda, tm_ptx = [], []
    for i in range(5):
      tm_cuda.append(cuda_prg(bufs, {}, wait=True))
      tm_ptx.append(ptx_prg(bufs, {}, wait=True))
    average_tm_cuda += min(tm_cuda)
    average_tm_ptx += min(tm_ptx)
    ratio = min(tm_ptx)/min(tm_cuda)
    print(f"{average_tm_ptx/average_tm_cuda:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_ptx)*1e6:7.2f} us", ptx_prg.p.name)
    if ratio > 1.5:
      def fix(x): return x.replace('\t', ' ').strip()
      ll1, ll2 = cuda_prg.p.lib.decode().split('\n'), ptx_prg.p.lib.decode().split('\n')
      if single != -1:
        for ln, (l1, l2) in enumerate(itertools.zip_longest(ll1, ll2, fillvalue='')):
          print(f"{ln:5d} | {fix(l1):80s} | {fix(l2):80s}")
      print(len(ll1), len(ll2), "RATIO", ratio, "us", min(tm_ptx)*1e6)
