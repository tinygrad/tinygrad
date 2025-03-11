import itertools
from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.ops_cpu import ClangJITCompiler
from tinygrad.runtime.ops_x86 import X86Renderer

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # no bfloat16 for ptx at the moment
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x and "dtypes.half" not in x]
  dev = Device["CPU"]
  x86 = X86Renderer()

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_clang, average_tm_x86 = 0, 0
  for num,ast in enumerate(ast_strs):
    # clang compile
    dev.compiler = ClangJITCompiler()
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    clang_prg = CompiledRunner(lin.to_program())

    bufs = bufs_from_lin(lin)

    # x86 compile
    dev.compiler = ClangJITCompiler(lang_args=['assembler', '-masm=intel'])
    lin = ast_str_to_lin(ast, opts=x86)
    lin.hand_coded_optimizations()
    x86_prg = CompiledRunner(lin.to_program())

    # warmup
    try:
      runtime = clang_prg(bufs, {}, wait=True)
    except RuntimeError:
      print("clang failed ast:", num)
      continue
    if runtime > 5:
      print("kernel timeout")
      continue
    x86_prg(bufs, {}, wait=True)

    tm_clang, tm_x86 = [], []
    for i in range(5):
      tm_clang.append(clang_prg(bufs, {}, wait=True))
      tm_x86.append(x86_prg(bufs, {}, wait=True))
    average_tm_clang += min(tm_clang)
    average_tm_x86 += min(tm_x86)
    ratio = min(tm_x86)/min(tm_clang)
    print(f"{average_tm_x86/average_tm_clang:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_x86)*1e6:7.2f} us", lin.name)
    #if ratio > 1.5:
    #  def fix(x): return x.replace('\t', ' ').strip()
    #  ll1, ll2 = clang_prg.lib.decode().split('\n'), x86_prg.lib.decode().split('\n')
    #  if single != -1:
    #    for ln, (l1, l2) in enumerate(itertools.zip_longest(ll1, ll2, fillvalue='')):
    #      print(f"{ln:5d} | {fix(l1):80s} | {fix(l2):80s}")
    #  print(len(ll1), len(ll2), "RATIO", ratio, "us", min(tm_x86)*1e6)