from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.ops_cpu import ClangJITCompiler
from tinygrad.runtime.ops_x86 import X86Renderer
from tinygrad.runtime.ops_arm64 import Arm64Renderer
from tinygrad.codegen.heuristic import hand_coded_optimizations
import platform

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # TODO: include float16
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x and "dtypes.half" not in x]
  dev = Device["CPU"]
  asm = X86Renderer() if platform.machine() in ("x86_64", "amd64") else Arm64Renderer()

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_clang, average_tm_asm = 0, 0
  for num,ast in enumerate(ast_strs):
    # clang compile
    dev.compiler = ClangJITCompiler(opt_args=['-O1', '-march=native'])
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.apply_opts(hand_coded_optimizations(lin))
    clang_prg = CompiledRunner(lin.to_program())

    bufs = bufs_from_lin(lin)

    # asm compile
    dev.compiler = ClangJITCompiler(lang_args=['assembler'] + (['-masm=intel']) if isinstance(asm, X86Renderer) else [])
    lin = ast_str_to_lin(ast, opts=asm)
    lin.apply_opts(hand_coded_optimizations(lin))
    asm_prg = CompiledRunner(lin.to_program())

    # warmup
    try:
      runtime = clang_prg(bufs, {}, wait=True)
    except RuntimeError:
      print("clang failed ast:", num)
      continue
    if runtime > 1:
      print("kernel timeout")
      continue
    asm_prg(bufs, {}, wait=True)

    tm_clang, tm_asm = [], []
    for i in range(5):
      tm_clang.append(clang_prg(bufs, {}, wait=True))
      tm_asm.append(asm_prg(bufs, {}, wait=True))
    average_tm_clang += min(tm_clang)
    average_tm_asm += min(tm_asm)
    ratio = min(tm_asm)/min(tm_clang)
    print(f"{average_tm_asm/average_tm_clang:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_asm)*1e6:7.2f} us", lin.name)
