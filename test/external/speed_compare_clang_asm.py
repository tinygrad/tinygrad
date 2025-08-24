from tinygrad import Device, Context
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.opt.search import bufs_from_lin
from tinygrad.runtime.ops_cpu import ClangJITCompiler
from tinygrad.runtime.ops_x86 import X86Renderer
from tinygrad.codegen.opt.heuristic import hand_coded_optimizations

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # TODO: include float16
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x and "dtypes.half" not in x]
  dev = Device["CPU"]

  # these kernels are all dividing by 0
  ast_strs = ast_strs[:209] + ast_strs[210:]
  ast_strs = ast_strs[:996] + ast_strs[997:]
  ast_strs = ast_strs[:1691] + ast_strs[1692:]
  ast_strs = ast_strs[:2260] + ast_strs[2261:]
  # these two seg fault
  ast_strs = ast_strs[:3001] + ast_strs[3002:]
  ast_strs = ast_strs[:3719] + ast_strs[3720:]


  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_clang, average_tm_asm = 0, 0
  for num,ast in enumerate(ast_strs):
    # clang compile
    dev.compiler = ClangJITCompiler()
    try:
      lin = ast_str_to_lin(ast, opts=dev.renderer)
    except TypeError:
      continue
    lin.apply_opts(hand_coded_optimizations(lin))
    clang_prg = CompiledRunner(get_program(lin.get_optimized_ast(), lin.opts))

    bufs = bufs_from_lin(lin)

    # asm compile
    with Context(DEVECTORIZE=0):
      lin = ast_str_to_lin(ast, opts=X86Renderer())
      lin.apply_opts(hand_coded_optimizations(lin))
      asm_prg = CompiledRunner(get_program(lin.get_optimized_ast(), lin.opts))

    tm_clang, tm_asm = [], []
    # warmup
    clang_prg(bufs, {}, wait=True)
    asm_prg(bufs, {}, wait=True)
    for i in range(5):
      tm_clang.append(clang_prg(bufs, {}, wait=True))
      tm_asm.append(asm_prg(bufs, {}, wait=True))
    average_tm_clang += min(tm_clang)
    average_tm_asm += min(tm_asm)
    ratio = min(tm_asm)/min(tm_clang)
    print(f"{average_tm_asm/average_tm_clang:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_asm)*1e6:7.2f} us", lin.name)