from tinygrad import Context
from tinygrad.engine.realize import CompiledRunner, get_program
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_ast
from tinygrad.codegen.opt.postrange import bufs_from_ast
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.isa import X86Renderer

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x and "dtypes.half" not in x]

  # these kernels are all dividing by 0
  ast_strs = ast_strs[:209] + ast_strs[210:]
  ast_strs = ast_strs[:408] + ast_strs[409:] # don't know about this one
  ast_strs = ast_strs[:995] + ast_strs[996:]
  ast_strs = ast_strs[:1021] + ast_strs[1022:] # don't know about this one
  ast_strs = ast_strs[:1689] + ast_strs[1690:]
  ast_strs = ast_strs[:2182] + ast_strs[2183:]

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_clang, average_tm_asm = 0, 0
  for num,ast_str in enumerate(ast_strs):
    try: ast = ast_str_to_ast(ast_str)
    except TypeError: continue
    # clang compile
    clang_prg = CompiledRunner(get_program(ast, ClangRenderer()))

    bufs = bufs_from_ast(ast, "CPU")
    for b in bufs: b = b.allocate()

    # asm compile
    with Context(DEVECTORIZE=0, CPU_X86=1):
      asm_prg = CompiledRunner(get_program(ast, X86Renderer()))

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
    print(f"{average_tm_asm/average_tm_clang:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_asm)*1e6:7.2f} us", clang_prg.display_name)