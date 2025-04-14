from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.support.compiler_amd import HIPCompiler, AMDLLVMCompiler
from tinygrad.renderer.cstyle import AMDRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  dev = Device["AMD"]
  amd_render = AMDRenderer(dev.arch)
  llvm_render = AMDLLVMRenderer()
  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]
  avg_tm_hip, avg_tm_llvm = 0, 0
  for num, ast in enumerate(ast_strs):
    dev.compiler = HIPCompiler(dev.arch)
    lin_hip = ast_str_to_lin(ast, opts=amd_render)
    lin_hip.hand_coded_optimizations()
    hip_prg = lin_hip.to_program()
    hip_runner = CompiledRunner(hip_prg)

    bufs = bufs_from_lin(lin_hip)

    # llvm compile
    dev.compiler = AMDLLVMCompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=llvm_render)
    lin.hand_coded_optimizations()
    lin.linearize()
    llvm_prg = lin.to_program()
    llvm_runner = CompiledRunner(llvm_prg)

    # warmup
    try:
      hip_runner(bufs, {}, wait=True)
    except RuntimeError:
      print("hip failed ast:", num)
      continue
    llvm_runner(bufs, {}, wait=True)

    tm_hip, tm_llvm = [], []
    for i in range(5):
      tm_hip.append(hip_runner(bufs, {}, wait=True))
      tm_llvm.append(llvm_runner(bufs, {}, wait=True))
    avg_tm_hip += min(tm_hip)
    avg_tm_llvm += min(tm_llvm)
    ratio = min(tm_llvm)/min(tm_hip)
    print(f"{avg_tm_llvm/avg_tm_hip:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_llvm)*1e6:7.2f}(hip={min(tm_hip)*1e6:7.2f}) us", lin.name)
