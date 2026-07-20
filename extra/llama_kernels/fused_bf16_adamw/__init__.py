import functools, pathlib
from dataclasses import replace
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import prod
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo

NUM_WG, THREADS = 512, 256

@functools.cache
def custom_fused_bf16_adamw(master:UOp, weight:UOp, m:UOp, v:UOp, grad:UOp, grad_scale:UOp, lr:UOp, b1_t:UOp, b2_t:UOp,
                            *, arch:str, b1:float, b2:float, eps:float, wd:float) -> UOp:
  elems = prod(master.shard_shape)
  assert elems % 2 == 0, f"unsupported bf16 optimizer shape {master.shape}"
  threads = UOp.special(THREADS, "lidx0")
  workgroups = UOp.special(NUM_WG, "gidx0")
  sink = UOp.sink(master.base, weight.base, m.base, v.base, grad.base, grad_scale.base, lr.base, b1_t.base, b2_t.base,
                  threads, workgroups, arg=KernelInfo(f"fused_bf16_adamw_{elems}", estimates=Estimates(ops=21*elems, mem=20*elems)))
  code = (pathlib.Path(__file__).parent / "fused_bf16_adamw.cpp").read_text()
  lib = HIPCCCompiler(arch, ["-std=c++20", "-ffast-math", f"-DELEMS={elems}", f"-DBETA1={b1}f", f"-DBETA2={b2}f",
                               f"-DONE_MINUS_BETA1={1.0-b1}f", f"-DONE_MINUS_BETA2={1.0-b2}f", f"-DEPS={eps}f",
                               f"-DWEIGHT_DECAY={wd}f"]).compile_cached(code)
  info = ProgramInfo.from_sink(sink)
  info = replace(info, outs=info.globals[:4], ins=(info.globals[0], *info.globals[2:]))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)), arg=info)

def fused_bf16_adamw(master:Tensor, weight:Tensor, m:Tensor, v:Tensor, grad:Tensor, grad_scale:Tensor, lr:Tensor,
                     b1_t:Tensor, b2_t:Tensor, *, b1:float, b2:float, eps:float, wd:float):
  assert master.dtype == dtypes.float32 and weight.dtype == m.dtype == v.dtype == grad.dtype == dtypes.bfloat16
  device = master.device[0] if isinstance(master.device, tuple) else master.device
  arch = Device[device].renderer.target.arch
  return Tensor.custom_kernel(master, weight, m, v, grad, grad_scale, lr, b1_t, b2_t,
    fxn=functools.partial(custom_fused_bf16_adamw, arch=arch, b1=b1, b2=b2, eps=eps, wd=wd))[:4]
