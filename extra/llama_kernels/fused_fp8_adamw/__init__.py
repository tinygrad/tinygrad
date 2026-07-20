import functools, pathlib
from dataclasses import replace
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import prod
from tinygrad.renderer import Estimates
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.uop.ops import UOp, Ops, KernelInfo, ProgramInfo

NUM_WG, THREADS = 512, 256

@functools.cache
def custom_fused_fp8_adamw(master:UOp, weight:UOp, next_inv:UOp, m:UOp, v:UOp, grad:UOp, grad_scale:UOp,
                           inv_scale:UOp, lr:UOp, b1_t:UOp, b2_t:UOp,
                           *, arch:str, b1:float, b2:float, eps:float, wd:float) -> UOp:
  layers = master.shape[0]
  assert master.shard_shape[0] == layers, f"unsupported layer-sharded fp8 optimizer shape {master.shape}"
  layer_elems = prod(master.shard_shape) // layers
  assert layer_elems % (NUM_WG * THREADS) == 0, f"unsupported fp8 optimizer shape {master.shape}"
  threads = UOp.special(THREADS, "lidx0")
  workgroups = UOp.special(layers * NUM_WG, "gidx0")
  sink = UOp.sink(master.base, weight.base, next_inv.base, m.base, v.base, grad.base, grad_scale.base, inv_scale.base, lr.base, b1_t.base, b2_t.base,
                  threads, workgroups, arg=KernelInfo(f"fused_fp8_adamw_{layers}_{layer_elems}",
                  estimates=Estimates(ops=25*layers*layer_elems, mem=19*layers*layer_elems)))
  code = (pathlib.Path(__file__).parent / "fused_fp8_adamw.cpp").read_text()
  lib = HIPCCCompiler(arch, ["-std=c++20", "-ffast-math", f"-DLAYERS={layers}", f"-DLAYER_ELEMS={layer_elems}",
                               f"-DBETA1={b1}f", f"-DBETA2={b2}f", f"-DONE_MINUS_BETA1={1.0-b1}f", f"-DONE_MINUS_BETA2={1.0-b2}f",
                               f"-DEPS={eps}f", f"-DWEIGHT_DECAY={wd}f"]).compile_cached(code)
  info = ProgramInfo.from_sink(sink)
  info = replace(info, outs=info.globals[:5], ins=(info.globals[0], *info.globals[2:]))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=code), UOp(Ops.BINARY, arg=lib)), arg=info)

def fused_fp8_adamw(master:Tensor, weight:Tensor, next_inv:Tensor, m:Tensor, v:Tensor, grad:Tensor, grad_scale:Tensor, inv_scale:Tensor,
                    lr:Tensor, b1_t:Tensor, b2_t:Tensor, *, b1:float, b2:float, eps:float, wd:float):
  assert master.dtype == dtypes.float32 and weight.dtype == dtypes.fp8e4m3
  assert m.dtype == v.dtype == grad.dtype == dtypes.bfloat16
  device = master.device[0] if isinstance(master.device, tuple) else master.device
  arch = Device[device].renderer.target.arch
  return Tensor.custom_kernel(master, weight, next_inv, m, v, grad, grad_scale, inv_scale, lr, b1_t, b2_t,
    fxn=functools.partial(custom_fused_fp8_adamw, arch=arch, b1=b1, b2=b2, eps=eps, wd=wd))[:5]
