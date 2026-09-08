"""AITER gfx950 native FP8 causal FA, specialized to the MLPerf Llama shape.

The pinned code object is MIT-licensed ROCm/aiter. Only its kernarg loads are
adapted: buffer pointers use tinygrad's ABI and fixed-shape scalars become
literal moves of the same size, leaving every branch displacement unchanged.
"""
import functools, math, struct
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import fetch
from tinygrad.renderer import Estimates
from tinygrad.renderer.amd import decode_inst
from tinygrad.runtime.autogen.amd.cdna.ins import SMEM, s_mov_b32, LIT
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import UOp, Ops, KernelInfo

@functools.cache
def build_kernel(B:int, N:int, H:int, H_KV:int, D:int, pre_scaled:bool=False, saved_bf16:bool=False):
  assert B in (1, 2) and (N, H, H_KV, D) == (8192, 32, 8, 128)
  co = fetch('https://raw.githubusercontent.com/ROCm/aiter/ba59a37aa65a34ed19edb169c7b8ea59e4781af5/'
             'hsa/gfx950/fmha_v3_fwd/fwd_hd128_fp8_causal.co',
             sha256='ca8f416739558e92a189e733efa77af0a3db00460ed7bed83361b839e39281b3').read_bytes()
  code = next(s.content for s in elf_loader(co)[1] if s.name == '.text')
  # fmha_fwd_v3_args uses 16-byte slots. Tensor.custom_kernel args: O,LSE,Q,K,V,Qscale,Kscale,Vscale.
  slots = (0, 4, 5, 9, 1, 7, 8, 10) if saved_bf16 else (0, 2, 3, 4, 1, 5, 6, 7)
  pointers = dict(zip((0, 16, 32, 48, 64, 512, 528, 544), (x*8 for x in slots)))
  scalar = 1 / math.log2(math.e) if pre_scaled else D ** -0.5
  constants = {80:struct.unpack('<I', struct.pack('<f', scalar))[0], 96:N, 112:H*D, 128:256*H*D,
               144:D, 160:N*H*D, 176:H//H_KV, 192:H_KV*D, 208:D, 224:N*H_KV*D, 240:5, 256:1, 272:N,
               288:D, 304:D, 320:H, 336:H_KV*D, 352:D, 368:N*H_KV*D, 384:H*D*2, 400:D*2, 416:N*H*D*2, 464:N*4}
  insts, pos = [], 0
  while pos < len(code):
    inst = decode_inst(code[pos:], 'cdna')
    size = inst.size()
    assert inst.to_bytes() == code[pos:pos+size], f'ISA roundtrip failed at {pos:#x}'
    if pos < 0xf4 and isinstance(inst, SMEM):
      if inst.offset in pointers: inst.offset = pointers[inst.offset]
      else: inst = s_mov_b32(inst.sdata, LIT, constants[inst.offset])
      assert inst.size() == size
    insts.append(inst)
    pos += size
  return insts

@functools.cache
def custom_asm_fp8_fa_forward(o:UOp, lse:UOp, *inputs:UOp,
                              B:int, N:int, H:int, H_KV:int, D:int, pre_scaled:bool=False, saved_bf16:bool=False):
  if saved_bf16: _, _, q, k, _, qs, ks, v, vs = inputs
  else: q, k, v, qs, ks, vs = inputs
  assert q.dtype == k.dtype == v.dtype == dtypes.fp8e4m3
  assert o.dtype == dtypes.bfloat16 and lse.dtype == qs.dtype == ks.dtype == vs.dtype == dtypes.float32
  assert all(math.prod(x.shape) == 1 for x in (qs, ks, vs))
  zero = UOp.const(0)
  writes = [x.flatten().index(zero).store(x.flatten().index(zero).load()) for x in (o, lse)]
  reads = [x.flatten().index(zero).load() for x in inputs]
  mem = B*N*(H*D*3+H_KV*D*2)+B*H*N*4
  sink = UOp.sink(*writes, *reads, UOp.placeholder((163840,), dtypes.uint8, 0, AddrSpace.LOCAL),
                  UOp.special(512, 'lidx0'), UOp.special(N//512, 'gidx0'), UOp.special(H, 'gidx1'), UOp.special(B, 'gidx2'),
                  arg=KernelInfo(name=f'asm_fa_fwd_fp8_causal_{B}_{N}_{H}_{H_KV}_{D}',
                                 estimates=Estimates(ops=2*B*H*N*N*D, mem=mem)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR,
    src=tuple(UOp(Ops.INS, arg=(x, dtypes.void)) for x in build_kernel(B, N, H, H_KV, D, pre_scaled, saved_bf16)))))

def flash_attention_fp8(q:Tensor, k:Tensor, v:Tensor, qs:Tensor, ks:Tensor, vs:Tensor):
  """Prequantized forward only: Q/K/V are contiguous B,N,H,D E4M3; scales are physical descales."""
  B, N, H, D = q.shape
  assert isinstance(q.device, str)
  from tinygrad import Device
  assert Device[q.device].renderer.target.arch == 'gfx950'
  assert k.shape == v.shape == (B, N, k.shape[2], D)
  o = Tensor.empty(B, N, H, D, device=q.device, dtype=dtypes.bfloat16)
  lse = Tensor.empty(B, H, 1, N, device=q.device, dtype=dtypes.float32)
  return Tensor.custom_kernel(o, lse, q, k, v, qs, ks, vs,
    fxn=functools.partial(custom_asm_fp8_fa_forward, B=B, N=N, H=H, H_KV=k.shape[2], D=D))[:2]
