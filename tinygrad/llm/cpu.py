from __future__ import annotations
import functools, platform, math
from typing import TYPE_CHECKING
from tinygrad import Tensor, nn, UOp, getenv, dtypes
from tinygrad.dtype import DType, AddrSpace
from tinygrad.llm.gguf import _GGML_QUANT, ggml_data_to_tensor
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import Ops, KernelInfo, AxisType
if TYPE_CHECKING:
  from tinygrad.llm.model import ExpertWeights, FFNBlock, Linear

SUPPORTED = platform.system() != "Windows"

def _concrete_int(value:int|UOp) -> int:
  assert isinstance(value, int)
  return value

def _dot_byte_parts(a:tuple[UOp, ...], b:tuple[UOp, ...]) -> UOp:
  av, bv = UOp.stack(*a), UOp.stack(*b)
  return _dot_byte_vectors(av, bv)

def _dot_byte_vectors(a:UOp, b:UOp, scale:UOp|None=None) -> UOp:
  assert a.shape == b.shape and len(a.shape) == 1 and a.shape[0] % 4 == 0
  product = a.int() * b.int()
  if scale is not None: product = product * scale
  return product.reshape(a.shape[0]//4, 4)._rop(Ops.ADD, (1,))

def _unpack_nibbles(packed:UOp, values:tuple[int, ...]) -> UOp:
  assert packed.shape == (16,) and len(values) == 16
  def lookup(index:UOp) -> UOp:
    value = UOp.stack(*(UOp.const(dtypes.int8, values[-1]) for _ in range(packed.shape[0])))
    for i in range(14, -1, -1): value = index.eq(i).where(UOp.const(dtypes.int8, values[i]), value)
    return value
  lowv, highv = lookup(packed & 15), lookup((packed >> 4) & 15)
  return UOp.stack(lowv, highv).reshape(32)

def _dot_bytes(a:tuple[UOp, ...], b:tuple[UOp, ...]) -> UOp:
  parts = _dot_byte_parts(a, b)
  return sum((parts[i] for i in range(len(a)//4)), UOp.const(dtypes.int32, 0))

def _contiguous_vector_load(ptr:UOp, lanes:int, dtype:DType|None=None) -> UOp:
  assert ptr.op is Ops.INDEX
  buf, coords = ptr.src[0], ptr.src[1:]
  index = sum((coord * math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(dtypes.weakint, 0))
  address = UOp(Ops.SHRINK, src=(buf.flatten(), index, UOp.const(dtypes.weakint, lanes)))
  return address.load(dtype=dtype or ptr.dtype)

def _contiguous_vector_ptr(buf:UOp, index:UOp, lanes:int) -> UOp:
  return UOp(Ops.SHRINK, src=(buf, index, UOp.const(dtypes.weakint, lanes)))

def _dot_bytes_ptr(a:UOp, b:UOp) -> UOp:
  av = _contiguous_vector_load(a, 32, dtypes.int8)
  bv = _contiguous_vector_load(b, 32, dtypes.int8)
  return _dot_byte_vectors(av, bv)

def _dot_nibbles_ptr(packed:UOp, x:UOp, values:tuple[int, ...]) -> UOp:
  assert len(values) == 16
  pv = _contiguous_vector_load(packed, 16)
  qvalues = _unpack_nibbles(pv, values)
  xv = _contiguous_vector_load(x, 32, dtypes.int8)
  return _dot_byte_vectors(qvalues, xv)

def _dot_nibbles_pair_ptr(packed:UOp, x:UOp, values:tuple[int, ...]) -> tuple[UOp, UOp]:
  assert len(values) == 16
  packed1, x1 = (ptr.replace(src=(*ptr.src[:-1], ptr.src[-1] + off)) for ptr,off in ((packed, 16), (x, 32)))
  return _dot_nibbles_ptr(packed, x, values), _dot_nibbles_ptr(packed1, x1, values)

def _dot_q6_ptr(block:UOp, x:UOp, subgroup:int) -> UOp:
  lane, half = subgroup & 3, subgroup >> 2
  low_offset, low_shift = half * 64 + (lane & 1) * 32, (lane >> 1) * 4
  high_offset, high_shift = 128 + half * 32, lane * 2
  lo_ptr = block.replace(src=(*block.src[:-1], block.src[-1] + low_offset))
  hi_ptr = block.replace(src=(*block.src[:-1], block.src[-1] + high_offset))
  lo = (_contiguous_vector_load(lo_ptr, 32) >> low_shift) & 15
  hi = ((_contiguous_vector_load(hi_ptr, 32) >> high_shift) & 3) << 4
  qvalues = ((lo | hi) - 32).bitcast(dtypes.int8)
  xv = _contiguous_vector_load(x, 32, dtypes.int8)
  return _dot_byte_vectors(qvalues, xv)

def _rms_f16_product_ptr(x:UOp, norm:UOp, weight:UOp, scale:UOp) -> UOp:
  return _contiguous_vector_load(x, 8).float() * _contiguous_vector_load(norm, 8).float() * \
         _contiguous_vector_load(weight, 8).float() * scale

def _parallel_work(total:int, core_limit:int=32) -> tuple[UOp, UOp, UOp]:
  cores = min(total, core_limit)
  # Equal-size partitions keep symbolic estimates independent of the runtime core_id.
  while total % cores: cores -= 1
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  begin, end = total * core // cores, total * (core + 1) // cores
  job = UOp.range(end - begin, 90)
  return core, job, begin + job

@functools.cache
def _q8_quantize_kernel(quant:UOp, scale:UOp, x:UOp, in_features:int) -> UOp:
  tokens, groups = quant.shape[:2]
  cores = 1 if tokens * groups <= 128 else min(tokens * groups, 32)
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  begin, end = tokens * groups * core // cores, tokens * groups * (core + 1) // cores
  job = UOp.range(end - begin, 90)
  work = begin + job
  token, group = work // groups, work % groups
  values = tuple(x[token * in_features + group * 32 + i].load().float() for i in range(32))
  amax = functools.reduce(lambda a,b:a.maximum(b), (value.abs() for value in values))
  d = (amax / 127).maximum(1e-8)
  stores = [scale[token, group].store(d)] + \
           [quant[token, group, i].store((value / d).round().maximum(-127).minimum(127).cast(dtypes.int8))
            for i,value in enumerate(values)]
  return UOp.group(*stores).end(job, core).sink(arg=KernelInfo(name="q8_quantize_cpu", optimize=False, parallel=True))

def q8_quantize(x:Tensor, in_features:int) -> tuple[Tensor, Tensor]:
  tokens, xc = int(x.numel()) // in_features, x.reshape(-1).contiguous()
  quant = Tensor.empty(tokens, in_features // 32, 32, dtype=dtypes.int8, device=x.device)
  scale = Tensor.empty(tokens, in_features // 32, dtype=dtypes.float32, device=x.device)
  return tuple(Tensor.custom_kernel(quant, scale, xc, fxn=lambda quant,scale,x:
    _q8_quantize_kernel(quant, scale, x, in_features))[:2])  # type: ignore[return-value]

@functools.cache
def _rmsnorm_q8_quantize_kernel(normed:UOp, quant:UOp, qscale:UOp, x:UOp, weight:UOp, in_features:int, eps:float) -> UOp:
  acc = UOp.placeholder((8,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = _vector_acc_init(acc)
  chunk = UOp.range(in_features // 8, 90)
  values = _contiguous_vector_load(x[chunk * 8], 8).float()
  summed = _vector_acc_update(acc, values * values, chunk).end(chunk)
  sumsq = sum((acc.after(summed)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  rms_scale = (sumsq / in_features + eps).rsqrt()

  values_reg = UOp.placeholder((32,), dtypes.float16, slot=1, addrspace=AddrSpace.REG)
  group = UOp.range(in_features // 32, 91)
  value_stores:list[UOp] = []
  for lane in range(32):
    idx = group * 32 + lane
    value = x[idx].load().float() * weight[idx].load().float() * rms_scale
    rounded = value.cast(dtypes.float16)
    value_stores.extend((values_reg[lane].after(group).store(rounded), normed[idx].store(rounded)))
  values_ready = UOp.group(*value_stores)
  group_values = [values_reg.after(values_ready)[lane].load().float() for lane in range(32)]
  amax = functools.reduce(lambda a,b:a.maximum(b), (value.abs() for value in group_values))
  d = (amax / 127).maximum(1e-8)
  quant_stores = [qscale[0, group].store(d)] + [
    quant[0, group, lane].store((value / d).round().maximum(-127).minimum(127).cast(dtypes.int8))
    for lane,value in enumerate(group_values)]
  return UOp.group(*quant_stores).end(group).sink(
    arg=KernelInfo(name=f"rmsnorm_q8_quantize_cpu_{in_features}", opts_to_apply=()))

def rmsnorm_q8_quantize(x:Tensor, weight:Tensor, eps:float, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  xc, wc = x.reshape(-1).contiguous(), weight.reshape(-1).contiguous()
  normed = Tensor.empty(in_features, dtype=dtypes.float16, device=x.device)
  quant = Tensor.empty(1, in_features // 32, 32, dtype=dtypes.int8, device=x.device)
  scale = Tensor.empty(1, in_features // 32, dtype=dtypes.float32, device=x.device)
  return tuple(Tensor.custom_kernel(normed, quant, scale, xc, wc, fxn=lambda normed,quant,scale,x,weight:
    _rmsnorm_q8_quantize_kernel(normed, quant, scale, x, weight, in_features, eps))[:3])  # type: ignore[return-value]

@functools.cache
def _q8_silu_quantize_kernel(quant:UOp, scale:UOp, gate:UOp, up:UOp, in_features:int) -> UOp:
  tokens, groups = quant.shape[:2]
  cores = 1 if tokens == 1 else min(tokens * groups, 32)
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  begin, end = tokens * groups * core // cores, tokens * groups * (core + 1) // cores
  job = UOp.range(end - begin, 90)
  work = begin + job
  token, group = work // groups, work % groups
  values_reg = UOp.placeholder((32,), dtypes.float16, slot=0, addrspace=AddrSpace.REG)
  value_stores = []
  for lane in range(32):
    idx = token * in_features + group * 32 + lane
    g = gate[idx].load()
    value_stores.append(values_reg[lane].after(job).store((g * g.sigmoid() * up[idx].load()).cast(dtypes.float16)))
  values_ready = UOp.group(*value_stores)
  values = [values_reg.after(values_ready)[lane].load().float() for lane in range(32)]
  amax = functools.reduce(lambda a,b:a.maximum(b), (value.abs() for value in values))
  d = (amax / 127).maximum(1e-8)
  stores = [scale[token, group].store(d)] + \
           [quant[token, group, lane].store((value / d).round().maximum(-127).minimum(127).cast(dtypes.int8))
            for lane,value in enumerate(values)]
  return UOp.group(*stores).end(job, core).sink(
    arg=KernelInfo(name="q8_silu_quantize_cpu", optimize=False, parallel=True))

def q8_silu_quantize(gate:Tensor, up:Tensor, in_features:int) -> tuple[Tensor, Tensor]:
  assert gate.shape == up.shape
  tokens = int(gate.numel()) // in_features
  gatec, upc = gate.reshape(-1).contiguous(), up.reshape(-1).contiguous()
  quant = Tensor.empty(tokens, in_features // 32, 32, dtype=dtypes.int8, device=gate.device)
  scale = Tensor.empty(tokens, in_features // 32, dtype=dtypes.float32, device=gate.device)
  return tuple(Tensor.custom_kernel(quant, scale, gatec, upc, fxn=lambda quant,scale,gate,up:
    _q8_silu_quantize_kernel(quant, scale, gate, up, in_features))[:2])  # type: ignore[return-value]

@functools.cache
def _q8k_quantize_kernel(quant:UOp, scale:UOp, x:UOp, in_features:int) -> UOp:
  tokens, blocks = _concrete_int(quant.shape[0]), in_features // 256
  work_items = tokens * blocks
  core, job, work = _parallel_work(work_items, 1 if work_items <= 16 else 32)
  token, block = work // blocks, work % blocks
  best = UOp.placeholder((8,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  best = best.after(best.after(job).store(best.const_like(0)))
  chunk = UOp.range(32, 100)
  values = UOp.stack(*(x[token * in_features + block * 256 + chunk * 8 + lane].load().float() for lane in range(8)))
  previous = best.after(chunk)
  selected = (values.abs() > previous.abs()).where(values, previous)
  found = best.after(chunk).store(selected).end(chunk)
  signed_max = best.after(found)[0]
  for lane in range(1, 8):
    candidate = best.after(found)[lane]
    signed_max = (candidate.abs() > signed_max.abs()).where(candidate, signed_max)
  d = (signed_max.ne(0)).where(-signed_max / 127, UOp.const(dtypes.float32, 0))
  scale_store = scale[token, block].store(d)
  qchunk = UOp.range(32, 101)
  quant_stores = [quant[token, block, qchunk * 8 + lane].store(
    d.ne(0).where((x[token * in_features + block * 256 + qchunk * 8 + lane].load().float() / d).round(),
                   UOp.const(dtypes.float32, 0)).maximum(-127).minimum(127).cast(dtypes.int8)) for lane in range(8)]
  return UOp.group(scale_store, UOp.group(*quant_stores).end(qchunk)).end(job, core).sink(
    arg=KernelInfo(name="q8k_quantize_cpu", optimize=False, parallel=True))

def q8k_quantize(x:Tensor, in_features:int) -> tuple[Tensor, Tensor]:
  tokens, xc = int(x.numel()) // in_features, x.reshape(-1).contiguous()
  quant = Tensor.empty(tokens, in_features // 256, 256, dtype=dtypes.int8, device=x.device)
  scale = Tensor.empty(tokens, in_features // 256, dtype=dtypes.float32, device=x.device)
  return tuple(Tensor.custom_kernel(quant, scale, xc, fxn=lambda quant,scale,x:
    _q8k_quantize_kernel(quant, scale, x, in_features))[:2])  # type: ignore[return-value]

@functools.cache
def _q8k_q8_quantize_kernel(qk:UOp, dk:UOp, q:UOp, d:UOp, xk:UOp, x:UOp, in_features:int) -> UOp:
  qk_tokens, blocks = _concrete_int(qk.shape[0]), in_features // 256
  qk_work = UOp.range(qk_tokens * blocks, 90)
  qk_token, block = qk_work // blocks, qk_work % blocks
  best = UOp.placeholder((8,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  best = best.after(best.after(qk_work).store(best.const_like(0)))
  chunk = UOp.range(32, 100)
  values = UOp.stack(*(xk[qk_token * in_features + block * 256 + chunk * 8 + lane].load().float() for lane in range(8)))
  previous = best.after(chunk)
  selected = (values.abs() > previous.abs()).where(values, previous)
  found = best.after(chunk).store(selected).end(chunk)
  signed_max = best.after(found)[0]
  for lane in range(1, 8):
    candidate = best.after(found)[lane]
    signed_max = (candidate.abs() > signed_max.abs()).where(candidate, signed_max)
  qkd = (signed_max.ne(0)).where(-signed_max / 127, UOp.const(dtypes.float32, 0))
  qk_scale = dk[qk_token, block].store(qkd)
  qk_chunk = UOp.range(32, 101)
  qk_stores = [qk[qk_token, block, qk_chunk * 8 + lane].store(
    qkd.ne(0).where((xk[qk_token * in_features + block * 256 + qk_chunk * 8 + lane].load().float() / qkd).round(),
                    UOp.const(dtypes.float32, 0)).maximum(-127).minimum(127).cast(dtypes.int8)) for lane in range(8)]
  qk_done = UOp.group(qk_scale, UOp.group(*qk_stores).end(qk_chunk)).end(qk_work)

  q_tokens, groups = _concrete_int(q.shape[0]), in_features // 32
  q_work = UOp.range(q_tokens * groups, 91)
  token, group = q_work // groups, q_work % groups
  qvalues = tuple(x[token * in_features + group * 32 + lane].load().float() for lane in range(32))
  amax = functools.reduce(lambda a,b:a.maximum(b), (value.abs() for value in qvalues))
  qd = (amax / 127).maximum(1e-8)
  q_stores = [d[token, group].store(qd)] + [
    q[token, group, lane].store((value / qd).round().maximum(-127).minimum(127).cast(dtypes.int8))
    for lane,value in enumerate(qvalues)]
  q_done = UOp.group(*q_stores).end(q_work)
  return UOp.group(qk_done, q_done).sink(
    arg=KernelInfo(name=f"q8k_q8_quantize_cpu_{qk_tokens}_{q_tokens}_{in_features}", opts_to_apply=()))

def q8k_q8_quantize(xk:Tensor, x:Tensor, in_features:int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  qk_tokens, q_tokens = int(xk.numel()) // in_features, int(x.numel()) // in_features
  xkc, xc = xk.reshape(-1).contiguous(), x.reshape(-1).contiguous()
  qk = Tensor.empty(qk_tokens, in_features // 256, 256, dtype=dtypes.int8, device=x.device)
  dk = Tensor.empty(qk_tokens, in_features // 256, dtype=dtypes.float32, device=x.device)
  q = Tensor.empty(q_tokens, in_features // 32, 32, dtype=dtypes.int8, device=x.device)
  d = Tensor.empty(q_tokens, in_features // 32, dtype=dtypes.float32, device=x.device)
  return tuple(Tensor.custom_kernel(qk, dk, q, d, xkc, xc, fxn=lambda qk,dk,q,d,xk,x:
    _q8k_q8_quantize_kernel(qk, dk, q, d, xk, x, in_features))[:4])  # type: ignore[return-value]

def _load_f16(raw:UOp, offset:UOp) -> UOp:
  bits = raw[offset].load().cast(dtypes.uint16) | (raw[offset + 1].load().cast(dtypes.uint16) << 8)
  return bits.bitcast(dtypes.float16).float()

def _load_f16x8_ptr(raw:UOp) -> UOp:
  return _contiguous_vector_load(raw, 8, dtypes.float16).float()

def _vector_reg(reg:UOp, *deps:UOp) -> UOp:
  reg = reg.after(*deps)
  return UOp(Ops.SHRINK, src=(reg, UOp.const(dtypes.weakint, 0), UOp.const(dtypes.weakint, reg.max_numel())))

def _vector_acc_init(reg:UOp, *deps:UOp) -> UOp:
  return reg.after(_vector_reg(reg, *deps).store(reg.const_like(0)))

def _vector_acc_update(reg:UOp, value:UOp, *deps:UOp) -> UOp:
  previous = _vector_reg(reg, *deps).load()
  return _vector_reg(reg, *deps).store(previous + value)

def _finite_exp2(x:UOp) -> UOp:
  # The causal-convolution activation is finite. Bounding the exponent preserves sigmoid saturation while avoiding
  # the generic transcendental decomposition's NaN/Inf and overflow branches.
  x = x.maximum(-126.0).minimum(126.0)
  q = (x + (x < 0).where(-0.5, 0.5)).int()
  s = x - q.float()
  poly = UOp.const(dtypes.float32, 0.0001535920892)
  for coefficient in (0.001339262701, 0.009618384764, 0.05550347269, 0.2402264476, 0.6931471825, 1.0):
    poly = poly * s + coefficient
  half_q = q // 2
  pow0 = ((half_q + 127) << 23).bitcast(dtypes.float32)
  pow1 = ((q - half_q + 127) << 23).bitcast(dtypes.float32)
  return poly * pow0 * pow1

@functools.cache
def _cpu_expert_lut(device:str, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common
  if ggml_type == 21:
    sign_masks = [sum((0xff << (8*i)) for i in range(4) if signs & (1 << i)) for signs in range(16)]
    return Tensor([*ggml_common.iq3s_grid, *sign_masks], dtype=dtypes.uint32, device=device).contiguous().realize()
  values = [((ggml_common.kvalues_iq4nl[i] & 0xff) | ((ggml_common.kvalues_iq4nl[j] & 0xff) << 8)) for j in range(16) for i in range(16)]
  return Tensor(values, dtype=dtypes.uint16, device=device).contiguous().realize()

def _cpu_expert_group_dot(raw:UOp, lut:UOp, xq:UOp, xd:UOp, expert:UOp, xidx:UOp, group:UOp, output:UOp,
                          out_features:int, in_features:int, ggml_type:int, repacked:bool) -> UOp:
  type_size = _GGML_QUANT[ggml_type][1]
  block, subgroup = group // 8, group % 8
  if ggml_type == 21 and repacked:
    blocks = in_features // 256
    meta_size = (blocks * 6 + 63) // 64 * 64
    row_size = meta_size + blocks * 128
    base = (expert * out_features + output) * row_size
    meta, data = base + block * 6, base + meta_size + block * 128 + subgroup * 16
    parts = _dot_nibbles_ptr(raw[data], xq[xidx, block, subgroup * 32],
                             (1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15))
    dot = sum((parts[i] for i in range(8)), UOp.const(dtypes.int32, 0))
    scale_byte = raw[meta + 2 + subgroup // 2].load()
    scale = 1 + 2 * ((scale_byte >> (4 * (subgroup % 2)).cast(dtypes.uint8)) & 15).float()
    return dot.float() * scale * xd[xidx, block] * _load_f16(raw, meta)
  row_size = in_features // 256 * type_size
  base = (expert * out_features + output) * row_size + block * type_size
  qvalues:list[UOp] = []
  if ggml_type == 21:
    for word_idx in range(8):
      qi = raw[base + 2 + subgroup * 8 + word_idx].load().cast(dtypes.uint16) | \
           (((raw[base + 66 + subgroup].load() >> word_idx) & 1).cast(dtypes.uint16) << 8)
      values = lut[qi.cast(dtypes.weakint)].load()
      signs = raw[base + 74 + subgroup * 4 + word_idx // 2].load()
      for byte_idx in range(4):
        magnitude = ((values >> (8 * byte_idx)) & 255).cast(dtypes.int16)
        negative = ((signs >> (4 * (word_idx % 2) + byte_idx)) & 1).ne(0)
        qvalues.append(negative.where(-magnitude, magnitude).cast(dtypes.int8))
    scale_byte = raw[base + 106 + subgroup // 2].load()
    scale = 1 + 2 * ((scale_byte >> (4 * (subgroup % 2)).cast(dtypes.uint8)) & 15).float()
  elif ggml_type == 23:
    parts = _dot_nibbles_ptr(raw[base + 8 + subgroup * 16], xq[xidx, block, subgroup * 32],
                             (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113))
    dot = sum((parts[i] for i in range(8)), UOp.const(dtypes.int32, 0))
    low_byte = raw[base + 4 + subgroup // 2].load()
    low = (low_byte >> (4 * (subgroup % 2)).cast(dtypes.uint8)) & 15
    high_word = raw[base + 2].load().cast(dtypes.uint16) | (raw[base + 3].load().cast(dtypes.uint16) << 8)
    high = (high_word >> (2 * subgroup).cast(dtypes.uint16)) & 3
    scale = (low.cast(dtypes.uint16) | (high << 4)).cast(dtypes.uint8).bitcast(dtypes.int8).float() - 32
  else:
    assert ggml_type == 14
    for pos in range(32):
      full_pos, within = subgroup * 32 + pos, (subgroup * 32 + pos) % 128
      low_byte = raw[base + (full_pos // 128) * 64 + within % 64].load()
      low = (low_byte >> ((within // 64) * 4).cast(dtypes.uint8)) & 15
      high_byte = raw[base + 128 + (full_pos // 128) * 32 + within % 32].load()
      high = (high_byte >> ((within // 32) * 2).cast(dtypes.uint8)) & 3
      qvalues.append((low | (high << 4)).cast(dtypes.uint8).bitcast(dtypes.int8) - 32)
    scale = UOp.const(dtypes.float32, 1)
  xvalues = tuple(xq[xidx, block, subgroup * 32 + pos].load() if ggml_type in (21, 23) else
                  xq[xidx, group, pos].load() for pos in range(32))
  if ggml_type == 14:
    dot0, dot1 = _dot_bytes(tuple(qvalues[:16]), xvalues[:16]), _dot_bytes(tuple(qvalues[16:]), xvalues[16:])
    scales = tuple(raw[base + 192 + subgroup * 2 + i].load().bitcast(dtypes.int8).float() for i in range(2))
    return (dot0.float() * scales[0] + dot1.float() * scales[1]) * xd[xidx, group] * _load_f16(raw, base + 208)
  if ggml_type == 23: return dot.float() * scale * xd[xidx, block] * _load_f16(raw, base)
  return _dot_bytes(tuple(qvalues), xvalues).float() * scale * xd[xidx, block] * _load_f16(raw, base)

def _cpu_iq3_repacked_block_parts(raw:UOp, xq:UOp, expert:UOp, xidx:UOp, block:UOp, output:UOp,
                                  out_features:int, in_features:int) -> tuple[UOp, UOp]:
  blocks = in_features // 256
  meta_size = (blocks * 6 + 63) // 64 * 64
  row_size = meta_size + blocks * 128
  base = (expert * out_features + output) * row_size
  meta, data = base + block * 6, base + meta_size + block * 128
  total = UOp.stack(*(UOp.const(dtypes.int32, 0) for _ in range(8)))
  values = (1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15)
  for subgroup in range(0, 8, 2):
    parts0, parts1 = _dot_nibbles_pair_ptr(raw[data + subgroup * 16], xq[xidx, block, subgroup * 32], values)
    scale_byte = raw[meta + 2 + subgroup // 2].load()
    scale0, scale1 = 1 + 2 * (scale_byte & 15).int(), 1 + 2 * (scale_byte >> 4).int()
    total = total + parts0 * scale0 + parts1 * scale1
  return total, _load_f16(raw, meta)

def _cpu_iq4_block_parts(raw:UOp, xq:UOp, expert:UOp, xidx:UOp, block:UOp, output:UOp,
                         out_features:int, in_features:int) -> tuple[UOp, UOp]:
  row_size = in_features // 256 * _GGML_QUANT[23][1]
  base = (expert * out_features + output) * row_size + block * 136
  high_word = raw[base + 2].load().cast(dtypes.uint16) | (raw[base + 3].load().cast(dtypes.uint16) << 8)
  total = UOp.stack(*(UOp.const(dtypes.int32, 0) for _ in range(8)))
  values = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
  for subgroup in range(0, 8, 2):
    parts = _dot_nibbles_pair_ptr(raw[base + 8 + subgroup * 16], xq[xidx, block, subgroup * 32], values)
    low_byte = raw[base + 4 + subgroup // 2].load()
    low0, low1 = low_byte & 15, low_byte >> 4
    high0, high1 = (high_word >> (2 * subgroup)) & 3, (high_word >> (2 * (subgroup + 1))) & 3
    scale0 = (low0.cast(dtypes.uint16) | (high0 << 4)).cast(dtypes.uint8).bitcast(dtypes.int8).int() - 32
    scale1 = (low1.cast(dtypes.uint16) | (high1 << 4)).cast(dtypes.uint8).bitcast(dtypes.int8).int() - 32
    total = total + parts[0] * scale0 + parts[1] * scale1
  return total, _load_f16(raw, base)

def _cpu_q6_block_parts(raw:UOp, xq:UOp, xd:UOp, expert:UOp, xidx:UOp, block:UOp, output:UOp,
                        out_features:int, in_features:int) -> UOp:
  row_size = in_features // 256 * _GGML_QUANT[14][1]
  base = (expert * out_features + output) * row_size + block * 210
  total = UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8)))
  for subgroup in range(8):
    parts = _dot_q6_ptr(raw[base], xq[xidx, block * 8 + subgroup, 0], subgroup).float()
    scales = UOp.stack(*(raw[base + 192 + subgroup * 2 + i].load().bitcast(dtypes.int8).float() for i in range(2)))
    total = total + parts * UOp.stack(*(scales[i // 4] for i in range(8))) * xd[xidx, block * 8 + subgroup]
  return total * _load_f16(raw, base + 208)

@functools.cache
def _cpu_expert_kernel(out:UOp, raw:UOp, sel:UOp, xq:UOp, xd:UOp, lut:UOp,
                       out_features:int, in_features:int, ggml_type:int, routes_per_input:int, repacked:bool) -> UOp:
  routes, groups = _concrete_int(out.shape[0]), in_features // 32
  core, job, work = _parallel_work(routes * out_features)
  route, output = work // out_features, work % out_features
  expert, xidx = sel[route].load().cast(dtypes.weakint), route // routes_per_input
  vectorized = ggml_type in (14, 23) or (ggml_type == 21 and repacked)
  acc = UOp.placeholder((8 if vectorized else 1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = _vector_acc_init(acc, job) if vectorized else acc.after(acc.after(job).store(acc.const_like(0)))
  if ggml_type == 23:
    block = UOp.range(in_features // 256, 110)
    parts, scale = _cpu_iq4_block_parts(raw, xq, expert, xidx, block, output, out_features, in_features)
    value = parts.float() * xd[xidx, block] * scale
    accumulated = _vector_acc_update(acc, value, block).end(block)
  elif ggml_type == 21 and repacked:
    block = UOp.range(in_features // 256, 110)
    parts, scale = _cpu_iq3_repacked_block_parts(raw, xq, expert, xidx, block, output, out_features, in_features)
    value = parts.float() * xd[xidx, block] * scale
    accumulated = _vector_acc_update(acc, value, block).end(block)
  elif ggml_type == 14:
    block = UOp.range(in_features // 256, 110)
    value = _cpu_q6_block_parts(raw, xq, xd, expert, xidx, block, output, out_features, in_features)
    accumulated = _vector_acc_update(acc, value, block).end(block)
  else:
    group = UOp.range(groups, 110)
    value = _cpu_expert_group_dot(raw, lut, xq, xd, expert, xidx, group, output, out_features, in_features, ggml_type, repacked)
    accumulated = acc.after(group).store(acc.after(group) + value).end(group)
  result = sum((acc.after(accumulated)[i] for i in range(8)), UOp.const(dtypes.float32, 0)) if vectorized else \
    acc.after(accumulated)[0]
  return out[route, output].store(result.cast(out.dtype)).end(job, core).sink(
    arg=KernelInfo(name=f"expert_uop_cpu_{ggml_type}_{routes}_{out_features}_{in_features}", optimize=False, parallel=True))

def uop_expert(layer:ExpertWeights, sel:Tensor, x:Tensor, prepared:tuple[Tensor, Tensor]|None=None) -> Tensor:
  assert layer.ggml_type in (14, 21, 23)
  input_count, routes = int(x.numel()) // layer.in_features, int(sel.numel())
  routes_per_input = routes // input_count
  flat_sel = sel.flatten().contiguous()
  xq, xd = prepared if prepared is not None else \
    (q8k_quantize(x, layer.in_features) if layer.ggml_type in (21, 23) else q8_quantize(x, layer.in_features))
  out = Tensor.empty(routes, layer.out_features, dtype=x.dtype, device=x.device)
  lut = _cpu_expert_lut(str(x.device), layer.ggml_type)
  repacked = layer.ggml_type == 21 and layer.cpu_repacked is not None
  raw = layer.cpu_repacked if repacked else layer.weight
  assert raw is not None
  out = Tensor.custom_kernel(out, raw, flat_sel, xq, xd, lut, fxn=lambda out,raw,sel,xq,xd,lut:
    _cpu_expert_kernel(out, raw, sel, xq, xd, lut, layer.out_features, layer.in_features,
                       layer.ggml_type, routes_per_input, repacked))[0]
  return out if len(sel.shape) == 1 else out.reshape(*sel.shape, layer.out_features)

@functools.cache
def _cpu_expert_weighted_uop(out:UOp, raw:UOp, probs:UOp, sel:UOp, xq:UOp, xd:UOp,
                             out_features:int, in_features:int, ggml_type:int, routes_per_input:int) -> UOp:
  inputs = _concrete_int(out.shape[0])
  core, job, work = _parallel_work(inputs * out_features)
  input_idx, output = work // out_features, work % out_features
  total = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  total = total.after(total.after(job).store(total.const_like(0)))
  local_route = UOp.range(routes_per_input, 100)
  route = input_idx * routes_per_input + local_route
  expert = sel[route].load().cast(dtypes.weakint)
  acc = UOp.placeholder((8,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  acc = _vector_acc_init(acc, total, local_route)
  block = UOp.range(in_features // 256, 101)
  if ggml_type == 23:
    parts, scale = _cpu_iq4_block_parts(raw, xq, expert, route, block, output, out_features, in_features)
    value = parts.float() * xd[route, block] * scale
  else:
    assert ggml_type == 14
    value = _cpu_q6_block_parts(raw, xq, xd, expert, route, block, output, out_features, in_features)
  route_dot = _vector_acc_update(acc, value, block).end(block)
  route_value = sum((acc.after(route_dot)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  accumulated = total.after(local_route).store(total.after(local_route) + route_value * probs[route].load()).end(local_route)
  return out[input_idx, output].store(total.after(accumulated)[0].load()).end(job, core).sink(
    arg=KernelInfo(name=f"expert_weighted_uop_cpu_{ggml_type}_{inputs}_{routes_per_input}_{out_features}_{in_features}",
                   optimize=False, parallel=True))

@functools.cache
def _cpu_expert_weighted_grouped_uop(out:UOp, raw:UOp, probs:UOp, head:UOp, next_route:UOp, unique:UOp,
                                     unique_count:UOp, xq:UOp, xd:UOp, num_experts:int, out_features:int,
                                     in_features:int, routes_per_input:int) -> UOp:
  inputs, routes = _concrete_int(out.shape[0]), _concrete_int(next_route.shape[1])
  cores = math.gcd(out_features, 32)
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  output_tile = min(4, out_features // cores)
  assert out_features % (cores * output_tile) == 0
  job = UOp.range(out_features // (cores * output_tile), 90)
  output_base = core * (out_features // cores) + job * output_tile
  totals = UOp.placeholder((output_tile, inputs), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  clear = UOp.range(output_tile * inputs, 91)
  initialized = totals.after(job)[clear // inputs, clear % inputs].store(0.0).end(clear)
  cursor = UOp.placeholder((1,), dtypes.int32, slot=1, addrspace=AddrSpace.REG)
  matched_probs = UOp.placeholder((routes,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  route_dtype = dtypes.uint16 if routes <= 65535 else dtypes.uint32
  matched_routes = UOp.placeholder((routes,), route_dtype, slot=3, addrspace=AddrSpace.REG)

  unique_idx = UOp.range(unique_count[0].load().cast(dtypes.weakint).minimum(num_experts), 100)
  expert = unique[unique_idx].load().cast(dtypes.weakint)
  matched_count = head[expert].load().cast(dtypes.weakint).minimum(routes)
  match_reset = cursor.after(initialized, unique_idx)[0].store(0)
  match_loop = UOp.loop(110)
  match_loop = match_loop.replace(src=match_loop.src + (match_reset,))
  match_idx = cursor.after(match_loop)[0].load().cast(dtypes.weakint)
  matched_route = next_route[expert, match_idx].load().cast(dtypes.weakint)
  copied = UOp.group(matched_probs[match_idx].store(probs[matched_route].load()),
                     matched_routes[match_idx].store(matched_route.cast(route_dtype)))
  match_next = match_idx + 1
  routes_ready = cursor.after(copied)[0].store(match_next.int()).end(match_loop, match_next < matched_count)

  output_lane = UOp.range(output_tile, 101)
  output = output_base + output_lane
  block = UOp.range(in_features // 256, 102)
  base = (expert * out_features + output) * (in_features // 256 * 136) + block * 136
  high_word = raw[base + 2].load().cast(dtypes.uint16) | (raw[base + 3].load().cast(dtypes.uint16) << 8)
  values = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
  qvalues, scales = [], []
  for subgroup in range(8):
    packed = _contiguous_vector_load(raw[base + 8 + subgroup * 16], 16)
    qvalues.append(_unpack_nibbles(packed, values))
    low_byte = raw[base + 4 + subgroup // 2].load()
    low = (low_byte >> (4 * (subgroup % 2))) & 15
    high = (high_word >> (2 * subgroup)) & 3
    scales.append((low.cast(dtypes.uint16) | (high << 4)).cast(dtypes.uint8).bitcast(dtypes.int8).int() - 32)
  block_scale = _load_f16(raw, base)

  reset = cursor.after(routes_ready, output_lane, block)[0].store(0)
  route_loop = UOp.loop(111)
  route_loop = route_loop.replace(src=route_loop.src + (reset, block_scale, *qvalues, *scales))
  route_idx = cursor.after(route_loop)[0].load().cast(dtypes.weakint)
  route = matched_routes[route_idx].load().cast(dtypes.weakint)
  input_idx = route // routes_per_input
  block_acc = UOp.placeholder((8,), dtypes.int32, slot=4, addrspace=AddrSpace.REG)
  stage = _vector_reg(block_acc, route_loop).store(block_acc.const_like(0))
  for subgroup in range(8):
    xv = _contiguous_vector_load(xq[route, block, subgroup * 32], 32, dtypes.int8)
    stage = _vector_acc_update(block_acc, _dot_byte_vectors(qvalues[subgroup], xv, scales[subgroup]), stage)
  dot = sum((block_acc.after(stage)[i] for i in range(8)), UOp.const(dtypes.int32, 0)).float()
  contribution = dot * xd[route, block] * block_scale * matched_probs[route_idx].load()
  updated = totals[output_lane, input_idx].store(totals.after(route_loop)[output_lane, input_idx].load() + contribution)
  next_idx = route_idx + 1
  routes_done = cursor.after(updated)[0].store(next_idx.int()).end(route_loop, next_idx < matched_count)
  accumulated = routes_done.end(block, output_lane, unique_idx)

  output_store = UOp.range(output_tile * inputs, 120)
  store_lane, input_store = output_store // inputs, output_store % inputs
  return out[input_store, output_base + store_lane].store(totals.after(accumulated)[store_lane, input_store].load()).end(
    output_store, job, core).sink(
    arg=KernelInfo(name=f"expert_weighted_grouped_uop_cpu_23_{inputs}_{routes_per_input}_{out_features}_{in_features}",
                   optimize=False, parallel=True, estimates=Estimates(routes * out_features * in_features * 2)))

def uop_expert_weighted_sum(layer:ExpertWeights, sel:Tensor, x:Tensor, probs:Tensor,
                            route_links:tuple[Tensor, Tensor, Tensor, Tensor]|None=None) -> Tensor:
  routes, routes_per_input = int(sel.numel()), probs.shape[-1]
  inputs = routes // routes_per_input
  xq, xd = q8k_quantize(x, layer.in_features) if layer.ggml_type == 23 else q8_quantize(x, layer.in_features)
  flat_sel, flat_probs = sel.flatten().contiguous(), probs.flatten().contiguous()
  out = Tensor.empty(inputs, layer.out_features, dtype=x.dtype, device=x.device)
  if layer.ggml_type == 23 and routes > layer.num_experts:
    head, next_route, unique, unique_count = route_links or _expert_route_links(flat_sel, layer.num_experts)
    out = Tensor.custom_kernel(out, layer.weight, flat_probs, head, next_route, unique, unique_count, xq, xd,
      fxn=lambda out,raw,probs,head,next_route,unique,count,xq,xd:
        _cpu_expert_weighted_grouped_uop(out, raw, probs, head, next_route, unique, count, xq, xd,
                                         layer.num_experts, layer.out_features, layer.in_features, routes_per_input))[0]
  else:
    out = Tensor.custom_kernel(out, layer.weight, flat_probs, flat_sel, xq, xd, fxn=lambda out,raw,probs,sel,xq,xd:
      _cpu_expert_weighted_uop(out, raw, probs, sel, xq, xd, layer.out_features, layer.in_features,
                               layer.ggml_type, routes_per_input))[0]
  return out.reshape(*probs.shape[:-1], layer.out_features)

@functools.cache
def _expert_route_links_uop(head:UOp, next_route:UOp, unique:UOp, unique_count:UOp, sel:UOp, num_experts:int) -> UOp:
  routes = _concrete_int(next_route.shape[1])
  init = UOp.range(num_experts, 90)
  initialized = head[init].store(0).end(init)
  count = UOp.placeholder((1,), dtypes.int32, slot=0, addrspace=AddrSpace.REG)
  count = count.after(count.after(initialized).store(0))
  route = UOp.range(routes, 91)
  expert = sel.after(initialized)[route].load().cast(dtypes.weakint)
  expert_count = head.after(initialized, route)[expert].load()
  count_value = count.after(route)[0].load()
  is_new = expert_count.eq(0)
  saved_unique = unique[count_value.cast(dtypes.weakint).valid(is_new)].store(expert.int())
  advanced_count = count[0].store(count_value + is_new.int())
  slot = expert_count.maximum(0).minimum(routes - 1).cast(dtypes.weakint)
  linked = UOp.group(saved_unique, advanced_count, next_route[expert, slot].store(route.int()),
                     head[expert].store(expert_count + 1)).end(route)
  return unique_count[0].store(count.after(linked)[0].load()).sink(
    arg=KernelInfo(name=f"expert_route_links_{routes}_{num_experts}", opts_to_apply=()))

def _expert_route_links(sel:Tensor, num_experts:int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  flat_sel = sel.flatten().contiguous()
  head = Tensor.empty(num_experts, dtype=dtypes.int32, device=sel.device)
  next_route = Tensor.empty(num_experts, int(sel.numel()), dtype=dtypes.int32, device=sel.device)
  unique = Tensor.empty(num_experts, dtype=dtypes.int32, device=sel.device)
  unique_count = Tensor.empty(1, dtype=dtypes.int32, device=sel.device)
  return tuple(Tensor.custom_kernel(head, next_route, unique, unique_count, flat_sel,
    fxn=lambda head,next_route,unique,count,sel:
      _expert_route_links_uop(head, next_route, unique, count, sel, num_experts))[:4])  # type: ignore[return-value]

@functools.cache
def _cpu_expert_silu_grouped_uop(out:UOp, raw0:UOp, raw1:UOp, head:UOp, next_route:UOp, unique:UOp, unique_count:UOp,
                                 xq:UOp, xd:UOp, num_experts:int, out_features:int, in_features:int,
                                 routes_per_input:int) -> UOp:
  routes = _concrete_int(next_route.shape[1])
  cores = math.gcd(out_features, 32)
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  total = unique_count[0].load().cast(dtypes.weakint).minimum(num_experts) * out_features
  job = UOp.range(total // cores, 90)
  rows_per_core = out_features // cores
  unique_idx, output_value = job // rows_per_core, core * rows_per_core + job % rows_per_core
  accs = tuple(UOp.placeholder((routes,), dtypes.float32, slot=i, addrspace=AddrSpace.REG) for i in range(2))
  cursor = UOp.placeholder((1,), dtypes.int32, slot=2, addrspace=AddrSpace.REG)
  output_reg = UOp.placeholder((1,), dtypes.int32, slot=3, addrspace=AddrSpace.REG)
  expert_reg = UOp.placeholder((1,), dtypes.int32, slot=4, addrspace=AddrSpace.REG)
  matched = UOp.placeholder((routes,), dtypes.int32, slot=5, addrspace=AddrSpace.REG)
  expert_ready = expert_reg.after(expert_reg.after(job).store(unique[unique_idx].load()))
  expert = expert_reg.after(expert_ready)[0].load().cast(dtypes.weakint)
  matched_count = head.after(expert_ready)[expert].load().cast(dtypes.weakint).minimum(routes)
  output_ready = output_reg.after(output_reg.after(job, expert_ready).store(output_value))
  output = output_reg.after(output_ready)[0].load().cast(dtypes.weakint)

  cursor_ready = cursor.after(cursor.after(job, output_ready).store(0))
  init_loop = UOp.loop(111)
  init_loop = init_loop.replace(src=init_loop.src + (cursor_ready,))
  init_idx = cursor.after(init_loop)[0].load().cast(dtypes.weakint)
  cleared = UOp.group(*(acc[init_idx].store(UOp.const(dtypes.float32, 0)) for acc in accs),
                      matched[init_idx].store(next_route[expert, init_idx].load()))
  init_next = init_idx + 1
  initialized = cursor.after(cleared)[0].store(init_next.int()).end(init_loop, init_next < matched_count)

  blocks, meta_size = in_features // 256, (in_features // 256 * 6 + 63) // 64 * 64
  row_size = meta_size + blocks * 128
  row_base = (expert * out_features + output) * row_size
  block = UOp.range(blocks, 112)
  projection_done = initialized
  values = (1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15)
  for projection,raw in enumerate((raw0, raw1)):
    meta = row_base + block * 6
    raw = raw.after(projection_done)
    qvalues:list[UOp] = []
    scales:list[UOp] = []
    for subgroup in range(0, 8, 2):
      for offset in (0, 16):
        packed = _contiguous_vector_load(raw[row_base + meta_size + block * 128 + subgroup * 16 + offset], 16)
        qvalues.append(_unpack_nibbles(packed, values))
    for subgroup in range(8):
      scale_byte = raw[meta + 2 + subgroup // 2].load()
      scales.append(1 + 2 * ((scale_byte >> (4 * (subgroup % 2))) & 15).int())

    reset = cursor.after(projection_done, block)[0].store(0)
    route_loop = UOp.loop(120 + projection)
    route_loop = route_loop.replace(src=route_loop.src + (reset, *qvalues, *scales))
    route_idx = cursor.after(route_loop)[0].load().cast(dtypes.weakint)
    route = matched[route_idx].load().cast(dtypes.weakint)
    xidx = (route.cast(dtypes.uint32) // routes_per_input).cast(dtypes.weakint)
    block_acc = UOp.placeholder((8,), dtypes.int32, slot=6 + projection, addrspace=AddrSpace.REG)
    stage = _vector_reg(block_acc, route_loop).store(block_acc.const_like(0))
    for subgroup in range(8):
      xv = _contiguous_vector_load(xq[xidx, block, subgroup * 32], 32, dtypes.int8)
      parts = _dot_byte_vectors(qvalues[subgroup], xv, scales[subgroup])
      stage = _vector_acc_update(block_acc, parts, stage)
    block_sum = block_acc.after(stage)
    value = sum((block_sum[i] for i in range(8)), UOp.const(dtypes.int32, 0)).float() * \
      xd[xidx, block] * _load_f16(raw, meta)
    updated = accs[projection][route_idx].store(accs[projection].after(route_loop)[route_idx].load() + value)
    next_idx = route_idx + 1
    projection_done = cursor.after(updated)[0].store(next_idx.int()).end(route_loop, next_idx < matched_count)
  accumulated = projection_done.end(block)

  final_reset = cursor.after(accumulated)[0].store(0)
  final_loop = UOp.loop(130)
  final_loop = final_loop.replace(src=final_loop.src + (final_reset,))
  final_idx = cursor.after(final_loop)[0].load().cast(dtypes.weakint)
  final_route = matched[final_idx].load().cast(dtypes.weakint)
  final_output = output_reg.after(final_loop)[0].load().cast(dtypes.weakint)
  gate, up = (acc.after(final_loop)[final_idx].load() for acc in accs)
  saved = out[final_route, final_output].store((gate * gate.sigmoid() * up).cast(out.dtype))
  final_next = final_idx + 1
  routes_done = cursor.after(saved)[0].store(final_next.int()).end(final_loop, final_next < matched_count)
  return routes_done.end(job, core).sink(
    arg=KernelInfo(name=f"expert_silu_grouped_local_uop_cpu_{routes}_{out_features}_{in_features}",
                   optimize=False, parallel=True, estimates=Estimates(next_route.shape[0] * out_features * in_features * 8)))

def uop_expert_silu(first:ExpertWeights, second:ExpertWeights, sel:Tensor, x:Tensor,
                    route_links:tuple[Tensor, Tensor, Tensor, Tensor]|None=None) -> Tensor:
  assert first.ggml_type == second.ggml_type and first.ggml_type in (14, 21, 23)
  input_count, routes = int(x.numel()) // first.in_features, int(sel.numel())
  routes_per_input = routes // input_count
  xq, xd = q8k_quantize(x, first.in_features) if first.ggml_type in (21, 23) else q8_quantize(x, first.in_features)
  out = Tensor.empty(routes, first.out_features, dtype=x.dtype, device=x.device)
  repacked = first.ggml_type == 21 and first.cpu_repacked is not None and second.cpu_repacked is not None
  raw0, raw1 = (first.cpu_repacked, second.cpu_repacked) if repacked else (first.weight, second.weight)
  assert raw0 is not None and raw1 is not None
  flat_sel = sel.flatten().contiguous()
  if repacked and routes > first.num_experts:
    head, next_route, unique, unique_count = route_links or _expert_route_links(flat_sel, first.num_experts)
    out = Tensor.custom_kernel(out, raw0, raw1, head, next_route, unique, unique_count, xq, xd,
      fxn=lambda out,raw0,raw1,head,next_route,unique,count,xq,xd:
        _cpu_expert_silu_grouped_uop(out, raw0, raw1, head, next_route, unique, count, xq, xd,
                                     first.num_experts, first.out_features, first.in_features, routes_per_input))[0]
  else:
    return silu_mul(uop_expert(first, sel, x, (xq, xd)), uop_expert(second, sel, x, (xq, xd)))
  return out if len(sel.shape) == 1 else out.reshape(*sel.shape, first.out_features)

def uop_expert_silu_weighted(first:ExpertWeights, second:ExpertWeights, down:ExpertWeights,
                             sel:Tensor, x:Tensor, probs:Tensor) -> Tensor:
  assert first.num_experts == second.num_experts == down.num_experts
  route_links = _expert_route_links(sel, first.num_experts)
  hidden = uop_expert_silu(first, second, sel, x, route_links)
  return uop_expert_weighted_sum(down, sel, hidden, probs, route_links)

@functools.cache
def _moe_stage1_uop(rhidden:UOp, shidden:UOp, rgate:UOp, rup:UOp, sgate:UOp, sup:UOp, sel:UOp,
                    xkq:UOp, xkd:UOp, xq:UOp, xd:UOp, lut:UOp, dim:int, hidden:int, routes:int,
                    expert_repacked:bool, shared_repacked:bool) -> UOp:
  cores = math.gcd(routes * hidden, hidden, 32)
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  stores = []

  routed_begin, routed_end = routes * hidden * core // cores, routes * hidden * (core + 1) // cores
  routed_job = UOp.range(routed_end - routed_begin, 90)
  routed_work = routed_begin + routed_job
  route, output = routed_work // hidden, routed_work % hidden
  expert = sel[route].load().cast(dtypes.weakint)
  racc0 = UOp.placeholder((8 if expert_repacked else 1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  racc1 = UOp.placeholder((8 if expert_repacked else 1,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  rinit = UOp.group(*((_vector_reg(acc, routed_job).store(acc.const_like(0)) if expert_repacked else
                       acc.after(routed_job).store(acc.const_like(0))) for acc in (racc0, racc1)))
  if expert_repacked:
    rblock = UOp.range(dim // 256, 100)
    rparts0, rscale0 = _cpu_iq3_repacked_block_parts(
      rgate, xkq, expert, UOp.const(dtypes.weakint, 0), rblock, output, hidden, dim)
    rparts1, rscale1 = _cpu_iq3_repacked_block_parts(
      rup, xkq, expert, UOp.const(dtypes.weakint, 0), rblock, output, hidden, dim)
    rv0, rv1 = (rparts0.float() * xkd[0, rblock] * rscale0,
                 rparts1.float() * xkd[0, rblock] * rscale1)
    rdone = UOp.group(_vector_acc_update(racc0, rv0, rinit, rblock),
                      _vector_acc_update(racc1, rv1, rinit, rblock)).end(rblock)
  else:
    rgroup = UOp.range(dim // 32, 100)
    rv0 = _cpu_expert_group_dot(rgate, lut, xkq, xkd, expert, UOp.const(dtypes.weakint, 0), rgroup, output,
                                 hidden, dim, 21, False)
    rv1 = _cpu_expert_group_dot(rup, lut, xkq, xkd, expert, UOp.const(dtypes.weakint, 0), rgroup, output,
                                 hidden, dim, 21, False)
    rdone = UOp.group(racc0.after(rinit, rgroup).store(racc0.after(rgroup) + rv0),
                      racc1.after(rinit, rgroup).store(racc1.after(rgroup) + rv1)).end(rgroup)
  if expert_repacked:
    rg = sum((racc0.after(rdone)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
    ru = sum((racc1.after(rdone)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  else: rg, ru = racc0.after(rdone)[0], racc1.after(rdone)[0]
  stores.append(rhidden[route, output].store(rg * rg.sigmoid() * ru).end(routed_job))

  shared_begin, shared_end = hidden * core // cores, hidden * (core + 1) // cores
  shared_job = UOp.range(shared_end - shared_begin, 91)
  shared_output, groups = shared_begin + shared_job, dim // 32
  sacc0 = UOp.placeholder((8,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  sacc1 = UOp.placeholder((8,), dtypes.float32, slot=3, addrspace=AddrSpace.REG)
  sinit = UOp.group(*(_vector_reg(acc, shared_job).store(acc.const_like(0)) for acc in (sacc0, sacc1)))
  if shared_repacked and groups % 8 == 0:
    sblock = UOp.range(groups // 8, 101)
    svalues = []
    for acc,raw in ((sacc0, sgate), (sacc1, sup)):
      row_base = shared_output * groups * 34
      factors = _load_f16x8_ptr(raw[row_base + sblock * 16]) * UOp.stack(*(xd[0, sblock * 8 + lane].load() for lane in range(8)))
      block_sum = sum((_dot_bytes_ptr(raw[row_base + groups * 2 + (sblock * 8 + lane) * 32],
                                      xq[0, sblock * 8 + lane, 0]).float() * factors[lane]
                       for lane in range(8)), UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8))))
      svalues.append(_vector_acc_update(acc, block_sum, sinit, sblock))
    sdone = UOp.group(*svalues).end(sblock)
  else:
    sgroup = UOp.range(groups, 101)
    svalues = []
    for acc,raw in ((sacc0, sgate), (sacc1, sup)):
      row_base = shared_output * groups * 34
      scale_base = row_base + sgroup * (2 if shared_repacked else 34)
      weight_base = row_base + (groups * 2 + sgroup * 32 if shared_repacked else sgroup * 34 + 2)
      svalues.append(_vector_acc_update(acc,
        _dot_bytes_ptr(raw[weight_base], xq[0, sgroup, 0]).float() *
        _load_f16(raw, scale_base) * xd[0, sgroup], sinit, sgroup))
    sdone = UOp.group(*svalues).end(sgroup)
  sg = sum((sacc0.after(sdone)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  su = sum((sacc1.after(sdone)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  stores.append(shidden[shared_output].store(sg * sg.sigmoid() * su).end(shared_job))
  return UOp.group(*stores).end(core).sink(
    arg=KernelInfo(name=f"moe_stage1_uop_{routes}_{dim}_{hidden}", optimize=False, parallel=True))

@functools.cache
def _moe_stage2_uop(out:UOp, rdown:UOp, sdown:UOp, sel:UOp, probs:UOp, rhq:UOp, rhd:UOp,
                    shq:UOp, shd:UOp, shared_scale:UOp, lut:UOp, dim:int, hidden:int, routes:int,
                    down_type:int, shared_repacked:bool) -> UOp:
  cores = math.gcd(dim, 32)
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  begin, end = dim * core // cores, dim * (core + 1) // cores
  job = UOp.range(end - begin, 90)
  output = begin + job

  total = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  total = total.after(total.after(job).store(total.const_like(0)))
  route = UOp.range(routes, 91)
  racc = UOp.placeholder((8,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  racc = _vector_acc_init(racc, total, route)
  expert = sel[route].load().cast(dtypes.weakint)
  if down_type == 23:
    rblock = UOp.range(hidden // 256, 100)
    rparts, rscale = _cpu_iq4_block_parts(rdown, rhq, expert, route, rblock, output, dim, hidden)
    rv = rparts.float() * rhd[route, rblock] * rscale
    route_dot = _vector_acc_update(racc, rv, rblock).end(rblock)
  else:
    rblock = UOp.range(hidden // 256, 100)
    rv = _cpu_q6_block_parts(rdown, rhq, rhd, expert, route, rblock, output, dim, hidden)
    route_dot = _vector_acc_update(racc, rv, rblock).end(rblock)
  route_value = sum((racc.after(route_dot)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  routed = total.after(route).store(total.after(route) + route_value * probs[route].load()).end(route)

  sacc = UOp.placeholder((8,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  sacc = _vector_acc_init(sacc, routed)
  groups = hidden // 32
  row_base = output * groups * 34
  if shared_repacked and groups % 8 == 0:
    sblock = UOp.range(groups // 8, 101)
    factors = _load_f16x8_ptr(sdown[row_base + sblock * 16]) * UOp.stack(*(shd[0, sblock * 8 + lane].load() for lane in range(8)))
    block_sum = sum((_dot_bytes_ptr(sdown[row_base + groups * 2 + (sblock * 8 + lane) * 32],
                                    shq[0, sblock * 8 + lane, 0]).float() * factors[lane]
                     for lane in range(8)), UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8))))
    shared_done = _vector_acc_update(sacc, block_sum, sblock).end(sblock)
  else:
    sgroup = UOp.range(groups, 101)
    scale_base = row_base + sgroup * (2 if shared_repacked else 34)
    weight_base = row_base + (groups * 2 + sgroup * 32 if shared_repacked else sgroup * 34 + 2)
    shared_done = _vector_acc_update(sacc,
      _dot_bytes_ptr(sdown[weight_base], shq[0, sgroup, 0]).float() *
      _load_f16(sdown, scale_base) * shd[0, sgroup], sgroup).end(sgroup)
  shared = sum((sacc.after(shared_done)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  return out[output].store(total.after(shared_done)[0].load() + shared * shared_scale[0].load()).end(job, core).sink(
    arg=KernelInfo(name=f"moe_stage2_uop_{down_type}_{routes}_{dim}_{hidden}", optimize=False, parallel=True))

def uop_moe_ffn(block:FFNBlock, x:Tensor, probs:Tensor, sel:Tensor) -> Tensor:
  routes, dim, hidden = int(sel.numel()), block.config.dim, block.config.hidden_dim
  assert int(x.numel()) == dim and block.ffn_gate_exps.ggml_type == block.ffn_up_exps.ggml_type == 21
  assert block.ffn_down_exps.ggml_type in (14, 23)
  assert block.ffn_gate_exps.cpu_repacked is not None and block.ffn_up_exps.cpu_repacked is not None
  assert block.ffn_gate_shexp.cpu_repacked is not None and block.ffn_up_shexp.cpu_repacked is not None
  assert block.ffn_down_shexp.cpu_repacked is not None
  xc, flat_sel, flat_probs = x.reshape(1, dim).contiguous(), sel.flatten().contiguous(), probs.flatten().contiguous()
  xkq, xkd, xq, xd = q8k_q8_quantize(xc, xc, dim)
  rhidden = Tensor.empty(routes, hidden, dtype=dtypes.float32, device=x.device)
  shidden = Tensor.empty(hidden, dtype=dtypes.float32, device=x.device)
  lut21 = _cpu_expert_lut(str(x.device), 21)
  stage1 = Tensor.custom_kernel(rhidden, shidden, block.ffn_gate_exps.cpu_repacked, block.ffn_up_exps.cpu_repacked,
    block.ffn_gate_shexp.cpu_repacked, block.ffn_up_shexp.cpu_repacked, flat_sel, xkq, xkd, xq, xd, lut21,
    fxn=lambda rhidden,shidden,rgate,rup,sgate,sup,sel,xkq,xkd,xq,xd,lut:
      _moe_stage1_uop(rhidden, shidden, rgate, rup, sgate, sup, sel, xkq, xkd, xq, xd, lut,
                      dim, hidden, routes, True, True))
  rhidden, shidden = stage1[0], stage1[1]
  if block.ffn_down_exps.ggml_type == 23: rhq, rhd, shq, shd = q8k_q8_quantize(rhidden, shidden, hidden)
  else:
    rhq, rhd = q8_quantize(rhidden, hidden)
    shq, shd = q8_quantize(shidden, hidden)
  shared_scale = shared_gate(xc, block.ffn_gate_inp_shexp["weight"]).reshape(1)
  out = Tensor.empty(dim, dtype=dtypes.float32, device=x.device)
  lut_down = _cpu_expert_lut(str(x.device), block.ffn_down_exps.ggml_type)
  out = Tensor.custom_kernel(out, block.ffn_down_exps.weight, block.ffn_down_shexp.cpu_repacked, flat_sel, flat_probs,
    rhq, rhd, shq, shd, shared_scale, lut_down, fxn=lambda out,rdown,sdown,sel,probs,rhq,rhd,shq,shd,scale,lut:
      _moe_stage2_uop(out, rdown, sdown, sel, probs, rhq, rhd, shq, shd, scale, lut, dim, hidden, routes,
                      block.ffn_down_exps.ggml_type, True))[0]
  return out.reshape(x.shape)

@functools.cache
def _q8_linear_uop(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int, repacked:bool=False) -> UOp:
  out_tokens = _concrete_int(out.shape[0])
  token_tile = 8 if out_tokens % 8 == 0 else 1
  token_blocks = out_tokens // token_tile
  core_limit = 32 if out_tokens > 1 else getenv("CPU_Q8_UOP_CORES", 32)
  core, job, work = _parallel_work(token_blocks * out_features, core_limit)
  token_block, output = work // out_features, work % out_features
  tokens, groups = tuple(token_block * token_tile + i for i in range(token_tile)), in_features // 32
  accs = tuple(UOp.placeholder((8,), dtypes.float32, slot=i, addrspace=AddrSpace.REG) for i in range(token_tile))
  accs = tuple(_vector_acc_init(acc, job) for acc in accs)
  row_base = output * groups * 34
  if repacked:
    block = UOp.range(groups // 8, 100)
    scales = _load_f16x8_ptr(raw[row_base + block * 16])
    updates = []
    for acc,token in zip(accs, tokens):
      factors = scales * UOp.stack(*(xd[token, block * 8 + lane].load() for lane in range(8)))
      if token_tile == 1:
        block_sum = sum((_dot_bytes_ptr(raw[row_base + groups * 2 + (block * 8 + lane) * 32],
                                        xq[token, block * 8 + lane, 0]).float() * factors[lane]
                         for lane in range(8)), UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8))))
        updates.append(_vector_acc_update(acc, block_sum, block))
      else:
        previous = block
        for lane in range(8):
          value = _dot_bytes_ptr(raw[row_base + groups * 2 + (block * 8 + lane) * 32],
                                 xq[token, block * 8 + lane, 0]).float() * factors[lane]
          previous = _vector_acc_update(acc, value, previous)
        updates.append(previous)
    done = UOp.group(*updates).end(block)
  else:
    group = UOp.range(groups, 100)
    weight_scale = _load_f16(raw, row_base + group * 34)
    updates = [_vector_acc_update(acc,
      _dot_bytes_ptr(raw[row_base + group * 34 + 2], xq[token, group, 0]).float() *
      weight_scale * xd[token, group], group) for acc,token in zip(accs, tokens)]
    done = UOp.group(*updates).end(group)
  stores = [out[token, output].store(sum((acc.after(done)[i] for i in range(8)),
                                         UOp.const(dtypes.float32, 0)).cast(out.dtype)) for acc,token in zip(accs, tokens)]
  return UOp.group(*stores).end(job, core).sink(
    arg=KernelInfo(name=f"linear_q8_cpu_{out_features}_{in_features}{'_repacked' if repacked else ''}",
                   optimize=False, parallel=True))

def uop_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in (8, 14)
  tokens, xc = int(x.numel()) // layer.in_features, x.reshape(-1, layer.in_features).contiguous()
  xq, xd = q8_quantize(xc, layer.in_features)
  if layer.ggml_type == 14:
    weight = ggml_data_to_tensor(layer.weight, layer.out_features * layer.in_features, 14,
                                 contiguous=False).reshape(layer.out_features, layer.in_features)
    activation = (xq.float() * xd.unsqueeze(-1)).reshape(tokens, layer.in_features)
    return (activation @ weight.T).reshape(*x.shape[:-1], layer.out_features)
  out = Tensor.empty(tokens, layer.out_features, dtype=x.dtype, device=x.device)
  repacked = tokens == 1 and layer.in_features % 256 == 0 and layer.cpu_repacked is not None
  raw = layer.cpu_repacked if repacked else layer.weight
  assert raw is not None
  out = Tensor.custom_kernel(out, raw, xq, xd, fxn=lambda out,raw,xq,xd:
    _q8_linear_uop(out, raw, xq, xd, layer.out_features, layer.in_features, repacked))[0]
  return out.reshape(*x.shape[:-1], layer.out_features)

def uop_q8_prequant_linear(layer:Linear, xq:Tensor, xd:Tensor, out_dtype:DType=dtypes.float16) -> Tensor:
  assert layer.ggml_type == 8 and xq.dtype == dtypes.int8 and xd.dtype == dtypes.float32
  tokens = xq.shape[0]
  assert int(xq.numel()) == tokens * layer.in_features and int(xd.numel()) == tokens * layer.in_features // 32
  out = Tensor.empty(tokens, layer.out_features, dtype=out_dtype, device=xq.device)
  repacked = tokens == 1 and layer.in_features % 256 == 0 and layer.cpu_repacked is not None
  raw = layer.cpu_repacked if repacked else layer.weight
  assert raw is not None
  return Tensor.custom_kernel(out, raw, xq, xd, fxn=lambda out,raw,xq,xd:
    _q8_linear_uop(out, raw, xq, xd, layer.out_features, layer.in_features, repacked))[0]

@functools.cache
def _q8_linear_pair_uop(out0:UOp, out1:UOp, raw0:UOp, raw1:UOp, xq:UOp, xd:UOp,
                        out_features0:int, out_features1:int, in_features:int, repacked:bool) -> UOp:
  cores = math.gcd(out_features0, out_features1, 32)
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  projection_stores = []
  for projection,(out,raw,out_features) in enumerate(((out0, raw0, out_features0), (out1, raw1, out_features1))):
    begin, end = out_features * core // cores, out_features * (core + 1) // cores
    job = UOp.range(end - begin, 90 + projection)
    output, groups = begin + job, in_features // 32
    acc = UOp.placeholder((8,), dtypes.float32, slot=projection, addrspace=AddrSpace.REG)
    acc = _vector_acc_init(acc, job)
    row_base = output * groups * 34
    if repacked and groups % 8 == 0:
      block = UOp.range(groups // 8, 100 + projection)
      factors = _load_f16x8_ptr(raw[row_base + block * 16]) * UOp.stack(*(xd[0, block * 8 + lane].load() for lane in range(8)))
      block_sum = sum((_dot_bytes_ptr(raw[row_base + groups * 2 + (block * 8 + lane) * 32],
                                      xq[0, block * 8 + lane, 0]).float() * factors[lane]
                       for lane in range(8)), UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8))))
      done = _vector_acc_update(acc, block_sum, block).end(block)
    else:
      group = UOp.range(groups, 100 + projection)
      scale_base = row_base + group * (2 if repacked else 34)
      weight_base = row_base + (groups * 2 + group * 32 if repacked else group * 34 + 2)
      done = _vector_acc_update(acc,
        _dot_bytes_ptr(raw[weight_base], xq[0, group, 0]).float() *
        _load_f16(raw, scale_base) * xd[0, group], group).end(group)
    projection_stores.append(out[0, output].store(sum((acc.after(done)[i] for i in range(8)),
                                                       UOp.const(dtypes.float32, 0)).cast(out.dtype)).end(job))
  return UOp.group(*projection_stores).end(core).sink(
    arg=KernelInfo(name=f"linear_pair_q8_cpu_{out_features0}_{out_features1}_{in_features}", optimize=False, parallel=True))

def uop_q8_linear_pair(first:Linear, second:Linear, x:Tensor) -> tuple[Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features and int(x.numel()) == first.in_features
  xc = x.reshape(1, first.in_features).contiguous()
  xq, xd = q8_quantize(xc, first.in_features)
  out0 = Tensor.empty(1, first.out_features, dtype=x.dtype, device=x.device)
  out1 = Tensor.empty(1, second.out_features, dtype=x.dtype, device=x.device)
  repacked = first.cpu_repacked is not None and second.cpu_repacked is not None
  raw0, raw1 = (first.cpu_repacked, second.cpu_repacked) if repacked else (first.weight, second.weight)
  assert raw0 is not None and raw1 is not None
  outputs = Tensor.custom_kernel(out0, out1, raw0, raw1, xq, xd, fxn=lambda out0,out1,raw0,raw1,xq,xd:
    _q8_linear_pair_uop(out0, out1, raw0, raw1, xq, xd, first.out_features, second.out_features, first.in_features, repacked))[:2]
  shape = x.shape[:-1]
  return outputs[0].reshape(*shape, first.out_features), outputs[1].reshape(*shape, second.out_features)

@functools.cache
def _q8_gdn_projections_uop(out0:UOp, out1:UOp, out2:UOp, raw0:UOp, raw1:UOp, xq:UOp, xd:UOp,
                            x:UOp, weight:UOp, out_features0:int, out_features1:int, in_features:int, repacked:bool) -> UOp:
  cores = math.gcd(out_features0, out_features1, out2.shape[0], getenv("CPU_GDN_UOP_CORES", 32))
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  x, weight = x.flatten(), weight.flatten()
  stores = []
  for projection,(out,raw,out_features) in enumerate(((out0, raw0, out_features0), (out1, raw1, out_features1))):
    begin, end = out_features * core // cores, out_features * (core + 1) // cores
    job = UOp.range(end - begin, 90 + projection)
    output, groups = begin + job, in_features // 32
    acc = UOp.placeholder((8,), dtypes.float32, slot=projection, addrspace=AddrSpace.REG)
    acc = _vector_acc_init(acc, job)
    row_base = output * groups * 34
    if repacked and groups % 8 == 0:
      block = UOp.range(groups // 8, 100 + projection)
      scales = _load_f16x8_ptr(raw[row_base + block * 16])
      factors = scales * UOp.stack(*(xd[0, block * 8 + lane].load() for lane in range(8)))
      block_sum = sum((_dot_bytes_ptr(raw[row_base + groups * 2 + (block * 8 + lane) * 32],
                                      xq[0, block * 8 + lane, 0]).float() * factors[lane]
                       for lane in range(8)), UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8))))
      done = _vector_acc_update(acc, block_sum, block).end(block)
    else:
      group = UOp.range(groups, 100 + projection)
      scale_base = row_base + group * (2 if repacked else 34)
      weight_base = row_base + (groups * 2 + group * 32 if repacked else group * 34 + 2)
      done = _vector_acc_update(acc,
        _dot_bytes_ptr(raw[weight_base], xq[0, group, 0]).float() *
        _load_f16(raw, scale_base) * xd[0, group], group).end(group)
    stores.append(out[output].store(sum((acc.after(done)[i] for i in range(8)),
                                        UOp.const(dtypes.float32, 0)).cast(out.dtype)).end(job))

  f16_begin, f16_end = out2.shape[0] * core // cores, out2.shape[0] * (core + 1) // cores
  f16_job = UOp.range(f16_end - f16_begin, 92)
  f16_output = f16_begin + f16_job
  f16_acc = UOp.placeholder((8,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  f16_acc = _vector_acc_init(f16_acc, f16_job)
  chunk = UOp.range(in_features // 8, 102)
  xv = UOp.stack(*(x[chunk * 8 + lane].load().float() for lane in range(8)))
  wv = UOp.stack(*(weight[f16_output * in_features + chunk * 8 + lane].load().float() for lane in range(8)))
  f16_done = _vector_acc_update(f16_acc, xv * wv, chunk).end(chunk)
  stores.append(out2[f16_output].store(sum((f16_acc.after(f16_done)[i] for i in range(8)),
                                           UOp.const(dtypes.float32, 0)).cast(out2.dtype)).end(f16_job))
  return UOp.group(*stores).end(core).sink(
    arg=KernelInfo(name=f"q8_gdn_projections_uop_{out_features0}_{out_features1}_{in_features}", optimize=False, parallel=True))

def uop_q8_gdn_projections(first:Linear, second:Linear, f16_weight:Tensor, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  xc = x.reshape(-1).contiguous()
  xq, xd = q8_quantize(xc, first.in_features)
  out0 = Tensor.empty(first.out_features, dtype=x.dtype, device=x.device)
  out1 = Tensor.empty(second.out_features, dtype=x.dtype, device=x.device)
  out2 = Tensor.empty(f16_weight.shape[0], dtype=x.dtype, device=x.device)
  repacked = first.cpu_repacked is not None and second.cpu_repacked is not None
  raw0, raw1 = (first.cpu_repacked, second.cpu_repacked) if repacked else (first.weight, second.weight)
  assert raw0 is not None and raw1 is not None
  outputs = Tensor.custom_kernel(out0, out1, out2, raw0, raw1, xq, xd, xc, f16_weight,
    fxn=lambda out0,out1,out2,raw0,raw1,xq,xd,x,weight:
      _q8_gdn_projections_uop(out0, out1, out2, raw0, raw1, xq, xd, x, weight, first.out_features,
                              second.out_features, first.in_features, repacked))
  shape = x.shape[:-1]
  return (outputs[0].reshape(*shape, first.out_features), outputs[1].reshape(*shape, second.out_features),
          outputs[2].reshape(*shape, f16_weight.shape[0]))

def uop_q8_gdn_norm_projections(first:Linear, second:Linear, f16_weight:Tensor, x:Tensor,
                                norm:nn.RMSNorm) -> tuple[Tensor, Tensor, Tensor]:
  assert norm.weight is not None
  normed, xq, xd = rmsnorm_q8_quantize(x, norm.weight, norm.eps, first.in_features)
  out0 = Tensor.empty(first.out_features, dtype=dtypes.float16, device=x.device)
  out1 = Tensor.empty(second.out_features, dtype=dtypes.float16, device=x.device)
  out2 = Tensor.empty(f16_weight.shape[0], dtype=dtypes.float16, device=x.device)
  assert first.cpu_repacked is not None and second.cpu_repacked is not None
  outputs = Tensor.custom_kernel(out0, out1, out2, first.cpu_repacked, second.cpu_repacked, xq, xd, normed, f16_weight,
    fxn=lambda out0,out1,out2,raw0,raw1,xq,xd,x,weight:
      _q8_gdn_projections_uop(out0, out1, out2, raw0, raw1, xq, xd, x, weight, first.out_features,
                              second.out_features, first.in_features, True))
  shape = x.shape[:-1]
  return (outputs[0].reshape(*shape, first.out_features), outputs[1].reshape(*shape, second.out_features),
          outputs[2].reshape(*shape, f16_weight.shape[0]))

def recurrent_decode_bucket(pos:int, max_context:int, device:str) -> int:
  short_decode_len = min(8192, max_context)
  # Fused CPU decode receives the full KV cache and applies start_pos itself, so one graph covers every position.
  # The short key is only a JIT cache identifier on CPU; it does not window or truncate attention.
  return short_decode_len if device.startswith("CPU") or pos < short_decode_len else max_context

def _iq3_repack_uop(out:UOp, raw:UOp, grid:UOp, rows:int, blocks_per_row:int, meta_size:int) -> UOp:
  raw_row_size, row_size = blocks_per_row * 110, meta_size + blocks_per_row * 128
  core, job, idx = _parallel_work(rows * row_size)
  row, within = idx // row_size, idx % row_size

  meta_valid = within < blocks_per_row * 6
  meta_block, meta_byte = within // 6, within % 6
  meta_offset = (meta_byte < 2).where(meta_byte, meta_byte + 104)
  meta_value = raw[(row * raw_row_size + meta_block * 110 + meta_offset).valid(meta_valid)].load()

  data_valid, data_idx = within >= meta_size, within - meta_size
  block, block_idx = data_idx // 128, data_idx % 128
  subgroup, packed_pos = block_idx // 16, block_idx % 16
  base = row * raw_row_size + block * 110
  def code(pos:UOp) -> UOp:
    word, byte = pos // 4, (pos % 4).cast(dtypes.uint32)
    q = raw[(base + 2 + subgroup * 8 + word).valid(data_valid)].load().cast(dtypes.uint16)
    qh = raw[(base + 66 + subgroup).valid(data_valid)].load()
    grid_idx = q | ((((qh >> word.cast(dtypes.uint8)) & 1).cast(dtypes.uint16)) << 8)
    magnitude = (grid[grid_idx.valid(data_valid)].load() >> (byte * 8)) & 255
    sign_byte = raw[(base + 74 + subgroup * 4 + pos // 8).valid(data_valid)].load()
    return ((magnitude - 1) >> 1) | (((sign_byte >> (pos % 8).cast(dtypes.uint8)) & 1).cast(dtypes.uint32) << 3)
  data_value = (code(packed_pos) | (code(packed_pos + 16) << 4)).cast(dtypes.uint8)
  value = data_valid.where(data_value, meta_valid.where(meta_value, UOp.const(dtypes.uint8, 0)))
  return out[idx].store(value).end(job, core).sink(
    arg=KernelInfo(name=f"cpu_uop_iq3_repack_{rows}_{blocks_per_row}", optimize=False, parallel=True))

def iq3_repack(raw:Tensor, rows:int, in_features:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common
  assert raw.dtype == dtypes.uint8 and in_features % 256 == 0
  blocks_per_row = in_features // 256
  assert int(raw.numel()) == rows * blocks_per_row * 110
  meta_size = (blocks_per_row * 6 + 63) // 64 * 64
  out = Tensor.empty(rows * (meta_size + blocks_per_row * 128), dtype=dtypes.uint8, device=raw.device)
  grid = Tensor(ggml_common.iq3s_grid, dtype=dtypes.uint32, device=raw.device)
  return Tensor.custom_kernel(out, raw.flatten(), grid,
    fxn=lambda out,raw,grid:_iq3_repack_uop(out, raw, grid, rows, blocks_per_row, meta_size))[0]

def _q8_repack_uop(out:UOp, raw:UOp, rows:int, groups:int) -> UOp:
  row_size = groups * 34
  core, job, idx = _parallel_work(rows * row_size)
  row, within = idx // row_size, idx % row_size
  is_scale = within < groups * 2
  group = is_scale.where(within // 2, (within - groups * 2) // 32)
  byte = is_scale.where(within % 2, (within - groups * 2) % 32 + 2)
  return out[idx].store(raw[row * row_size + group * 34 + byte].load()).end(job, core).sink(
    arg=KernelInfo(name=f"cpu_uop_q8_repack_{rows}_{groups}", optimize=False, parallel=True))

def q8_repack(raw:Tensor, rows:int, in_features:int) -> Tensor:
  assert raw.dtype == dtypes.uint8 and in_features % 32 == 0 and int(raw.numel()) == rows * (in_features // 32) * 34
  groups = in_features // 32
  out = Tensor.empty(rows * groups * 34, dtype=dtypes.uint8, device=raw.device)
  return Tensor.custom_kernel(out, raw.flatten(), fxn=lambda out,raw:_q8_repack_uop(out, raw, rows, groups))[0]

def attention_decode(q:Tensor, cache:Tensor, start_pos:int|UOp) -> Tensor:
  batch, heads, query_len, head_dim = q.shape
  assert query_len == 1 and q.dtype == dtypes.float32 and cache.dtype == dtypes.float16
  cache_len = cache.shape[3]
  assert isinstance(cache_len, int) and head_dim % 8 == 0
  out = Tensor.empty(batch, heads, 1, head_dim, dtype=dtypes.float32, device=q.device)
  pos = UOp.variable("cpu_attention_pos", 0, cache_len-1).bind(start_pos) if isinstance(start_pos, int) else start_pos
  pos_var = pos.unbind()[0]
  srcs = (out.uop, q.contiguous().uop, cache.uop)
  params = [UOp.placeholder_like(x, slot=i) for i,x in enumerate(srcs)]
  kernel = _attention_decode_uop(params[0], params[1], params[2], pos_var + 1)
  return Tensor(srcs[0].after(kernel.call(*srcs, pos)))

@functools.cache
def _attention_decode_uop(out:UOp, q:UOp, cache:UOp, valid_len:UOp) -> UOp:
  batch, heads, _, dim = map(_concrete_int, q.shape)
  _, _, kv_heads, cache_len, _ = map(_concrete_int, cache.shape)
  assert heads % kv_heads == 0 and dim % 8 == 0
  chunks, log2e = dim // 8, math.log2(math.e)
  outf, qf, cachef = out.flatten(), q.flatten(), cache.flatten()
  bh = UOp.range(batch * heads, 0, AxisType.GLOBAL)
  bi, kv_head = bh // heads, (bh % heads) // (heads // kv_heads)

  numerator = UOp.placeholder((dim,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  row_max = UOp.placeholder((1,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  row_sum = UOp.placeholder((1,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  initialized = UOp.group(numerator.after(bh).store(numerator.const_like(0)),
                          row_max.after(bh).store(row_max.const_like(-math.inf)),
                          row_sum.after(bh).store(row_sum.const_like(0)))
  position = UOp.range(valid_len, 90)
  qk_acc = UOp.placeholder((8,), dtypes.float32, slot=3, addrspace=AddrSpace.REG)
  qk_acc = qk_acc.after(qk_acc.after(initialized, position).store(qk_acc.const_like(0)))
  chunk = UOp.range(chunks, 91)
  qv = UOp.stack(*(qf[bh * dim + chunk * 8 + lane].load() for lane in range(8)))
  kbase = ((bi * kv_heads + kv_head) * cache_len + position) * dim
  kval = UOp.stack(*(cachef[kbase + chunk * 8 + lane].load().float() for lane in range(8)))
  qk_done = qk_acc.after(chunk).store(qk_acc.after(chunk) + qv * kval).end(chunk)
  score = sum((qk_acc.after(qk_done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0)) / math.sqrt(dim)
  previous_max, previous_sum = row_max.after(qk_done)[0].load(), row_sum.after(qk_done)[0].load()
  next_max = previous_max.maximum(score)
  old_scale, weight = ((previous_max - next_max) * log2e).exp2(), ((score - next_max) * log2e).exp2()

  value_chunk = UOp.range(chunks, 92)
  vbase = batch * kv_heads * cache_len * dim + ((bi * kv_heads + kv_head) * cache_len + position) * dim
  updates = [numerator[value_chunk * 8 + lane].store(
    numerator.after(position)[value_chunk * 8 + lane].load() * old_scale +
    cachef[vbase + value_chunk * 8 + lane].load().float() * weight) for lane in range(8)]
  values_done = UOp.group(*updates).end(value_chunk)
  update = UOp.group(values_done, row_max[0].store(next_max), row_sum[0].store(previous_sum * old_scale + weight)).end(position)

  output = UOp.range(dim, 93)
  return outf[bh * dim + output].store(numerator.after(update)[output].load() / row_sum.after(update)[0].load()).end(output, bh).sink(
    arg=KernelInfo(name=f"attention_decode_uop_{batch}_{heads}_{kv_heads}_{dim}_{cache_len}", optimize=False, parallel=True))

@functools.cache
def _attention_prefill_online_uop(out:UOp, q:UOp, cache:UOp, start_pos:UOp) -> UOp:
  batch, heads, tokens, dim = map(_concrete_int, q.shape)
  _, _, kv_heads, cache_len, _ = map(_concrete_int, cache.shape)
  assert heads % kv_heads == 0 and dim % 8 == 0
  chunks, log2e = dim // 8, math.log2(math.e)
  token_tile = 4 if tokens % 4 == 0 else 1
  query_blocks = tokens // token_tile
  outf, qf, cachef = out.flatten(), q.flatten(), cache.flatten()
  core, job, work = _parallel_work(batch * heads * query_blocks)
  token_block, bh = work % query_blocks, work // query_blocks
  token_base = token_block * token_tile
  bi, kv_head = bh // heads, (bh % heads) // (heads // kv_heads)

  numerators = tuple(UOp.placeholder((dim,), dtypes.float32, slot=i, addrspace=AddrSpace.REG) for i in range(token_tile))
  row_maxes = tuple(UOp.placeholder((1,), dtypes.float32, slot=token_tile+i, addrspace=AddrSpace.REG) for i in range(token_tile))
  row_sums = tuple(UOp.placeholder((1,), dtypes.float32, slot=2*token_tile+i, addrspace=AddrSpace.REG) for i in range(token_tile))
  clear = UOp.range(chunks, 90)
  cleared = UOp.group(*(_contiguous_vector_ptr(numerator.after(job), clear * 8, 8).store(
    UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8)))) for numerator in numerators)).end(clear)
  initialized = UOp.group(cleared,
    *(row_max.after(job).store(-math.inf) for row_max in row_maxes),
    *(row_sum.after(job).store(0.0) for row_sum in row_sums))

  position = UOp.range((start_pos + tokens).minimum(cache_len), 100)
  qk_accs = tuple(UOp.placeholder((8,), dtypes.float32, slot=3*token_tile+i, addrspace=AddrSpace.REG) for i in range(token_tile))
  qk_accs = tuple(_vector_acc_init(acc, initialized, position) for acc in qk_accs)
  qchunk = UOp.range(chunks, 101)
  kbase = ((bi * kv_heads + kv_head) * cache_len + position) * dim + qchunk * 8
  kval = _load_f16x8_ptr(cachef[kbase])
  qk_updates = []
  for token_offset,acc in enumerate(qk_accs):
    qbase = (bh * tokens + token_base + token_offset) * dim + qchunk * 8
    qk_updates.append(_vector_acc_update(acc, _contiguous_vector_load(qf[qbase], 8) * kval, qchunk))
  qk_done = UOp.group(*qk_updates).end(qchunk)

  next_maxes, old_scales, weights = [], [], []
  for token_offset,(acc,row_max) in enumerate(zip(qk_accs, row_maxes)):
    score = sum((acc.after(qk_done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0)) / math.sqrt(dim)
    valid = position < start_pos + token_base + token_offset + 1
    previous_max = row_max.after(qk_done)[0].load()
    next_max = valid.where(previous_max.maximum(score), previous_max)
    next_maxes.append(next_max)
    old_scales.append(((previous_max - next_max) * log2e).exp2())
    weights.append(valid.where(((score - next_max) * log2e).exp2(), UOp.const(dtypes.float32, 0)))

  vchunk = UOp.range(chunks, 102)
  vbase = batch * kv_heads * cache_len * dim + ((bi * kv_heads + kv_head) * cache_len + position) * dim + vchunk * 8
  values = _load_f16x8_ptr(cachef[vbase])
  numerator_updates = []
  for numerator,old_scale,weight in zip(numerators, old_scales, weights):
    nbase = vchunk * 8
    previous = _contiguous_vector_load(numerator.after(initialized, position)[nbase], 8)
    updated = previous * old_scale + values * weight
    numerator_updates.append(_contiguous_vector_ptr(numerator.after(qk_done), nbase, 8).store(updated))
  values_done = UOp.group(*numerator_updates).end(vchunk)
  state_updates = [row_max[0].store(next_max) for row_max,next_max in zip(row_maxes, next_maxes)]
  state_updates += [row_sum[0].store(row_sum.after(initialized, position)[0].load() * old_scale + weight)
                    for row_sum,old_scale,weight in zip(row_sums, old_scales, weights)]
  positions_done = UOp.group(values_done, *state_updates).end(position)

  output = UOp.range(chunks, 103)
  stores = []
  for token_offset,(numerator,row_sum) in enumerate(zip(numerators, row_sums)):
    query = bh * tokens + token_base + token_offset
    value = _contiguous_vector_load(numerator.after(positions_done)[output * 8], 8) / row_sum.after(positions_done)[0].load()
    stores.append(_contiguous_vector_ptr(outf, query * dim + output * 8, 8).store(value))
  return UOp.group(*stores).end(output, job, core).sink(
    arg=KernelInfo(name=f"attention_prefill_online_uop_{batch}_{heads}_{tokens}_{kv_heads}_{dim}_{cache_len}",
                   optimize=False, parallel=True))

def uop_attention_prefill(q:Tensor, cache:Tensor, start_pos:int|UOp) -> Tensor:
  batch, heads, tokens, head_dim = q.shape
  assert tokens > 1 and q.dtype == dtypes.float32 and cache.dtype == dtypes.float16
  cache_len = cache.shape[3]
  assert isinstance(cache_len, int) and head_dim % 8 == 0
  out = Tensor.empty(batch, heads, tokens, head_dim, dtype=dtypes.float32, device=q.device)
  pos = UOp.variable("cpu_attention_prefill_pos", 0, cache_len-1).bind(start_pos) if isinstance(start_pos, int) else start_pos
  pos_var = pos.unbind()[0]
  srcs = (out.uop, q.contiguous().uop, cache.uop)
  params = [UOp.placeholder_like(x, slot=i) for i,x in enumerate(srcs)]
  return Tensor(out.uop.after(_attention_prefill_online_uop(*params, pos_var).call(*srcs, pos)))

def attention_prefill(q:Tensor, cache:Tensor, start_pos:int|UOp) -> Tensor:
  batch, heads, tokens, head_dim = q.shape
  assert tokens > 1 and q.dtype == dtypes.float32 and cache.dtype == dtypes.float16
  cache_len = cache.shape[3]
  assert isinstance(cache_len, int) and head_dim % 8 == 0
  return uop_attention_prefill(q, cache, start_pos)

def gated_delta(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor,
                     inplace:bool=False, norm_weight:Tensor|None=None, norm_eps:float=0.0) -> tuple[Tensor, Tensor]:
  batch, heads, dim = map(_concrete_int, q.shape)
  assert q.shape == k.shape == v.shape == (batch, heads, dim)
  assert beta.shape == alpha.shape == (batch, heads) and state.shape == (batch, heads, dim, dim)
  assert all(x.dtype == dtypes.float32 for x in (q, k, v, beta, alpha))
  assert state.dtype in (dtypes.float16, dtypes.float32)
  assert norm_weight is None or norm_weight.shape == (dim,) and norm_weight.dtype == dtypes.float16
  core, next_state = Tensor.empty_like(q), state if inplace else Tensor.empty_like(state)
  normalize = norm_weight is not None
  norm_arg = state if norm_weight is None else norm_weight
  outputs = Tensor.custom_kernel(core, next_state, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(),
    alpha.contiguous(), state, norm_arg, fxn=lambda core,next_state,q,k,v,beta,alpha,state,norm_weight:
      _gated_delta_uop(core, next_state, q, k, v, beta, alpha, state, norm_weight, normalize, norm_eps))
  return outputs[0], outputs[1]

@functools.cache
def _gated_delta_prefill_uop(core:UOp, next_state:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp,
                             norm_weight:UOp, norm_eps:float) -> UOp:
  batch, heads, tokens, dim = map(_concrete_int, q.shape)
  chunks = dim // 8
  coref, nextf, qf, kf, vf = core.flatten(), next_state.flatten(), q.flatten(), k.flatten(), v.flatten()
  betaf, alphaf, statef, normf = beta.flatten(), alpha.flatten(), state.flatten(), norm_weight.flatten()
  bh = UOp.range(batch * heads, 0, AxisType.GLOBAL)

  current = UOp.placeholder((dim * dim,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  init_chunk = UOp.range(dim * dim // 8, 90)
  initial_values = _contiguous_vector_load(statef[bh * dim * dim + init_chunk * 8], 8).float()
  initialized_state = _contiguous_vector_ptr(current, init_chunk * 8, 8).store(initial_values).end(init_chunk)

  token = UOp.range(tokens, 91)
  token_base = (bh * tokens + token) * dim
  kq_acc = UOp.placeholder((8,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  kq_acc = _vector_acc_init(kq_acc, initialized_state, token)
  chunk = UOp.range(chunks, 92)
  qvec = _contiguous_vector_load(qf[token_base + chunk * 8], 8)
  kvec = _contiguous_vector_load(kf[token_base + chunk * 8], 8)
  kq_done = _vector_acc_update(kq_acc, qvec * kvec, chunk).end(chunk)
  kq = sum((kq_acc.after(kq_done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))

  core_values = UOp.placeholder((dim,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  row = UOp.range(dim, 93)
  state_k = UOp.placeholder((8,), dtypes.float32, slot=3, addrspace=AddrSpace.REG)
  state_q = UOp.placeholder((8,), dtypes.float32, slot=4, addrspace=AddrSpace.REG)
  dot_init = UOp.group(_vector_reg(state_k, row).store(state_k.const_like(0)),
                       _vector_reg(state_q, row).store(state_q.const_like(0)))
  col = UOp.range(chunks, 94)
  state_vec = _contiguous_vector_load(current.after(initialized_state, token)[row * dim + col * 8], 8)
  q_vec = _contiguous_vector_load(qf[token_base + col * 8], 8)
  k_vec = _contiguous_vector_load(kf[token_base + col * 8], 8)
  dots = UOp.group(_vector_acc_update(state_k, state_vec * k_vec, dot_init, col),
                   _vector_acc_update(state_q, state_vec * q_vec, dot_init, col)).end(col)
  sk = sum((state_k.after(dots)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))
  sq = sum((state_q.after(dots)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))
  av, bv = alphaf[bh * tokens + token].load(), betaf[bh * tokens + token].load()
  delta = (vf[token_base + row].load() - sk * av) * bv
  saved_core = core_values.after(dots)[row].store(sq * av + delta * kq)
  update = UOp.range(chunks, 95)
  update_base = row * dim + update * 8
  current_values = _contiguous_vector_load(current.after(initialized_state, token)[update_base], 8)
  key_values = _contiguous_vector_load(kf[token_base + update * 8], 8)
  next_values = current_values * av + delta * key_values
  updated_state = _contiguous_vector_ptr(current.after(dots), update_base, 8).store(next_values).end(update)
  rows_done = UOp.group(saved_core, updated_state).end(row)

  norm_acc = UOp.placeholder((8,), dtypes.float32, slot=5, addrspace=AddrSpace.REG)
  norm_acc = _vector_acc_init(norm_acc, rows_done)
  norm_chunk = UOp.range(chunks, 96)
  cv = _contiguous_vector_load(core_values.after(rows_done)[norm_chunk * 8], 8)
  norm_done = _vector_acc_update(norm_acc, cv * cv, norm_chunk).end(norm_chunk)
  norm_sum = sum((norm_acc.after(norm_done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))
  scale = (norm_sum / dim + norm_eps).rsqrt()
  output = UOp.range(dim, 97)
  token_done = coref[token_base + output].store(
    core_values.after(norm_done)[output].load() * scale * normf[output].load().float()).end(output, token)

  save_chunk = UOp.range(dim * dim // 8, 98)
  saved_values = _contiguous_vector_load(current.after(token_done)[save_chunk * 8], 8).cast(next_state.dtype)
  saved_state = _contiguous_vector_ptr(nextf, bh * dim * dim + save_chunk * 8, 8).store(saved_values).end(save_chunk)
  return saved_state.end(bh).sink(
    arg=KernelInfo(name=f"gated_delta_prefill_uop_{batch}_{heads}_{tokens}_{dim}_{state.dtype.name}",
                   optimize=False, parallel=True))

@functools.cache
def _gated_delta_uop(core:UOp|None, next_state:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp,
                     norm_weight:UOp, normalize:bool, norm_eps:float, quant:UOp|None=None, quant_scale:UOp|None=None,
                     gate:UOp|None=None) -> UOp:
  batch, heads, dim = map(_concrete_int, q.shape)
  assert dim % 8 == 0
  coref = core.flatten() if core is not None else None
  nextf, qf, kf, vf = next_state.flatten(), q.flatten(), k.flatten(), v.flatten()
  betaf, alphaf, statef = beta.flatten(), alpha.flatten(), state.flatten()
  bh = UOp.range(batch * heads, 0, AxisType.GLOBAL)
  chunks = dim // 8

  kq_acc = UOp.placeholder((8,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  kq_acc = _vector_acc_init(kq_acc, bh)
  chunk = UOp.range(chunks, 90)
  qv = _contiguous_vector_load(qf[bh * dim + chunk * 8], 8)
  kv = _contiguous_vector_load(kf[bh * dim + chunk * 8], 8)
  kq_done = _vector_acc_update(kq_acc, qv * kv, chunk).end(chunk)
  kq = sum((kq_acc.after(kq_done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))

  core_values = UOp.placeholder((dim,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  row = UOp.range(dim, 91)
  state_k = UOp.placeholder((8,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  state_q = UOp.placeholder((8,), dtypes.float32, slot=3, addrspace=AddrSpace.REG)
  initialized = UOp.group(_vector_reg(state_k, row).store(state_k.const_like(0)),
                          _vector_reg(state_q, row).store(state_q.const_like(0)))
  col = UOp.range(chunks, 92)
  state_vec = _contiguous_vector_load(statef[bh * dim * dim + row * dim + col * 8], 8).float()
  q_vec = _contiguous_vector_load(qf[bh * dim + col * 8], 8)
  k_vec = _contiguous_vector_load(kf[bh * dim + col * 8], 8)
  dots = UOp.group(_vector_acc_update(state_k, state_vec * k_vec, initialized, col),
                   _vector_acc_update(state_q, state_vec * q_vec, initialized, col)).end(col)
  sk = sum((state_k.after(dots)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))
  sq = sum((state_q.after(dots)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))
  av, bv = alphaf[bh].load(), betaf[bh].load()
  delta = (vf[bh * dim + row].load() - sk * av) * bv
  core_value = sq * av + delta * kq
  saved_core = core_values.after(dots)[row].store(core_value)
  update = UOp.range(chunks, 93)
  state_base = bh * dim * dim + row * dim + update * 8
  state_values = _contiguous_vector_load(statef[state_base], 8).float()
  key_values = _contiguous_vector_load(kf[bh * dim + update * 8], 8)
  next_values = (state_values * av + delta * key_values).cast(next_state.dtype)
  update_state = _contiguous_vector_ptr(nextf.after(dots), state_base, 8).store(next_values).end(update)
  rows_done = UOp.group(saved_core, update_state).end(row)

  if normalize:
    norm_acc = UOp.placeholder((8,), dtypes.float32, slot=4, addrspace=AddrSpace.REG)
    norm_acc = _vector_acc_init(norm_acc, rows_done)
    norm_chunk = UOp.range(chunks, 94)
    cv = _contiguous_vector_load(core_values.after(rows_done)[norm_chunk * 8], 8)
    norm_done = _vector_acc_update(norm_acc, cv * cv, norm_chunk).end(norm_chunk)
    norm_sum = sum((norm_acc.after(norm_done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))
    scale = (norm_sum / dim + norm_eps).rsqrt()
    source = core_values.after(norm_done)
  else:
    scale, source = UOp.const(dtypes.float32, 1), core_values.after(rows_done)
  if quant is None:
    assert coref is not None and quant_scale is None and gate is None
    output = UOp.range(dim, 95)
    value = source[output].load() * scale
    if normalize: value = value * norm_weight[output].load().float()
    stores = coref[bh * dim + output].store(value).end(output)
    name = f"gated_delta_uop_{batch}_{heads}_{dim}_{state.dtype.name}"
  else:
    assert coref is None and quant_scale is not None and gate is not None and normalize and dim % 32 == 0
    quantf, scalef, gatef = quant.flatten(), quant_scale.flatten(), gate.flatten()
    quant_group = UOp.range(dim // 32, 95)
    base = bh * dim + quant_group * 32
    value_vecs = []
    for chunk_idx in range(4):
      offset = chunk_idx * 8
      source_values = _contiguous_vector_load(source[quant_group * 32 + offset], 8)
      norm_values = _contiguous_vector_load(norm_weight[quant_group * 32 + offset], 8).float()
      gate_values = _contiguous_vector_load(gatef[base + offset], 8).float()
      gate_sigmoid = UOp.stack(*(gate_values[i].sigmoid() for i in range(8)))
      value_vecs.append(source_values * scale * norm_values * gate_values * gate_sigmoid)
    values = tuple(value_vecs[i // 8][i % 8] for i in range(32))
    amax = functools.reduce(lambda a,b:a.maximum(b), (value.abs() for value in values))
    d = (amax / 127).maximum(1e-8)
    quant_stores = []
    for chunk_idx,value_vec in enumerate(value_vecs):
      quant_values = UOp.stack(*((value_vec[i] / d).round().maximum(-127).minimum(127).cast(dtypes.int8) for i in range(8)))
      quant_stores.append(_contiguous_vector_ptr(quantf, base + chunk_idx * 8, 8).store(quant_values))
    stores = UOp.group(scalef[bh * (dim // 32) + quant_group].store(d), *quant_stores).end(quant_group)
    name = f"gated_delta_q8_uop_{batch}_{heads}_{dim}_{state.dtype.name}"
  return stores.end(bh).sink(arg=KernelInfo(name=name, optimize=False, parallel=True))

def gated_delta_q8(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, gate:Tensor,
                   norm_weight:Tensor, norm_eps:float=0.0, inplace:bool=False) -> tuple[Tensor, Tensor, Tensor]:
  batch, heads, dim = q.shape
  assert q.shape == k.shape == v.shape == gate.shape == (batch, heads, dim)
  assert beta.shape == alpha.shape == (batch, heads) and state.shape == (batch, heads, dim, dim)
  assert all(x.dtype == dtypes.float32 for x in (q, k, v, beta, alpha)) and gate.dtype == dtypes.float16
  assert state.dtype in (dtypes.float16, dtypes.float32) and norm_weight.shape == (dim,) and norm_weight.dtype == dtypes.float16
  quant = Tensor.empty(batch, heads * dim // 32, 32, dtype=dtypes.int8, device=q.device)
  scale = Tensor.empty(batch, heads * dim // 32, dtype=dtypes.float32, device=q.device)
  next_state = state if inplace else Tensor.empty_like(state)
  outputs = Tensor.custom_kernel(quant, scale, next_state, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(),
    alpha.contiguous(), state, norm_weight, gate.contiguous(),
    fxn=lambda quant,scale,next_state,q,k,v,beta,alpha,state,norm_weight,gate:
      _gated_delta_uop(None, next_state, q, k, v, beta, alpha, state, norm_weight, True, norm_eps, quant, scale, gate))
  return outputs[0], outputs[1], outputs[2]

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor,
                             norm_weight:Tensor, norm_eps:float) -> tuple[Tensor, Tensor]:
  batch, heads, tokens, dim = q.shape
  assert q.shape == k.shape == v.shape == (batch, heads, tokens, dim)
  assert beta.shape == alpha.shape == (batch, heads, tokens) and state.shape == (batch, heads, dim, dim)
  assert all(x.dtype == dtypes.float32 for x in (q, k, v, beta, alpha))
  assert state.dtype in (dtypes.float16, dtypes.float32) and norm_weight.shape == (dim,) and norm_weight.dtype == dtypes.float16
  core, next_state = Tensor.empty_like(q), Tensor.empty_like(state)
  outputs = Tensor.custom_kernel(core, next_state, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(),
    alpha.contiguous(), state, norm_weight, fxn=lambda core,next_state,q,k,v,beta,alpha,state,norm_weight:
      _gated_delta_prefill_uop(core, next_state, q, k, v, beta, alpha, state, norm_weight, norm_eps))
  return outputs[0], outputs[1]

def rmsnorm(norm:nn.RMSNorm, x:Tensor) -> Tensor:
  return norm(x)

def shared_gate(x:Tensor, weight:Tensor) -> Tensor:
  return (x * weight).sum(axis=-1, keepdim=True).sigmoid()

def silu_mul(gate:Tensor, up:Tensor) -> Tensor:
  assert gate.shape == up.shape
  out = Tensor.empty(int(gate.numel()), dtype=(gate + up).dtype, device=gate.device)
  return Tensor.custom_kernel(out, gate.flatten().contiguous(), up.flatten().contiguous(), fxn=silu_mul_kernel)[0].reshape(*gate.shape)

def silu(x:Tensor) -> Tensor:
  out = Tensor.empty(int(x.numel()), dtype=x.dtype, device=x.device)
  return Tensor.custom_kernel(out, x.flatten().contiguous(), fxn=silu_kernel)[0].reshape(*x.shape)

def causal_conv_silu(state:Tensor, x:Tensor, weight:Tensor) -> Tensor:
  assert len(state.shape) == len(x.shape) == 3 and state.shape[0] == x.shape[0] and state.shape[2] == x.shape[2]
  weight_transposed = weight.shape == (state.shape[1] + 1, x.shape[2])
  assert weight_transposed or weight.shape == (x.shape[2], state.shape[1] + 1)
  assert state.dtype in (dtypes.float16, dtypes.float32) and x.dtype in (dtypes.float16, dtypes.float32)
  assert weight.dtype in (dtypes.float16, dtypes.float32)
  batch, tokens, channels = x.shape
  out = Tensor.empty(batch, tokens, channels, dtype=dtypes.float32, device=x.device)
  kernel_size = weight.shape[0] if weight_transposed else weight.shape[1]
  assert channels % 8 == 0
  return Tensor.custom_kernel(out, state.contiguous(), x.contiguous(), weight.contiguous(),
    fxn=lambda out,state,x,weight:_causal_conv_silu_uop(out, state, x, weight, kernel_size, weight_transposed))[0]

@functools.cache
def _causal_conv_silu_uop(out:UOp, state:UOp, x:UOp, weight:UOp, kernel_size:int, weight_transposed:bool) -> UOp:
  batch, tokens, channels = map(_concrete_int, out.shape)
  core, job, work = _parallel_work(batch * tokens * channels // 8)
  base = work * 8
  token_batch, channel = base // channels, base % channels
  batch_idx, token = token_batch // tokens, token_batch % tokens
  total = UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8)))
  for tap in range(kernel_size):
    position = token + tap
    from_state = position < kernel_size - 1
    state_pos, x_pos = position.minimum(kernel_size - 2), (position - (kernel_size - 1)).maximum(0)
    values = UOp.stack(*(from_state.where(state[batch_idx, state_pos, channel + lane].load(),
                                          x[batch_idx, x_pos, channel + lane].load()).float()
                         for lane in range(8)))
    weights = UOp.stack(*(weight[tap, channel + lane].load().float() if weight_transposed else
                          weight[channel + lane, tap].load().float() for lane in range(8)))
    total = total + values * weights
  result = total / (1.0 + _finite_exp2(total * (-1 / math.log(2))))
  stores = UOp.group(*(out[batch_idx, token, channel + lane].store(result[lane]) for lane in range(8)))
  return stores.end(job, core).sink(
    arg=KernelInfo(name=f"causal_conv_silu_uop_{batch}_{tokens}_{channels}_{kernel_size}", optimize=False, parallel=True))

def gdn_qkv(conv:Tensor, k_heads:int, v_heads:int, dim:int) -> tuple[Tensor, Tensor, Tensor]:
  batch, tokens, channels = conv.shape
  assert conv.dtype == dtypes.float32 and channels == (2 * k_heads + v_heads) * dim
  assert dim % 8 == 0
  outputs = [Tensor.empty(batch, v_heads, tokens, dim, dtype=dtypes.float32, device=conv.device) for _ in range(3)]
  return tuple(Tensor.custom_kernel(*outputs, conv.flatten().contiguous(), fxn=lambda q,k,v,conv:
    _gdn_qkv_uop(q, k, v, conv, k_heads, v_heads, dim))[:3])  # type: ignore[return-value]

@functools.cache
def _gdn_qkv_uop(q:UOp, k:UOp, v:UOp, conv:UOp, k_heads:int, v_heads:int, dim:int) -> UOp:
  batch, _, tokens, _ = map(_concrete_int, q.shape)
  channels, q_dim = (2 * k_heads + v_heads) * dim, k_heads * dim
  core, job, work = _parallel_work(batch * v_heads * tokens)
  token, batch_head = work % tokens, work // tokens
  batch_idx, head = batch_head // v_heads, batch_head % v_heads
  k_head = head % k_heads
  base = (batch_idx * tokens + token) * channels
  qbase, kbase, vbase = base + k_head * dim, base + q_dim + k_head * dim, base + 2 * q_dim + head * dim

  qsum = UOp.placeholder((8,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  ksum = UOp.placeholder((8,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  initialized = UOp.group(_vector_reg(qsum, job).store(qsum.const_like(0)),
                          _vector_reg(ksum, job).store(ksum.const_like(0)))
  chunk = UOp.range(dim // 8, 100)
  qvalue = _contiguous_vector_load(conv[qbase + chunk * 8], 8).float()
  kvalue = _contiguous_vector_load(conv[kbase + chunk * 8], 8).float()
  summed = UOp.group(_vector_acc_update(qsum, qvalue * qvalue, initialized, chunk),
                     _vector_acc_update(ksum, kvalue * kvalue, initialized, chunk)).end(chunk)
  qtotal = sum((qsum.after(summed)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  ktotal = sum((ksum.after(summed)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  qscale, kscale = ((qtotal + 1e-6) * dim).rsqrt(), (ktotal + 1e-6).rsqrt()

  out_chunk = UOp.range(dim // 8, 101)
  qvalues = _contiguous_vector_load(conv[qbase + out_chunk * 8], 8).float() * qscale
  kvalues = _contiguous_vector_load(conv[kbase + out_chunk * 8], 8).float() * kscale
  vvalues = _contiguous_vector_load(conv[vbase + out_chunk * 8], 8).float()
  stores = UOp.group(*(
    target[batch_idx, head, token, out_chunk * 8 + lane].store(values[lane])
    for target,values in ((q, qvalues), (k, kvalues), (v, vvalues)) for lane in range(8))).end(out_chunk)
  return stores.end(job, core).sink(
    arg=KernelInfo(name=f"gdn_qkv_uop_{batch}_{tokens}_{k_heads}_{v_heads}_{dim}", optimize=False, parallel=True))

def silu_mul_kernel(out:UOp, gate:UOp, up:UOp) -> UOp:
  elements = _concrete_int(out.shape[0])
  if elements >= 131072:
    core, job, idx = _parallel_work(elements)
    value = gate[idx].load()
    return out[idx].store(value * value.sigmoid() * up[idx].load()).end(job, core).sink(
      arg=KernelInfo(name=f"cpu_silu_mul_{elements}", optimize=False, parallel=True))
  idx = UOp.range(elements, 0, axis_type=AxisType.WEAK)
  value = gate[idx].load()
  return out[idx].store(value * value.sigmoid() * up[idx].load()).end(idx).sink(
    arg=KernelInfo(name=f"cpu_silu_mul_{elements}", opts_to_apply=()))

def silu_kernel(out:UOp, x:UOp) -> UOp:
  elements = _concrete_int(out.shape[0])
  if elements >= 131072:
    core, job, idx = _parallel_work(elements)
    value = x[idx].load()
    return out[idx].store(value * value.sigmoid()).end(job, core).sink(
      arg=KernelInfo(name=f"cpu_silu_{elements}", optimize=False, parallel=True))
  idx = UOp.range(elements, 0, axis_type=AxisType.WEAK)
  value = x[idx].load()
  return out[idx].store(value * value.sigmoid()).end(idx).sink(arg=KernelInfo(name=f"cpu_silu_{elements}", opts_to_apply=()))

def _cpu_topk_uop(out:UOp, sel:UOp, x:UOp, k:int, bias:UOp|None=None, normalize:bool=False) -> UOp:
  outer = _concrete_int(out.shape[0])
  core, job, row = _parallel_work(outer)
  scores = UOp.placeholder((k,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  indices = UOp.placeholder((k,), dtypes.int32, slot=1, addrspace=AddrSpace.REG)
  init = UOp.range(k, 91)
  initialized = UOp.group(scores.after(job)[init].store(-math.inf), indices.after(job)[init].store(0)).end(init)

  index = UOp.range(256, 92)
  raw = x[row, index].load()
  score = raw.float() if bias is None else \
    (raw.sigmoid().cast(x.dtype) + bias[index].load()).cast(x.dtype).float()
  worst_score, worst_index, worst_slot = scores.after(initialized, index)[0].load(), indices.after(initialized, index)[0].load(), \
    UOp.const(dtypes.weakint, 0)
  for slot in range(1, k):
    candidate_score, candidate_index = scores.after(index)[slot].load(), indices.after(index)[slot].load()
    worse = (candidate_score < worst_score) | ((candidate_score == worst_score) & (candidate_index > worst_index))
    worst_score, worst_index = worse.where(candidate_score, worst_score), worse.where(candidate_index, worst_index)
    worst_slot = worse.where(UOp.const(dtypes.weakint, slot), worst_slot)
  take = (score > worst_score) | ((score == worst_score) & (index.int() < worst_index))
  selected = UOp.group(scores[worst_slot.valid(take)].store(score),
                       indices[worst_slot.valid(take)].store(index.int())).end(index)

  sorted_values = selected
  # Ascending score order, with larger indices first on ties, matches the reference implementation's reversed descending list.
  for end in range(k - 1, 0, -1):
    for slot in range(end):
      left_score, right_score = scores.after(sorted_values)[slot].load(), scores.after(sorted_values)[slot + 1].load()
      left_index, right_index = indices.after(sorted_values)[slot].load(), indices.after(sorted_values)[slot + 1].load()
      swap = (left_score > right_score) | ((left_score == right_score) & (left_index < right_index))
      sorted_values = UOp.group(scores[slot].store(swap.where(right_score, left_score)),
                                scores[slot + 1].store(swap.where(left_score, right_score)),
                                indices[slot].store(swap.where(right_index, left_index)),
                                indices[slot + 1].store(swap.where(left_index, right_index)))

  rank = UOp.range(k, 93)
  selected_index = indices.after(sorted_values, rank)[rank].load().cast(dtypes.weakint)
  if bias is not None:
    values = [x[row, indices.after(sorted_values)[slot].load().cast(dtypes.weakint)].load().sigmoid().cast(out.dtype)
              for slot in range(k)]
    denom = UOp.const(out.dtype, 0)
    for value in values: denom = (denom + value).cast(out.dtype)
    value = values[0] if k == 1 else x[row, selected_index].load().sigmoid().cast(out.dtype)
    if normalize: value = (value / denom).cast(out.dtype)
  else:
    maximum = scores.after(sorted_values)[k - 1].load()
    exps = [((scores.after(sorted_values)[slot].load() - maximum) * (1 / math.log(2))).exp2() for slot in range(k)]
    value = (exps[0] if k == 1 else
             ((scores.after(sorted_values, rank)[rank].load() - maximum) * (1 / math.log(2))).exp2()) / sum(
               exps, UOp.const(dtypes.float32, 0))
  stores = UOp.group(out[row, rank].store(value.cast(out.dtype)), sel[row, rank].store(selected_index.int()))
  return stores.end(rank, job, core).sink(
    arg=KernelInfo(name=f"cpu_uop_{'biased_' if bias is not None else ''}topk_{outer}_{k}", optimize=False, parallel=True))

def uop_biased_topk(x:Tensor, bias:Tensor, k:int, normalize:bool) -> tuple[Tensor, Tensor]:
  outer = int(x.numel()) // 256
  values, indices = Tensor.empty(outer, k, dtype=x.dtype, device=x.device), \
    Tensor.empty(outer, k, dtype=dtypes.int32, device=x.device)
  return tuple(Tensor.custom_kernel(values, indices, x.reshape(outer, 256).contiguous(), bias.contiguous(),
    fxn=lambda out,sel,x,bias:_cpu_topk_uop(out, sel, x, k, bias, normalize))[:2])  # type: ignore[return-value]

def uop_topk_softmax(x:Tensor, k:int) -> tuple[Tensor, Tensor]:
  outer = int(x.numel()) // 256
  values, indices = Tensor.empty(outer, k, dtype=x.dtype, device=x.device), \
    Tensor.empty(outer, k, dtype=dtypes.int32, device=x.device)
  outputs = Tensor.custom_kernel(values, indices, x.reshape(outer, 256).contiguous(),
    fxn=lambda out,sel,x:_cpu_topk_uop(out, sel, x, k))
  shape = (*x.shape[:-1], k)
  return outputs[0].reshape(*shape), outputs[1].reshape(*shape)

def q8_silu_linear(layer:Linear, gate:Tensor, up:Tensor) -> Tensor:
  assert layer.ggml_type == 8 and layer.bias is None and gate.shape == up.shape and gate.shape[-1] == layer.in_features
  assert gate.dtype == dtypes.float16 and up.dtype == dtypes.float32
  xq, xd = q8_silu_quantize(gate, up, layer.in_features)
  return uop_q8_prequant_linear(layer, xq, xd).reshape(*gate.shape[:-1], layer.out_features)

@functools.cache
def _f16_matvec_uop(out:UOp, x:UOp, weight:UOp) -> UOp:
  tokens, out_features, in_features = out.shape[0], out.shape[1], weight.shape[1]
  assert in_features % 8 == 0
  token_tile = 8 if tokens % 8 == 0 else 1
  cores = math.gcd(out_features, getenv("CPU_F16_UOP_CORES", 32))
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  output_job = UOp.range(out_features // cores, 90)
  output = core * (out_features // cores) + output_job
  token_block = UOp.range(tokens // token_tile, 91)
  token_base = token_block * token_tile
  xf, wf, outf = x.flatten(), weight.flatten(), out.flatten()
  accs = tuple(UOp.placeholder((8,), dtypes.float32, slot=i, addrspace=AddrSpace.REG) for i in range(token_tile))
  accs = tuple(acc.after(_vector_reg(acc, output_job, token_block).store(acc.const_like(0))) for acc in accs)
  chunk = UOp.range(in_features // 8, 92)
  weights = _load_f16x8_ptr(wf[output * in_features + chunk * 8])
  updates = []
  for token,acc in enumerate(accs):
    xbase = (token_base + token) * in_features + chunk * 8
    values = _load_f16x8_ptr(xf[xbase]) if x.dtype == dtypes.float16 else \
      UOp.stack(*(xf[xbase + lane].load() for lane in range(8)))
    previous = _vector_reg(acc, chunk).load()
    updates.append(_vector_reg(acc, chunk).store(previous + values * weights))
  done = UOp.group(*updates).end(chunk)
  stores = [outf[(token_base + token) * out_features + output].store(
    sum((acc.after(done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0)).cast(out.dtype))
            for token,acc in enumerate(accs)]
  return UOp.group(*stores).end(token_block, output_job, core).sink(
    arg=KernelInfo(name=f"f16_matvec_uop_{tokens}_{out_features}_{in_features}", optimize=False, parallel=True))

def uop_f16_matvec(x:Tensor, weight:Tensor) -> Tensor:
  assert weight.dtype == dtypes.float16 and len(weight.shape) == 2 and int(x.numel()) % weight.shape[1] == 0
  assert x.dtype in (dtypes.float16, dtypes.float32)
  tokens = int(x.numel()) // weight.shape[1]
  xc = x.reshape(tokens, weight.shape[1]).contiguous()
  out = Tensor.empty(tokens, weight.shape[0], dtype=x.dtype, device=x.device)
  return Tensor.custom_kernel(out, xc, weight, fxn=_f16_matvec_uop)[0].reshape(*x.shape[:-1], weight.shape[0])

def f16_matvec(x:Tensor, weight:Tensor) -> Tensor:
  assert weight.dtype == dtypes.float16 and len(weight.shape) == 2 and int(x.numel()) % weight.shape[1] == 0
  assert x.dtype in (dtypes.float16, dtypes.float32)
  return uop_f16_matvec(x, weight)

def f16_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type is None and layer.bias is None and layer.weight.dtype == dtypes.float16
  assert int(x.numel()) % layer.in_features == 0
  return f16_matvec(x, layer.weight)

def rmsnorm_f16_linear(norm:nn.RMSNorm, layer:Linear, x:Tensor) -> tuple[Tensor, Tensor]:
  assert x.dtype == dtypes.float32 and int(x.numel()) == x.shape[-1] == layer.in_features
  assert norm.weight is not None and norm.weight.dtype == layer.weight.dtype == dtypes.float16
  assert layer.ggml_type is None and layer.bias is None
  normalized = Tensor.empty(x.shape, dtype=dtypes.float32, device=x.device)
  out = Tensor.empty(*x.shape[:-1], layer.out_features, dtype=dtypes.float32, device=x.device)
  outputs = Tensor.custom_kernel(normalized, out, x, norm.weight, layer.weight,
    fxn=lambda normalized,out,x,norm_weight,weight:
      _rmsnorm_f16_linear_uop(normalized, out, x, norm_weight, weight, norm.eps))
  return outputs[0], outputs[1]

@functools.cache
def _rmsnorm_f16_linear_uop(normalized:UOp, out:UOp, x:UOp, norm_weight:UOp, weight:UOp, eps:float) -> UOp:
  dim, out_features = x.shape[-1], out.shape[-1]
  assert dim % 8 == 0
  cores = min(out_features, getenv("CPU_RMS_ROUTER_CORES", 16))
  while out_features % cores or dim % cores: cores -= 1
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  xf, normf, normalizedf, outf, weightf = x.flatten(), norm_weight.flatten(), normalized.flatten(), out.flatten(), weight.flatten()

  norm_acc = UOp.placeholder((8,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  norm_acc = norm_acc.after(norm_acc.after(core).store(norm_acc.const_like(0)))
  chunk = UOp.range(dim // 8, 90)
  xv = UOp.stack(*(xf[chunk * 8 + lane].load() for lane in range(8)))
  norm_done = norm_acc.after(chunk).store(norm_acc.after(chunk) + xv * xv).end(chunk)
  norm_sum = sum((norm_acc.after(norm_done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))
  scale = (norm_sum / dim + eps).rsqrt()

  norm_job = UOp.range(dim // cores, 91)
  norm_idx = core * (dim // cores) + norm_job
  normalized_done = normalizedf[norm_idx].store(
    xf[norm_idx].load() * scale * normf[norm_idx].load().float()).end(norm_job)

  output_job = UOp.range(out_features // cores, 92)
  output = core * (out_features // cores) + output_job
  linear_acc = UOp.placeholder((8,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  linear_acc = linear_acc.after(linear_acc.after(normalized_done, output_job).store(linear_acc.const_like(0)))
  linear_chunk = UOp.range(dim // 8, 93)
  products = _rms_f16_product_ptr(xf[linear_chunk * 8], normf[linear_chunk * 8],
                                   weightf[output * dim + linear_chunk * 8], scale)
  linear_done = linear_acc.after(linear_chunk).store(linear_acc.after(linear_chunk) + products).end(linear_chunk)
  value = sum((linear_acc.after(linear_done)[lane] for lane in range(8)), UOp.const(dtypes.float32, 0))
  return outf[output].store(value).end(output_job, core).sink(
    arg=KernelInfo(name=f"rmsnorm_f16_linear_uop_{out_features}_{dim}", optimize=False, parallel=True))

def q8_batched_pair(first:Linear, second:Linear, x:Tensor) -> tuple[Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features == x.shape[-1]
  tokens = int(x.numel()) // x.shape[-1]
  assert tokens > 1 and x.dtype in (dtypes.float16, dtypes.float32)
  xq, xd = q8_quantize(x.reshape(tokens, first.in_features), first.in_features)
  shape = x.shape[:-1]
  return (uop_q8_prequant_linear(first, xq, xd, x.dtype).reshape(*shape, first.out_features),
          uop_q8_prequant_linear(second, xq, xd, x.dtype).reshape(*shape, second.out_features))

def q8_linear_pair(first:Linear, second:Linear, x:Tensor) -> tuple[Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features and int(x.numel()) == first.in_features
  return uop_q8_linear_pair(first, second, x)

def q8_gdn_projections(first:Linear, second:Linear, f16_weight:Tensor, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features == x.shape[-1]
  assert x.dtype == f16_weight.dtype == dtypes.float16 and f16_weight.shape[1] == x.shape[-1] and int(x.numel()) == x.shape[-1]
  return uop_q8_gdn_projections(first, second, f16_weight, x)

def q8_gdn_norm_projections(first:Linear, second:Linear, f16_weight:Tensor, x:Tensor, norm:nn.RMSNorm) -> tuple[Tensor, Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features == x.shape[-1]
  assert x.dtype == dtypes.float32 and f16_weight.dtype == dtypes.float16 and f16_weight.shape[1] == x.shape[-1]
  assert norm.weight is not None and norm.weight.dtype == dtypes.float16 and int(x.numel()) == x.shape[-1]
  assert first.cpu_repacked is not None and second.cpu_repacked is not None
  return uop_q8_gdn_norm_projections(first, second, f16_weight, x, norm)

def q6_argmax(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type == 14 and int(x.numel()) == layer.in_features
  cores = math.gcd(layer.out_features, 32)
  values = Tensor.empty(cores, dtype=dtypes.float32, device=x.device)
  indices = Tensor.empty(cores, dtype=dtypes.int32, device=x.device)
  xq, xd = q8k_quantize(x, layer.in_features)
  parts = Tensor.custom_kernel(values, indices, layer.weight, xq, xd, fxn=lambda values,indices,raw,xq,xd:
    _q6_argmax_uop(values, indices, raw, xq, xd, layer.out_features, layer.in_features))
  out = Tensor.empty(1, 1, dtype=dtypes.int32, device=x.device)
  return Tensor.custom_kernel(out, parts[0], parts[1], fxn=_argmax_parts_uop)[0]

def _argmax_parts_uop(out:UOp, values:UOp, indices:UOp) -> UOp:
  best = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  best_idx = UOp.placeholder((1,), dtypes.int32, slot=1, addrspace=AddrSpace.REG)
  initialized = UOp.group(best.store(-math.inf), best_idx.store(0))
  part = UOp.range(values.shape[0], 90)
  value, index = values.after(initialized, part)[part].load(), indices.after(initialized, part)[part].load()
  current, current_idx = best.after(part)[0].load(), best_idx.after(part)[0].load()
  take = (value > current) | ((value == current) & (index < current_idx))
  selected = UOp.group(best[0].store(take.where(value, current)),
                       best_idx[0].store(take.where(index, current_idx))).end(part)
  return out[0, 0].store(best_idx.after(selected)[0].load()).sink(
    arg=KernelInfo(name=f"q6_argmax_reduce_{values.shape[0]}", opts_to_apply=()))

@functools.cache
def _q6_argmax_uop(values:UOp, indices:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  cores, blocks, row_size = values.shape[0], in_features // 256, in_features // 256 * 210
  core = UOp.range(cores, 0, AxisType.GLOBAL)
  begin, rows = out_features * core // cores, out_features // cores
  best = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  best_idx = UOp.placeholder((1,), dtypes.int32, slot=1, addrspace=AddrSpace.REG)
  initialized = UOp.group(best.after(core).store(best.const_like(-math.inf)), best_idx.after(core).store(begin.int()))
  row_job = UOp.range(rows, 90)
  row = begin + row_job
  acc = UOp.placeholder((8,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  acc = acc.after(acc.after(initialized, row_job).store(acc.const_like(0)))
  block = UOp.range(blocks, 91)
  base = row * row_size + block * 210
  block_sum = UOp.stack(*(UOp.const(dtypes.float32, 0) for _ in range(8)))
  for subgroup in range(8):
    parts = _dot_q6_ptr(raw[base], xq[0, block, subgroup * 32], subgroup).float()
    scales = UOp.stack(*(raw[base + 192 + subgroup * 2 + j].load().bitcast(dtypes.int8).float() for j in range(2)))
    block_sum = block_sum + parts * UOp.stack(*(scales[j // 4] for j in range(8)))
  block_done = acc.after(block).store(acc.after(block) + block_sum * xd[0, block] * _load_f16(raw, base + 208)).end(block)
  value = sum((acc.after(block_done)[i] for i in range(8)), UOp.const(dtypes.float32, 0))
  take = value > best.after(block_done)[0].load()
  selected = UOp.group(best[0].store(take.where(value, best.after(block_done)[0].load())),
                       best_idx[0].store(take.where(row.int(), best_idx.after(block_done)[0].load()))).end(row_job)
  return UOp.group(values[core].store(best.after(selected)[0].load()),
                   indices[core].store(best_idx.after(selected)[0].load())).end(core).sink(
    arg=KernelInfo(name=f"q6_argmax_uop_{out_features}_{in_features}", optimize=False, parallel=True))

def expert_weighted_sum(layer:ExpertWeights, sel:Tensor, x:Tensor, probs:Tensor) -> Tensor:
  return uop_expert_weighted_sum(layer, sel, x, probs)

def expert_pair(first:ExpertWeights, second:ExpertWeights, sel:Tensor, x:Tensor) -> tuple[Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type and first.ggml_type in (14, 21, 23)
  assert (first.in_features, first.out_features) == (second.in_features, second.out_features)
  return uop_expert(first, sel, x), uop_expert(second, sel, x)

def expert_silu(first:ExpertWeights, second:ExpertWeights, sel:Tensor, x:Tensor) -> Tensor:
  assert first.ggml_type == second.ggml_type and first.ggml_type in (14, 21, 23)
  assert (first.in_features, first.out_features) == (second.in_features, second.out_features) and x.dtype == dtypes.float32
  return uop_expert_silu(first, second, sel, x)

def weighted_sum(x:Tensor, probs:Tensor) -> Tensor:
  inputs, routes, dim = int(probs.numel()) // probs.shape[-1], probs.shape[-1], x.shape[-1]
  assert x.shape[-2] == routes and x.dtype == probs.dtype and x.dtype in (dtypes.float16, dtypes.float32)
  return (x.reshape(inputs, routes, dim) * probs.reshape(inputs, routes, 1)).sum(axis=1).reshape(*probs.shape[:-1], dim)

def moe_ffn(block:FFNBlock, x:Tensor, probs:Tensor, sel:Tensor) -> Tensor:
  inputs = int(x.numel()) // block.config.dim
  if inputs == 1 and all(layer.cpu_repacked is not None for layer in
                         (block.ffn_gate_exps, block.ffn_up_exps, block.ffn_gate_shexp,
                          block.ffn_up_shexp, block.ffn_down_shexp)):
    return uop_moe_ffn(block, x, probs, sel)
  routed_hidden = uop_expert_silu(block.ffn_gate_exps, block.ffn_up_exps, sel, x)
  routed = weighted_sum(uop_expert(block.ffn_down_exps, sel, routed_hidden), probs)
  shared_gate_out, shared_up = block.ffn_gate_shexp(x), block.ffn_up_shexp(x)
  shared = block.ffn_down_shexp(silu_mul(shared_gate_out, shared_up))
  return routed + shared * shared_gate(x, block.ffn_gate_inp_shexp["weight"])
