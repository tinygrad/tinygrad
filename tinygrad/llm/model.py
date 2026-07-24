from __future__ import annotations
import functools, itertools, pathlib
from dataclasses import dataclass, replace
from tinygrad import Device, Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Context
from tinygrad.helpers import JIT_BATCH_SIZE
from tinygrad.dtype import AddrSpace
from tinygrad.llm.gguf import get_ggml_quantization, ggml_data_to_tensor, gguf_load, _GGML_QUANT
from tinygrad.uop.ops import resolve, Ops, KernelInfo, AxisType

@functools.cache
def _q8_kernel(quant:UOp, scale:UOp, x:UOp, in_features:int) -> UOp:
  x = x.flatten()
  token, group = UOp.range(quant.shape[0], 0), UOp.range(in_features // 32, 1)
  lane = UOp.range(32, 2, axis_type=AxisType.LOCAL)
  raw_value = x[token * in_features + group * 32 + lane].load().cast(dtypes.float32)
  amax = raw_value.abs()
  for offset in (16, 8, 4, 2, 1):
    amax = amax.maximum(UOp(Ops.CUSTOM, dtypes.float32, (((lane ^ offset) * 4).cast(dtypes.int32), amax),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))"))
  d = (amax / 127).maximum(1e-8)
  word = UOp.const(dtypes.uint32, 0)
  for byte_idx in range(4):
    source_lane = lane * 4 + byte_idx
    value = (UOp(Ops.CUSTOM, dtypes.float32, ((source_lane * 4).cast(dtypes.int32), raw_value),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))") / d).round().maximum(-127).minimum(127)
    byte = value.cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32)
    word = word | (byte << (8 * byte_idx))
  stores = [scale[token.valid(lane.eq(0)), group].store(d), quant[token, group, lane.valid(lane < 8)].store(word)]
  return UOp.group(*stores).end(token, group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def _q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  quant = Tensor.empty(tokens, in_features // 32, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, in_features // 32, dtype=dtypes.float32, device=x.device)
  return tuple(Tensor.custom_kernel(quant, scale, x,
    fxn=lambda quant,scale,x:_q8_kernel(quant, scale, x, in_features))[:2])  # type: ignore[return-value]

@functools.cache
def _q8_silu_mul_kernel(quant:UOp, scale:UOp, gate:UOp, up:UOp, in_features:int) -> UOp:
  gate, up = gate.flatten(), up.flatten()
  token, group = UOp.range(quant.shape[0], 0), UOp.range(in_features // 32, 1)
  lane = UOp.range(32, 2, axis_type=AxisType.LOCAL)
  def value(idx:UOp) -> UOp:
    x = gate[idx].cast(dtypes.float32)
    return x * x.sigmoid() * up[idx].cast(dtypes.float32)
  raw_value = value(token * in_features + group * 32 + lane)
  amax = raw_value.abs()
  for offset in (16, 8, 4, 2, 1):
    amax = amax.maximum(UOp(Ops.CUSTOM, dtypes.float32, (((lane ^ offset) * 4).cast(dtypes.int32), amax),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))"))
  d = (amax / 127).maximum(1e-8)
  word = UOp.const(dtypes.uint32, 0)
  for byte_idx in range(4):
    source_lane = lane * 4 + byte_idx
    packed_value = UOp(Ops.CUSTOM, dtypes.float32, ((source_lane * 4).cast(dtypes.int32), raw_value),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")
    byte = (packed_value / d).round().maximum(-127).minimum(127).cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32)
    word = word | (byte << (8 * byte_idx))
  stores = [scale[token.valid(lane.eq(0)), group].store(d), quant[token, group, lane.valid(lane < 8)].store(word)]
  return UOp.group(*stores).end(token, group, lane).sink(arg=KernelInfo(name="q8_silu_mul", opts_to_apply=()))

def _q8_silu_mul(gate:Tensor, up:Tensor, in_features:int) -> tuple[Tensor, Tensor]:
  tokens = int(gate.numel()) // in_features
  quant = Tensor.empty(tokens, in_features // 32, 8, dtype=dtypes.uint32, device=gate.device)
  scale = Tensor.empty(tokens, in_features // 32, dtype=dtypes.float32, device=gate.device)
  return tuple(Tensor.custom_kernel(quant, scale, gate, up, fxn=lambda quant,scale,gate,up:
    _q8_silu_mul_kernel(quant, scale, gate, up, in_features))[:2])  # type: ignore[return-value]

def _amd_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (a.cast(dtypes.int32), b.cast(dtypes.int32), c),
             arg="__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)")

def _amd_wave_sum(value:UOp, lane:UOp, lane_count:int, wave:UOp|None=None) -> UOp:
  assert lane_count in (8, 16, 32)
  for offset in (16, 8, 4, 2, 1)[{32:0, 16:1, 8:2}[lane_count]:]:
    source_lane = (lane ^ offset) + (wave * lane_count if wave is not None else 0)
    value = value + UOp(Ops.CUSTOM, dtypes.float32, ((source_lane * 4).cast(dtypes.int32), value),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")
  return value

@functools.cache
def _q8_linear_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int, raw_offset:int|UOp=0) -> UOp:
  if isinstance(raw_offset, UOp): raw_offset = raw_offset.cast(dtypes.weakint)
  token_tile = 8 if out.shape[0] % 8 == 0 else 1
  wave_count = 4 if token_tile == 1 else 1
  token_block, output_block = UOp.range(out.shape[0] // token_tile, 0), UOp.range(out_features // wave_count, 1)
  wave = UOp.range(wave_count, 3, axis_type=AxisType.LOCAL) if wave_count > 1 else UOp.const(dtypes.weakint, 0)
  output = output_block * wave_count + wave
  tokens = tuple(token_block * token_tile + i for i in range(token_tile))
  group_count, lane_count = in_features // 32, min(32, in_features // 32)
  lane = UOp.range(lane_count, 2, axis_type=AxisType.LOCAL)

  def group_dot(group:UOp) -> list[UOp]:
    block = output * group_count + group
    base, odd = raw_offset + block * 8 + block // 2, (block & 1).ne(0)
    dots = [UOp.const(dtypes.int32, 0)] * token_tile
    for word_idx in range(8):
      # Q8_0 blocks are 34 bytes, so their two-byte scale makes alternate blocks word-aligned. Read aligned u32s
      # directly; the other blocks need only two adjacent words instead of four individual byte loads.
      word = odd.where(raw[base + 1 + word_idx], (raw[base + word_idx] >> 16) | (raw[base + 1 + word_idx] << 16))
      dots = [_amd_dp4a(word, xq[token, group, word_idx], dot) for token,dot in zip(tokens, dots)]
    dbits = odd.where(raw[base] >> 16, raw[base] & 0xffff).cast(dtypes.uint16)
    return [dot.cast(dtypes.float32) * xd[token, group] * dbits.bitcast(dtypes.float16).float() for token,dot in zip(tokens, dots)]

  values = [UOp.const(dtypes.float32, 0)] * token_tile
  for offset in range(0, group_count, lane_count):
    dots = group_dot((lane + offset).valid(lane + offset < group_count))
    values = [value + dot for value,dot in zip(values, dots)]
  totals = [_amd_wave_sum(value, lane, lane_count, wave if wave_count > 1 else None) for value in values]
  stores = [out[token.valid(lane.eq(0)), output].store(total.cast(out.dtype)) for token,total in zip(tokens, totals)]
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(
    arg=KernelInfo(name="linear_q8", opts_to_apply=()))

@functools.cache
def _q8_linear_wmma_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int, raw_offset:UOp) -> UOp:
  raw = raw.replace(dtype=dtypes.uint32, src=(raw.src[0] * raw.dtype.itemsize // 4,), arg=replace(raw.arg, dtype=dtypes.uint32))
  raw_offset = raw_offset.cast(dtypes.weakint)
  def load_word(byte_offset:UOp) -> UOp:
    word_index, half_aligned = byte_offset // 4, (byte_offset & 2).ne(0)
    word = raw[word_index]
    return half_aligned.where((word >> 16) | (raw[word_index + 1] << 16), word)
  token_tile = 32 if out.shape[0] % 32 == 0 else 16
  token_block, output_block = UOp.range(out.shape[0] // token_tile, 0), UOp.range(out_features // 16, 1)
  lane = UOp.range(32, 2, axis_type=AxisType.LOCAL)
  # The codegen may factor this range into several local dimensions. Read the physical wave lane directly.
  hw_lane = UOp(Ops.CUSTOM, dtypes.int32, (lane.cast(dtypes.int32),), arg="__builtin_amdgcn_mbcnt_lo(-1, 0)").cast(dtypes.weakint)
  physical_col, physical_half = hw_lane % 16, hw_lane // 16
  output = output_block * 16 + physical_col
  input_tokens = tuple(token_block * token_tile + tile * 16 + physical_col for tile in range(token_tile // 16))
  tokens = tuple(tuple(token_block * token_tile + tile * 16 + physical_half * 8 + i for i in range(8))
                 for tile in range(token_tile // 16))
  group_count = in_features // 32
  accs = tuple(UOp.placeholder((8,), dtypes.float32, slot=tile, addrspace=AddrSpace.REG) for tile in range(token_tile // 16))
  accs = tuple(acc.after(acc.store(acc.const_like(0))) for acc in accs)
  group = UOp.range(group_count, 3, AxisType.REDUCE)
  raw_accs = [UOp.const(dtypes.int32, 0).broadcast(8) for _ in accs]
  for half in range(2):
    # rocWMMA's gfx11 loader gives each lane eight values, then appends the eight from lane^16.
    kbase = half * 16 + physical_half * 8
    def fragment(words:tuple[UOp, ...]|list[UOp]) -> UOp:
      swapped_words = tuple(UOp(Ops.CUSTOM, dtypes.uint32, (word,), arg="__builtin_amdgcn_ds_swizzle({0}, 16415)") for word in words)
      return UOp.stack(*(((word >> (byte * 8)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8)
                         for word in (*words, *swapped_words) for byte in range(4)))
    block = output * group_count + group
    bwords = [load_word(raw_offset + block * 34 + 2 + kbase + word * 4) for word in range(2)]
    bfrag = fragment(bwords)
    for tile,input_token in enumerate(input_tokens):
      awords = tuple(xq[input_token, group, kbase // 4 + i].load() for i in range(2))
      raw_accs[tile] = UOp.wmma(fragment(awords), bfrag, raw_accs[tile], (16, 16, 16), 'AMD', 32)
  logical_values = []
  for raw_acc in raw_accs:
    vals = tuple(raw_acc.index(i) for i in range(8))
    swapped = tuple(UOp(Ops.CUSTOM, dtypes.int32, (value,), arg="__builtin_amdgcn_ds_swizzle({0}, 50688)") for value in vals)
    low = physical_half.eq(0)
    logical_values.append((low.where(vals[0], swapped[4]), low.where(swapped[0], vals[4]),
                           low.where(vals[1], swapped[5]), low.where(swapped[1], vals[5]),
                           low.where(vals[2], swapped[6]), low.where(swapped[2], vals[6]),
                           low.where(vals[3], swapped[7]), low.where(swapped[3], vals[7])))
  block = output * group_count + group
  scale = (load_word(raw_offset + block * 34) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float()
  update = UOp.group(*(acc.after(group)[i].store(acc.after(group)[i] + value.float() * scale * xd[token, group])
                       for acc,tile_tokens,logical in zip(accs, tokens, logical_values)
                       for i,(token,value) in enumerate(zip(tile_tokens, logical)))).end(group)
  stores = [out[token, output].store(acc.after(update)[i]) for acc,tile_tokens in zip(accs, tokens) for i,token in enumerate(tile_tokens)]
  return UOp.group(*stores).end(token_block, output_block, lane).sink(arg=KernelInfo(name="linear_q8_wmma", opts_to_apply=()))

@functools.cache
def _q8_linear_pair_kernel(out0:UOp, out1:UOp, raw0:UOp, raw1:UOp, xq:UOp, xd:UOp,
                           out_features0:int, out_features1:int, in_features:int, offset0:UOp, offset1:UOp) -> UOp:
  token_tile = 4 if out0.shape[0] % 4 == 0 else 1
  token_block, output = UOp.range(out0.shape[0] // token_tile, 0), UOp.range(max(out_features0, out_features1), 1)
  tokens = tuple(token_block * token_tile + i for i in range(token_tile))
  group_count, lane_count = in_features // 32, min(32, in_features // 32)
  lane = UOp.range(lane_count, 2, axis_type=AxisType.LOCAL)

  def dot(raw:UOp, raw_offset:UOp, output_idx:UOp) -> list[UOp]:
    values = [UOp.const(dtypes.float32, 0)] * token_tile
    for group_offset in range(0, group_count, lane_count):
      group = (lane + group_offset).valid(lane + group_offset < group_count)
      block = output_idx * group_count + group
      base, odd = raw_offset.cast(dtypes.weakint) + block * 8 + block // 2, (block & 1).ne(0)
      accs = [UOp.const(dtypes.int32, 0)] * token_tile
      for word_idx in range(8):
        word = odd.where(raw[base + 1 + word_idx], (raw[base + word_idx] >> 16) | (raw[base + 1 + word_idx] << 16))
        accs = [_amd_dp4a(word, xq[token, group, word_idx], acc) for token,acc in zip(tokens, accs)]
      dbits = odd.where(raw[base] >> 16, raw[base] & 0xffff).cast(dtypes.uint16)
      values = [value + acc.cast(dtypes.float32) * xd[token, group] * dbits.bitcast(dtypes.float16).float()
                for token,acc,value in zip(tokens, accs, values)]
    return [_amd_wave_sum(value, lane, lane_count) for value in values]

  output0, output1 = output.valid(output < out_features0), output.valid(output < out_features1)
  totals0, totals1 = dot(raw0, offset0, output0), dot(raw1, offset1, output1)
  stores = [out0[token.valid(lane.eq(0)), output0].store(total) for token,total in zip(tokens, totals0)] + \
           [out1[token.valid(lane.eq(0)), output1].store(total) for token,total in zip(tokens, totals1)]
  return UOp.group(*stores).end(token_block, output, lane).sink(arg=KernelInfo(name="linear_q8_pair", opts_to_apply=()))

@functools.cache
def _q8_linear_concat_kernel(out0:UOp, out1:UOp, raw0:UOp, raw1:UOp, xq:UOp, xd:UOp,
                             out_features0:int, out_features1:int, in_features:int, offset0:UOp, offset1:UOp, shared_raw:bool) -> UOp:
  token_tile = 4 if out0.shape[0] % 4 == 0 else 1
  token_block, output = UOp.range(out0.shape[0] // token_tile, 0), UOp.range(out_features0 + out_features1, 1)
  tokens = tuple(token_block * token_tile + i for i in range(token_tile))
  group_count, lane_count = in_features // 32, min(32, in_features // 32)
  lane = UOp.range(lane_count, 2, axis_type=AxisType.LOCAL)
  first = output < out_features0
  output0, output1 = output.valid(first), (output - out_features0).valid(~first)

  values = [UOp.const(dtypes.float32, 0)] * token_tile
  for group_offset in range(0, group_count, lane_count):
    group = (lane + group_offset).valid(lane + group_offset < group_count)
    block0, block1 = output0 * group_count + group, output1 * group_count + group
    base0, odd0 = offset0.cast(dtypes.weakint) + block0 * 8 + block0 // 2, (block0 & 1).ne(0)
    base1, odd1 = offset1.cast(dtypes.weakint) + block1 * 8 + block1 // 2, (block1 & 1).ne(0)
    accs = [UOp.const(dtypes.int32, 0)] * token_tile
    if shared_raw:
      local_output = first.where(output, output - out_features0)
      block = local_output * group_count + group
      offset = first.where(offset0, offset1).cast(dtypes.weakint)
      base, odd = offset + block * 8 + block // 2, (block & 1).ne(0)
      for word_idx in range(8):
        word = odd.where(raw0[base + 1 + word_idx], (raw0[base + word_idx] >> 16) | (raw0[base + 1 + word_idx] << 16))
        accs = [_amd_dp4a(word, xq[token, group, word_idx], acc) for token,acc in zip(tokens, accs)]
      scale = odd.where(raw0[base] >> 16, raw0[base] & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float()
    else:
      for word_idx in range(8):
        word0 = odd0.where(raw0[base0 + 1 + word_idx], (raw0[base0 + word_idx] >> 16) | (raw0[base0 + 1 + word_idx] << 16))
        word1 = odd1.where(raw1[base1 + 1 + word_idx], (raw1[base1 + word_idx] >> 16) | (raw1[base1 + 1 + word_idx] << 16))
        word = first.where(word0, word1)
        accs = [_amd_dp4a(word, xq[token, group, word_idx], acc) for token,acc in zip(tokens, accs)]
      dbits0 = odd0.where(raw0[base0] >> 16, raw0[base0] & 0xffff).cast(dtypes.uint16)
      dbits1 = odd1.where(raw1[base1] >> 16, raw1[base1] & 0xffff).cast(dtypes.uint16)
      scale = first.where(dbits0, dbits1).bitcast(dtypes.float16).float()
    values = [value + acc.cast(dtypes.float32) * xd[token, group] * scale for token,acc,value in zip(tokens, accs, values)]
  totals = [_amd_wave_sum(value, lane, lane_count) for value in values]
  stores = [out0[token.valid(lane.eq(0) & first), output0].store(total) for token,total in zip(tokens, totals)] + \
           [out1[token.valid(lane.eq(0) & ~first), output1].store(total) for token,total in zip(tokens, totals)]
  return UOp.group(*stores).end(token_block, output, lane).sink(arg=KernelInfo(name="linear_q8_concat", opts_to_apply=()))

@functools.cache
def _q6_linear_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int, raw_offset:int|UOp=0) -> UOp:
  if isinstance(raw_offset, UOp): raw_offset = raw_offset.cast(dtypes.weakint)
  output_tile = 2
  output_block, lane = UOp.range(out_features // output_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  outputs, group_count = tuple(output_block * output_tile + i for i in range(output_tile)), in_features // 32
  type_size, output_size = _GGML_QUANT[14][1], in_features // 256 * _GGML_QUANT[14][1]

  def group_dot(group:UOp, output:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    base = raw_offset + output * output_size + block * type_size
    dots = [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 0)]
    for word_idx in range(8):
      word = UOp.const(dtypes.uint32, 0)
      for byte_idx in range(4):
        pos = subgroup * 32 + word_idx * 4 + byte_idx
        within = pos % 128
        low_byte = raw[base + (pos // 128) * 64 + within % 64]
        low = (low_byte >> ((within // 64) * 4).cast(dtypes.uint8)) & 15
        high_byte = raw[base + 128 + (pos // 128) * 32 + within % 32]
        high = (high_byte >> ((within // 32) * 2).cast(dtypes.uint8)) & 3
        q = (low | (high << 4)).cast(dtypes.uint8).bitcast(dtypes.int8) - 32
        word = word | (q.cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32) << (8 * byte_idx))
      dots[word_idx // 4] = _amd_dp4a(word, xq[0, group, word_idx], dots[word_idx // 4])
    scales = [raw[base + 192 + subgroup * 2 + i].cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    dbits = raw[base + 208].cast(dtypes.uint16) | (raw[base + 209].cast(dtypes.uint16) << 8)
    return (dots[0].float() * scales[0] + dots[1].float() * scales[1]) * xd[0, group] * dbits.bitcast(dtypes.float16).float()

  totals = [_amd_wave_sum(sum((group_dot((lane + offset).valid(lane + offset < group_count), output)
                                for offset in range(0, group_count, 32)), UOp.const(dtypes.float32, 0)), lane, 32) for output in outputs]
  stores = [out[0, output.valid(lane.eq(0))].store(total.cast(out.dtype)) for output,total in zip(outputs, totals)]
  return UOp.group(*stores).end(output_block, lane).sink(arg=KernelInfo(name="linear_q6", opts_to_apply=()))

class Linear(nn.Linear):
  def __init__(self, in_features:int, out_features:int, bias=True):
    # GGUF loading replaces every LLM weight. Lazy zeros avoid constructing hundreds of random-init graphs first,
    # while keeping directly-created test models deterministic and valid.
    self.weight = Tensor.zeros(out_features, in_features)
    self.bias = Tensor.zeros(out_features) if bias else None
    self.in_features, self.out_features = in_features, out_features
    self.ggml_type:int|None = None
    self._raw_uop:UOp|None = None
    self._raw_offset_uop:UOp|None = None
    self._raw_offset_words:int|None = None
  def set_quantized(self, packed:Tensor, ggml_type:int):
    self.weight, self.ggml_type = packed.flatten(), ggml_type
    self._raw_uop = self._raw_offset_uop = self._raw_offset_words = None
  def _packed_offset(self) -> Tensor:
    raw, raw_offset = self.weight.uop, 0
    while raw.op in (Ops.BITCAST, Ops.RESHAPE): raw = raw.src[0]
    while raw.op is Ops.SHRINK:
      raw_offset += raw.src[1].arg * raw.dtype.itemsize
      raw = raw.src[0]
    assert raw_offset % 4 == 0 and raw.dtype == dtypes.uint8
    self._raw_uop = raw
    self._raw_offset_words = raw_offset // 4
    return Tensor([self._raw_offset_words], dtype=dtypes.uint64, device=self.weight.device)
  def _prepare_packed(self):
    self._raw_offset_uop = self._packed_offset().realize().uop
  def prepare(self, x:Tensor) -> tuple[Tensor, Tensor]|None:
    return _q8_quantize(x, int(x.numel()) // self.in_features, self.in_features) \
      if self.ggml_type in (8, 14) and str(self.weight.device).startswith("AMD") else None
  def __call__(self, x:Tensor, prepared:tuple[Tensor, Tensor]|None=None) -> Tensor:
    if self.ggml_type in (8, 14) and str(self.weight.device).startswith("AMD") and (self.ggml_type == 8 or int(x.numel()) == self.in_features):
      tokens = int(x.numel()) // self.in_features
      xq, xd = prepared if prepared is not None else _q8_quantize(x, tokens, self.in_features)
      out = Tensor.empty(tokens, self.out_features, dtype=dtypes.float32, device=x.device)
      if self._raw_uop is None: self._prepare_packed()
      assert self._raw_uop is not None and self._raw_offset_uop is not None
      srcs = (out.uop, self._raw_uop, xq.uop, xd.uop, self._raw_offset_uop)
      params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(srcs)]
      if self.ggml_type == 8:
        if tokens % 16 == 0 and self.out_features % 16 == 0:
          kernel = _q8_linear_wmma_kernel(params[0], params[1], params[2], params[3], self.out_features,
                                          self.in_features, params[4][0] * 4).call(*srcs)
        else:
          params[1] = params[1].replace(dtype=dtypes.uint32, src=(params[1].src[0] * self._raw_uop.dtype.itemsize // 4,),
                                        arg=replace(params[1].arg, dtype=dtypes.uint32))
          kernel = _q8_linear_kernel(params[0], params[1], params[2], params[3], self.out_features,
                                     self.in_features, params[4][0]).call(*srcs)
      else:
        kernel = _q6_linear_kernel(params[0], params[1], params[2], params[3], self.out_features, self.in_features, params[4][0] * 4).call(*srcs)
      out = Tensor(srcs[0].after(kernel)).reshape(*x.shape[:-1], self.out_features)
      return out if self.bias is None else out + self.bias
    return super().__call__(x)

def _q8_linear_pair(first:Linear, second:Linear, x:Tensor, prepared:tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
  assert first.ggml_type == second.ggml_type == 8 and first.in_features == second.in_features
  if (tokens := int(x.numel()) // first.in_features) % 16 == 0: return first(x, prepared), second(x, prepared)
  if first._raw_uop is None: first._prepare_packed()
  if second._raw_uop is None: second._prepare_packed()
  assert first._raw_uop is not None and first._raw_offset_uop is not None
  assert second._raw_uop is not None and second._raw_offset_uop is not None
  tokens, xq, xd = tokens, *prepared
  out0 = Tensor.empty(tokens, first.out_features, dtype=dtypes.float32, device=x.device)
  out1 = Tensor.empty(tokens, second.out_features, dtype=dtypes.float32, device=x.device)
  srcs = (out0.uop, out1.uop, first._raw_uop, second._raw_uop, xq.uop, xd.uop, first._raw_offset_uop, second._raw_offset_uop)
  params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(srcs)]
  for i in (2, 3):
    params[i] = params[i].replace(dtype=dtypes.uint32, src=(params[i].src[0] * srcs[i].dtype.itemsize // 4,),
                                  arg=replace(params[i].arg, dtype=dtypes.uint32))
  kernel_args = (params[0], params[1], params[2], params[3], params[4], params[5], first.out_features,
                 second.out_features, first.in_features, params[6][0], params[7][0])
  kernel = _q8_linear_pair_kernel(*kernel_args).call(*srcs)
  return (Tensor(srcs[0].after(kernel)).reshape(*x.shape[:-1], first.out_features),
          Tensor(srcs[1].after(kernel)).reshape(*x.shape[:-1], second.out_features))

def _iq3_group_dot(raw:UOp, lut:UOp, xq:UOp, xd:UOp, expert:UOp, xidx:UOp, group:UOp, output:UOp,
                   out_features:int, in_features:int) -> UOp:
  type_size, expert_size = _GGML_QUANT[21][1], out_features * in_features // 256 * _GGML_QUANT[21][1]
  block, subgroup = group // 8, group % 8
  base = expert * expert_size + output * (in_features // 256 * type_size) + block * type_size
  def load_word(byte_offset:UOp) -> UOp:
    word_index, half_aligned = byte_offset // 4, (byte_offset & 2).ne(0)
    word = raw[word_index]
    return half_aligned.where((word >> 16) | (raw[word_index + 1] << 16), word)
  qs = (load_word(base + 2 + subgroup * 8), load_word(base + 6 + subgroup * 8))
  qh_word = load_word(base + 66 + (subgroup // 4) * 4)
  qh = (qh_word >> (8 * (subgroup % 4)).cast(dtypes.uint32)) & 255
  signs = load_word(base + 74 + subgroup * 4)
  dot = UOp.const(dtypes.int32, 0)
  for word_idx in range(8):
    qi = ((qs[word_idx // 4] >> (8 * (word_idx % 4))) & 255).cast(dtypes.uint16) + \
      (((qh >> word_idx) & 1).cast(dtypes.uint16) << 8)
    sign_bits = (signs >> (word_idx * 4)) & 15
    sign_mask = lut[512 + sign_bits.cast(dtypes.weakint)]
    word = (lut[qi.cast(dtypes.weakint)] ^ sign_mask) + (sign_mask & 0x01010101)
    dot = _amd_dp4a(word, xq[xidx, group, word_idx], dot)
  scale = 1 + 2 * ((load_word(base + 106) >> (4 * subgroup).cast(dtypes.uint32)) & 15).cast(dtypes.float32)
  dbits = (load_word(base) & 0xffff).cast(dtypes.uint16)
  return dot.cast(dtypes.float32) * xd[xidx, group] * dbits.bitcast(dtypes.float16).float() * scale

def _packed_group_dot(raw:UOp, lut:UOp, xq:UOp, xd:UOp, expert:UOp, xidx:UOp, group:UOp, output:UOp,
                      out_features:int, in_features:int, ggml_type:int) -> UOp:
  if ggml_type == 21: return _iq3_group_dot(raw, lut, xq, xd, expert, xidx, group, output, out_features, in_features)
  type_size = _GGML_QUANT[ggml_type][1]
  block, subgroup = group // 8, group % 8
  base = expert * (out_features * in_features // 256 * type_size) + output * (in_features // 256 * type_size) + block * type_size
  dot = UOp.const(dtypes.int32, 0)
  if ggml_type == 23:  # IQ4_XS
    for word_idx in range(8):
      qbase, shift = base + 8 + subgroup * 16 + (word_idx % 4) * 4, 4 * (word_idx // 4)
      packed = raw[qbase // 4]
      nibbles = tuple((packed >> (8*i + shift)) & 15 for i in range(4))
      low = lut[(nibbles[0] | (nibbles[1] << 4)).cast(dtypes.weakint)].cast(dtypes.uint32)
      high = lut[(nibbles[2] | (nibbles[3] << 4)).cast(dtypes.weakint)].cast(dtypes.uint32)
      dot = _amd_dp4a(low | (high << 16), xq[xidx, group, word_idx], dot)
    low_offset = base + 4 + subgroup // 2
    low_byte = (raw[low_offset // 4] >> (8 * (low_offset % 4)).cast(dtypes.uint32)) & 255
    low = (low_byte >> (4 * (subgroup % 2)).cast(dtypes.uint32)) & 15
    high_word = (raw[base // 4] >> 16).cast(dtypes.uint16)
    scale = ((low.cast(dtypes.uint16) | (((high_word >> (2 * subgroup).cast(dtypes.uint16)) & 3) << 4)).cast(dtypes.uint8).
             bitcast(dtypes.int8)-32).float()
    dbits = (raw[base // 4] & 0xffff).cast(dtypes.uint16)
    return dot.cast(dtypes.float32) * xd[xidx, group] * dbits.bitcast(dtypes.float16).float() * scale
  assert ggml_type == 14  # Q6_K
  dots = [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 0)]
  for word_idx in range(8):
    word = UOp.const(dtypes.uint32, 0)
    for byte_idx in range(4):
      pos = subgroup * 32 + word_idx * 4 + byte_idx
      within = pos % 128
      low_byte = raw[base + (pos // 128) * 64 + within % 64]
      low = (low_byte >> ((within // 64) * 4).cast(dtypes.uint8)) & 15
      high_byte = raw[base + 128 + (pos // 128) * 32 + within % 32]
      high = (high_byte >> ((within // 32) * 2).cast(dtypes.uint8)) & 3
      q = (low | (high << 4)).cast(dtypes.uint8).bitcast(dtypes.int8) - 32
      word = word | (q.cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32) << (8 * byte_idx))
    dots[word_idx // 4] = _amd_dp4a(word, xq[xidx, group, word_idx], dots[word_idx // 4])
  scales = [raw[base + 192 + subgroup * 2 + i].cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
  dbits = raw[base + 208].cast(dtypes.uint16) | (raw[base + 209].cast(dtypes.uint16) << 8)
  return (dots[0].float() * scales[0] + dots[1].float() * scales[1]) * xd[xidx, group] * dbits.bitcast(dtypes.float16).float()

@functools.cache
def _packed_expert_kernel(out:UOp, raw:UOp, sel:UOp, xq:UOp, xd:UOp, lut:UOp,
                          out_features:int, in_features:int, ggml_type:int, routes_per_input:int) -> UOp:
  prefill = out.shape[0] > routes_per_input
  output_tile = 2 if ggml_type == 21 and not prefill else 8 if ggml_type == 21 or prefill and ggml_type == 23 else 4
  wave_count = 2 if prefill else 1
  if prefill:
    output_block, route = UOp.range(out_features // (output_tile * wave_count), 0), UOp.range(out.shape[0], 1)
  else:
    route, output_block = UOp.range(out.shape[0], 0), UOp.range(out_features // output_tile, 1)
  group_count, lane_count = in_features // 32, min(32, in_features // 32)
  lane = UOp.range(lane_count, 2, axis_type=AxisType.LOCAL)
  wave = UOp.range(wave_count, 3, axis_type=AxisType.LOCAL) if wave_count > 1 else UOp.const(dtypes.weakint, 0)
  outputs = tuple(output_block * output_tile * wave_count + wave * output_tile + i for i in range(output_tile)) if wave_count > 1 else \
    tuple(output_block * output_tile + i for i in range(output_tile))
  if ggml_type in (21, 23):
    raw = raw.replace(dtype=dtypes.uint32, src=(raw.src[0] * raw.dtype.itemsize // 4,), arg=replace(raw.arg, dtype=dtypes.uint32))

  expert, xidx = sel[route].cast(dtypes.weakint), route // routes_per_input
  values = [sum((_packed_group_dot(raw, lut, xq, xd, expert, xidx, (lane + offset).valid(lane + offset < group_count), output,
                                    out_features, in_features, ggml_type)
                 for offset in range(0, group_count, lane_count)), UOp.const(dtypes.float32, 0)) for output in outputs]
  totals = [_amd_wave_sum(value, lane, lane_count, wave if wave_count > 1 else None) for value in values]
  stores = [out[route.valid(lane.eq(0)), output].store(total.cast(out.dtype)) for output,total in zip(outputs, totals)]
  result = UOp.group(*stores).end(lane, wave, output_block, route) if wave_count > 1 else \
    UOp.group(*stores).end(route, output_block, lane)
  return result.sink(
    arg=KernelInfo(name=f"expert_q8_{ggml_type}", opts_to_apply=()))

@functools.cache
def _expert_lut(device:str, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common
  if ggml_type == 21:
    sign_masks = [sum((0xff << (8*i)) for i in range(4) if signs & (1 << i)) for signs in range(16)]
    return Tensor([*ggml_common.iq3s_grid, *sign_masks], dtype=dtypes.uint32, device=device).contiguous().realize()
  values = [((ggml_common.kvalues_iq4nl[i] & 0xff) | ((ggml_common.kvalues_iq4nl[j] & 0xff) << 8)) for j in range(16) for i in range(16)]
  return Tensor(values, dtype=dtypes.uint16, device=device).contiguous().realize()

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2).to(device)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).to(device).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.num_experts, self.in_features, self.out_features = num_experts, in_features, out_features
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    self.ggml_type:int|None = None
  def set_quantized(self, weight:Tensor, packed:Tensor, ggml_type:int):
    assert weight.shape == (self.num_experts, self.out_features, self.in_features)
    self.weight, self.ggml_type = packed.flatten(), ggml_type
  def prepare(self, x:Tensor) -> tuple[Tensor, Tensor]:
    return _q8_quantize(x, int(x.numel()) // self.in_features, self.in_features)
  def __call__(self, sel:Tensor, x:Tensor, prepared:tuple[Tensor, Tensor]|None=None) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    ggml_type = self.ggml_type
    if ggml_type in (14, 21, 23) and str(self.weight.device).startswith("AMD"):
      input_count = int(x.numel()) // self.in_features
      routes_per_input = int(sel.numel()) // input_count
      xq, xd = prepared if prepared is not None else self.prepare(x)
      flat_sel = sel if len(sel.shape) == 1 else sel.flatten().clone()
      out = Tensor.empty(int(sel.numel()), self.out_features, dtype=dtypes.float32, device=x.device)
      out = Tensor.custom_kernel(out, self.weight, flat_sel, xq, xd, _expert_lut(str(x.device), ggml_type),
        fxn=lambda out,raw,sel,xq,xd,lut:_packed_expert_kernel(out, raw, sel, xq, xd, lut, self.out_features,
                                                              self.in_features, ggml_type, routes_per_input))[0]
      return out if len(sel.shape) == 1 else out.reshape(*sel.shape, self.out_features)
    if self.ggml_type is None: weight = self.weight[sel]
    else:
      packed = self.weight.reshape(self.num_experts, -1)[sel].flatten()
      weight = ggml_data_to_tensor(packed, int(sel.numel()) * self.out_features * self.in_features,
                                    self.ggml_type, contiguous=False).reshape(*sel.shape, self.out_features, self.in_features)
      if getenv("HALF", 1): weight = weight.cast('float16')
    return (x.unsqueeze(-2) @ weight.transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def _topk_256_sort(values:UOp, indices:UOp, next_values:UOp, next_indices:UOp, ready:UOp, lane:UOp) -> tuple[UOp, UOp, UOp]:
  for size in (2, 4, 8, 16, 32, 64, 128, 256):
    for stride in (128, 64, 32, 16, 8, 4, 2, 1)[9-size.bit_length():]:
      partner = lane ^ stride
      a_value, b_value = values.after(ready)[lane], values.after(ready)[partner]
      a_index, b_index = indices.after(ready)[lane], indices.after(ready)[partner]
      a_first = (a_value < b_value) | (a_value.eq(b_value) & (a_index > b_index))
      want_first = (lane & stride).eq(0).eq((lane & size).eq(0))
      take_a = want_first.eq(a_first)
      ready = UOp.group(next_values.after(ready)[lane].store(take_a.where(a_value, b_value)),
                        next_indices.after(ready)[lane].store(take_a.where(a_index, b_index))).barrier()
      values, next_values, indices, next_indices = next_values, values, next_indices, indices
  return values, indices, ready

@functools.cache
def _topk_256_kernel(out:UOp, sel:UOp, x:UOp, k:int, softmax:bool=False) -> UOp:
  outer, lane = UOp.range(out.shape[0], 0), UOp.range(256, 1, axis_type=AxisType.LOCAL)
  values = UOp.placeholder((256,), x.dtype, 0, addrspace=AddrSpace.LOCAL)
  indices = UOp.placeholder((256,), dtypes.int32, 1, addrspace=AddrSpace.LOCAL)
  next_values = UOp.placeholder((256,), x.dtype, 2, addrspace=AddrSpace.LOCAL)
  next_indices = UOp.placeholder((256,), dtypes.int32, 3, addrspace=AddrSpace.LOCAL)
  ready = UOp.group(values.after(outer)[lane].store(x[outer, lane]),
                    indices.after(outer)[lane].store(lane.cast(dtypes.int32))).barrier()
  values, indices, ready = _topk_256_sort(values, indices, next_values, next_indices, ready, lane)
  valid = lane < k
  src = 256 - k + lane
  value = values.after(ready)[src]
  if softmax:
    max_value = values.after(ready)[255]
    value = (value - max_value).exp() / sum(((values.after(ready)[256-k+i] - max_value).exp() for i in range(k)),
                                            UOp.const(x.dtype, 0))
  stores = (out[outer, lane.valid(valid)].store(value),
            sel[outer, lane.valid(valid)].store(indices.after(ready)[src]))
  return UOp.group(*stores).end(outer, lane).sink(arg=KernelInfo(name=f"topk_256{'_softmax' if softmax else ''}", opts_to_apply=()))

@functools.cache
def _biased_sigmoid_topk_kernel(out:UOp, sel:UOp, x:UOp, bias:UOp, k:int, normalize:bool) -> UOp:
  outer, lane = UOp.range(out.shape[0], 0), UOp.range(256, 1, axis_type=AxisType.LOCAL)
  values = UOp.placeholder((256,), x.dtype, 0, addrspace=AddrSpace.LOCAL)
  indices = UOp.placeholder((256,), dtypes.int32, 1, addrspace=AddrSpace.LOCAL)
  next_values = UOp.placeholder((256,), x.dtype, 2, addrspace=AddrSpace.LOCAL)
  next_indices = UOp.placeholder((256,), dtypes.int32, 3, addrspace=AddrSpace.LOCAL)
  score = x[outer, lane].sigmoid() + bias[lane]
  ready = UOp.group(values.after(outer)[lane].store(score),
                    indices.after(outer)[lane].store(lane.cast(dtypes.int32))).barrier()
  _, indices, ready = _topk_256_sort(values, indices, next_values, next_indices, ready, lane)
  valid = lane < k
  index = indices.after(ready)[(256 - k + lane).valid(valid)]
  prob = x[outer, index.cast(dtypes.weakint)].sigmoid()
  if normalize:
    prob = prob / sum((x[outer, indices.after(ready)[256-k+i].cast(dtypes.weakint)].sigmoid() for i in range(k)),
                      UOp.const(x.dtype, 0))
  stores = (out[outer, lane.valid(valid)].store(prob), sel[outer, lane.valid(valid)].store(index))
  return UOp.group(*stores).end(outer, lane).sink(arg=KernelInfo(name="biased_sigmoid_topk_256", opts_to_apply=()))

def biased_sigmoid_topk(x:Tensor, bias:Tensor, k:int, normalize:bool) -> tuple[Tensor, Tensor]:
  assert x.shape[-1] == bias.shape[0] == 256
  outer = int(x.numel()) // 256
  values = Tensor.empty(outer, k, dtype=x.dtype, device=x.device)
  indices = Tensor.empty(outer, k, dtype=dtypes.int32, device=x.device)
  return tuple(Tensor.custom_kernel(values, indices, x.contiguous(), bias.contiguous(),
    fxn=lambda out,sel,x,bias:_biased_sigmoid_topk_kernel(out, sel, x, bias, k, normalize))[:2])  # type: ignore[return-value]

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  outer = int(x.numel()) // n
  if n == 256 and outer <= 256 and str(x.device).startswith("AMD"):
    values = Tensor.empty(outer, k, dtype=x.dtype, device=x.device)
    indices = Tensor.empty(outer, k, dtype=dtypes.int32, device=x.device)
    values, indices = Tensor.custom_kernel(values, indices, x.reshape(outer, n),
      fxn=lambda out,sel,x:_topk_256_kernel(out, sel, x, k))[:2]
    return values.reshape(*x.shape[:-1], k), indices.reshape(*x.shape[:-1], k)
  vals = Tensor.arange(n).to(x.device).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).to(x.device).reshape(1,1,n,1) < Tensor.arange(n).to(x.device).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

def topk_softmax(x:Tensor, k:int) -> tuple[Tensor, Tensor]:
  n, outer = x.shape[-1], int(x.numel()) // x.shape[-1]
  if n == 256 and outer <= 256 and str(x.device).startswith("AMD"):
    values = Tensor.empty(outer, k, dtype=x.dtype, device=x.device)
    indices = Tensor.empty(outer, k, dtype=dtypes.int32, device=x.device)
    values, indices = Tensor.custom_kernel(values, indices, x.reshape(outer, n),
      fxn=lambda out,sel,x:_topk_256_kernel(out, sel, x, k, softmax=True))[:2]
    return values.reshape(*x.shape[:-1], k), indices.reshape(*x.shape[:-1], k)
  values, indices = pairwise_topk(x, k)
  return values.softmax(-1), indices

@functools.cache
def _inverse_unit_lower_kernel(out:UOp, x:UOp, n:int) -> UOp:
  outer_count = 1
  for dim in out.shape[:-2]:
    assert isinstance(dim, int)
    outer_count *= dim
  outer, lane = UOp.range(outer_count, 0), UOp.range(n, 1, axis_type=AxisType.LOCAL)
  raw = UOp.placeholder((n*n,), x.dtype, 0, addrspace=AddrSpace.LOCAL)
  solved = UOp.placeholder((n*n,), x.dtype, 1, addrspace=AddrSpace.LOCAL)
  ready = UOp.group(*(raw[row*n+lane].store(x.flatten()[outer*n*n+row*n+lane]) for row in range(n))).barrier()
  for row in range(n):
    base, previous = raw.after(ready), solved.after(ready)
    value = base[row*n+lane] + sum((base[row*n+i] * previous[i*n+lane] for i in range(row)), UOp.const(x.dtype, 0))
    ready = solved.after(ready)[row*n+lane].store((lane < row).where(value, UOp.const(x.dtype, 0))).barrier()
  result = solved.after(ready)
  stores = [out.flatten()[outer*n*n+row*n+lane].store(lane.eq(row).where(UOp.const(x.dtype, 1), result[row*n+lane]))
            for row in range(n)]
  return UOp.group(*stores).end(outer, lane).sink(arg=KernelInfo(name="inverse_unit_lower", opts_to_apply=()))

def inverse_unit_lower(x:Tensor) -> Tensor:
  """Reference-ordered inverse of I-x for a strictly lower-triangular x."""
  n = x.shape[-1]
  assert isinstance(n, int)
  if n == 64 and str(x.device).startswith("AMD"):
    out = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device)
    return Tensor.custom_kernel(out, x, fxn=lambda out,x:_inverse_unit_lower_kernel(out, x, n))[0]
  rows = [x[..., 0, :].const_like(0)]
  for i in range(1, n):
    prefix = x[..., i, :i]
    previous = Tensor.stack(*rows, dim=-2)[..., :, :i]
    rows.append((prefix + (prefix.unsqueeze(-1) * previous).sum(-2)).pad((0, n-i)))
  return Tensor.stack(*rows, dim=-2) + Tensor.eye(n, dtype=x.dtype).to(x.device)

def l2norm(x:Tensor) -> Tensor: return x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config
    self.pending_state:tuple[Tensor, Tensor]|None = None

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      prepared = self.ffn_gate_exps.prepare(h) \
        if self.ffn_gate_exps.ggml_type in (14, 21, 23) and str(h.device).startswith("AMD") else None
      logits = self.ffn_gate_inp(x, prepared)
      if hasattr(self, 'exp_probs_b'):
        if logits.shape[-1] == 256 and str(logits.device).startswith("AMD"):
          probs, sel = biased_sigmoid_topk(logits, self.exp_probs_b["bias"], self.config.num_experts_per_tok,
                                           self.config.norm_topk_prob)
          probs, sel = probs.reshape(*logits.shape[:-1], -1), sel.reshape(*logits.shape[:-1], -1)
        else:
          probs = logits.sigmoid()
          _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
          probs = probs.gather(-1, sel)
          if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        if self.config.norm_topk_prob:
          probs, sel = topk_softmax(logits, self.config.num_experts_per_tok)
        else:
          _, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
          probs = logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      if prepared is not None:
        flat_sel = sel.flatten().clone()
        gate, up = self.ffn_gate_exps(flat_sel, h, prepared), self.ffn_up_exps(flat_sel, h, prepared)
        x_down = self.ffn_down_exps(flat_sel, gate, _q8_silu_mul(gate, up, self.config.hidden_dim))
        x_down = x_down.reshape(*sel.shape, self.config.dim)
      else: x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        if prepared is not None and self.ffn_gate_shexp.ggml_type == self.ffn_up_shexp.ggml_type == 8:
          gate, up = _q8_linear_pair(self.ffn_gate_shexp, self.ffn_up_shexp, x, prepared)
        else: gate, up = self.ffn_gate_shexp(x, prepared), self.ffn_up_shexp(x, prepared)
        if self.ffn_down_shexp.ggml_type in (8, 14) and str(gate.device).startswith("AMD"):
          shexp = self.ffn_down_shexp(gate, _q8_silu_mul(gate, up, self.config.shared_expert_dim))
        else: shexp = self.ffn_down_shexp(gate.silu().contiguous() * up)
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    prepared = self.ffn_gate.prepare(x)
    if prepared is not None and self.ffn_gate.ggml_type == self.ffn_up.ggml_type == 8:
      gate, up = _q8_linear_pair(self.ffn_gate, self.ffn_up, x, prepared)
    else: gate, up = self.ffn_gate(x, prepared), self.ffn_up(x, prepared)
    return self.ffn_down(gate.silu().contiguous() * up)

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None, valid_len:int|UOp|None=None):
    self._init_state(x)
    if hasattr(self, 'ssm_a'):
      self.pending_state = None
      @function(precompile=True, allow_implicit=True)
      def _run_stateful(x:Tensor, start_pos:int|UOp, valid_len:int|UOp|None):
        h = x + self._attention(self.attn_norm(x), start_pos, use_flash, kv_len, valid_len)
        out = (h + self._feed_forward(self.ffn_norm(h))).contiguous()
        assert self.pending_state is not None
        return out, *self.pending_state
      out, conv_state, recurrent_state = _run_stateful(x, start_pos, valid_len)
      stores = (getattr(self, "conv_state").uop.store(conv_state.uop), getattr(self, "recurrent_state").uop.store(recurrent_state.uop))
      state = getattr(self, "recurrent_state").uop.after(*stores)
      return Tensor(out.uop.after(state))
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos, use_flash, kv_len)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return function(precompile=True, allow_implicit=True)(_run)(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None) -> Tensor:
    prepared = self.attn_q.prepare(x)
    q = self.attn_q(x, prepared)
    if prepared is not None and self.attn_k.ggml_type == self.attn_v.ggml_type == 8:
      k, v = _q8_linear_pair(self.attn_k, self.attn_v, x, prepared)
    else: k, v = self.attn_k(x, prepared), self.attn_v(x, prepared)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    assigned_kv = Tensor(self.cache_kv.uop.after(
      self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)))
    cache_len = start_pos + T if kv_len is None else kv_len
    k = assigned_kv[0, :, :, 0:cache_len, :]
    v = assigned_kv[1, :, :, 0:cache_len, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_causal = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    flash_decode = resolve(T == 1) and kv_len is not None and str(x.device).startswith("AMD") and self.config.head_dim == 256
    if flash_decode:
      decode_len = kv_len if isinstance(kv_len, int) else self.config.max_context
      decode_pos = (start_pos.unbind()[0] if isinstance(start_pos, UOp) else start_pos) + 1
      from extra.gemm.amd_flash_attention import amd_flash_attention_decode
      attn = amd_flash_attention_decode(q.half(), assigned_kv, decode_pos, decode_len)
    elif use_flash:
      from extra.gemm.amd_flash_attention import amd_flash_attention_causal_cached
      start = start_pos.unbind()[0] if isinstance(start_pos, UOp) else start_pos
      valid = valid_len.unbind()[0] if isinstance(valid_len, UOp) else valid_len
      valid_kv_len, key_limit = start + T, start + valid if valid is not None else None
      q_flat = q.half().reshape(B*self.config.n_heads, T, self.config.head_dim)
      out = Tensor.empty(B*self.config.n_heads, q_flat.shape[1], self.config.head_dim, dtype="float32", device=x.device)
      attn = Tensor.custom_kernel(out, q_flat, assigned_kv,
        fxn=functools.partial(amd_flash_attention_causal_cached, valid_kv_len=valid_kv_len, key_limit=key_limit))[0] \
        .reshape(B, self.config.n_heads, q_flat.shape[1], -1)
    else:
      mask:Tensor|None
      if kv_len is not None:
        mask = None if resolve(T == 1) and self.config.ssm is not None else \
          Tensor.full((1, 1, 1, kv_len), float("-inf"), dtype=x.dtype, device=x.device, buffer=False).triu(start_pos+1)
      else:
        mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device, buffer=False).triu(start_pos+1) \
          if resolve(T != 1) else None
      attn = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=mask, enable_gqa=True)  # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      # Decode uses fixed-size KV buckets. Unwritten entries must be zero: masking happens after QK, so values left
      # uninitialized by Tensor.empty can inject NaNs before the mask is applied.
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context+192, self.config.head_dim,
                                   dtype="float16", device=x.device).contiguous()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context+192, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context+192, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context+192, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = Linear(config.dim, self.conv_channels, bias=False), Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False), Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta_alpha_weight:Tensor|None = None
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, use_flash:bool=False, kv_len:int|UOp|None=None,
                 valid_len:int|UOp|None=None) -> Tensor:
    B, T, _ = x.shape
    conv_state, initial_state = self.conv_state, self.recurrent_state

    if T == 1:
      x = x.half()
      prepared = self.attn_gate.prepare(x)
      if prepared is not None and self.attn_gate.ggml_type == self.attn_qkv.ggml_type == 8:
        out_gate, qkv = _q8_linear_pair(self.attn_gate, self.attn_qkv, x, prepared)
      else: out_gate, qkv = self.attn_gate(x, prepared), self.attn_qkv(x, prepared)
      beta, alpha = (x @ self.ssm_beta_alpha_weight.T).split(self.num_v_heads, dim=-1) if self.ssm_beta_alpha_weight is not None else \
        (self.ssm_beta(x, prepared), self.ssm_alpha(x, prepared))
      out_gate = out_gate.reshape(B, 1, self.num_v_heads, self.head_v_dim)
      beta, alpha = beta.sigmoid(), ((alpha.float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).exp()
      conv_window = conv_state.cat(qkv, dim=1)
      conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
      q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
      q = l2norm(q.reshape(B, self.num_k_heads, self.head_k_dim)).repeat(1, self.num_v_heads//self.num_k_heads, 1)
      k = l2norm(k.reshape(B, self.num_k_heads, self.head_k_dim)).repeat(1, self.num_v_heads//self.num_k_heads, 1)
      v = v.reshape(B, self.num_v_heads, self.head_v_dim)
      beta, alpha = beta.reshape(B, self.num_v_heads, 1, 1), alpha.reshape(B, self.num_v_heads, 1, 1)
      q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)
      state_dots = initial_state @ k.cat(q, dim=-1)
      state_k, state_q = state_dots[..., :1] * alpha, state_dots[..., 1:] * alpha
      delta = (v - state_k) * beta
      recurrent_state = initial_state * alpha + delta @ k.transpose(-1, -2)
      self.pending_state = (conv_window[:, 1:, :].cast(self.conv_state.dtype).contiguous(),
                            recurrent_state.cast(self.recurrent_state.dtype).contiguous())
      core = state_q + delta * (k.transpose(-1, -2) @ q)
      core_attn_out = self.ssm_norm(core.squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
      return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

    # Batched projections and causal depthwise convolution.
    x = x.half()
    prepared = self.attn_gate.prepare(x)
    if prepared is not None and self.attn_gate.ggml_type == self.attn_qkv.ggml_type == 8:
      out_gate, qkv = _q8_linear_pair(self.attn_gate, self.attn_qkv, x, prepared)
    else: out_gate, qkv = self.attn_gate(x, prepared), self.attn_qkv(x, prepared)
    beta, alpha = (x @ self.ssm_beta_alpha_weight.T).split(self.num_v_heads, dim=-1) if self.ssm_beta_alpha_weight is not None else \
      (self.ssm_beta(x, prepared), self.ssm_alpha(x, prepared))
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = beta.sigmoid().reshape(B, T, self.num_v_heads)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, T, self.num_v_heads)
    if valid_len is not None:
      active = (Tensor.arange(T).to(x.device) < Tensor(valid_len, device=x.device)).reshape(1, T, 1)
      beta, log_alpha = beta * active, log_alpha * active
    conv_window = conv_state.cat(qkv, dim=1)
    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = l2norm(q.reshape(B, T, self.num_k_heads, self.head_k_dim)).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1)
    k = l2norm(k.reshape(B, T, self.num_k_heads, self.head_k_dim)).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)

    # Chunked gated delta rule. The strictly-lower update is the triangular solve from the reference implementation.
    q, k, v, beta, log_alpha = [z.transpose(1, 2).float() for z in (q, k, v, beta, log_alpha)]
    q = q * self.head_k_dim**-0.5
    state = initial_state.transpose(-1, -2).float()
    core_chunks = []
    for start in range(0, T, 64):
      qc, kc, vc, bc, gc = q[:,:,start:start+64], k[:,:,start:start+64], v[:,:,start:start+64], \
                            beta[:,:,start:start+64], log_alpha[:,:,start:start+64]
      chunk_len = qc.shape[2]
      g = (gc @ Tensor.ones(chunk_len, chunk_len, dtype=gc.dtype, device=gc.device).tril().T).contiguous()
      decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).exp().tril().contiguous()
      base = (-(kc * bc.unsqueeze(-1) @ kc.transpose(-1, -2) * decay).tril(-1)).contiguous()
      attn = inverse_unit_lower(base)
      value = attn @ (vc * bc.unsqueeze(-1))
      k_cumdecay = attn @ (kc * bc.unsqueeze(-1) * g.exp().unsqueeze(-1))
      value = value - k_cumdecay @ state
      core_chunks.append((qc * g.exp().unsqueeze(-1)) @ state + (qc @ kc.transpose(-1, -2) * decay) @ value)
      state = state * g[..., -1, None, None].exp() + \
              (kc * (g[..., -1, None] - g).exp().unsqueeze(-1)).transpose(-1, -2) @ value
    core_attn_out = functools.reduce(lambda a,b: a.cat(b, dim=2), core_chunks)
    core_attn_out = self.ssm_norm(core_attn_out.transpose(1, 2))
    out = self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, T, -1).cast(x.dtype)).contiguous()

    state_pos = T if valid_len is None else valid_len
    self.pending_state = (conv_window[:, state_pos:state_pos+self.ssm_conv_kernel-1, :].cast(self.conv_state.dtype).contiguous(),
                          state.transpose(-1, -2).cast(self.recurrent_state.dtype).contiguous())
    return out

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim,
                                          dtype=dtypes.float16, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.parameter_count = 0
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._state_checkpoints: list[Tensor] = []
    self._state_checkpoint_pos = 0
    self._save_state_jit: TinyJit|None = None
    self._restore_state_jit: TinyJit|None = None
    self._warming_up = False
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.flash_prefill_jit = TinyJit(functools.partial(self.forward, use_flash=True))
    self.sample_prefill_jit = TinyJit(functools.partial(self.forward, sample=True))
    self.rollout_jits:dict[int, TinyJit] = {}
    self.sample_rollout_jits:dict[int, TinyJit] = {}

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False, kv_len:int|UOp|None=None,
              valid_len:int|UOp|None=None, sample:bool=False) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos, use_flash, kv_len, valid_len)
    last = x[:, tokens.shape[1]-1:tokens.shape[1]] if valid_len is None else x[:, valid_len-1:valid_len]
    logits = self.output(self.output_norm(last))[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    if not sample: return logits.argmax(-1, keepdim=True)
    return (logits / temperature - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def forward_recurrent_decode(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, decode_len:int,
                               valid_len:int|UOp|None=None, sample:bool=False) -> Tensor:
    return self.forward(tokens, start_pos, temperature, kv_len=decode_len, valid_len=valid_len, sample=sample)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False,
               valid_len:int|UOp|None=None, sample:bool=False) -> Tensor:
    jit_kwargs = {"valid_len":valid_len}
    if resolve(tokens.shape[1] == 1):
      pos = start_pos.unbind()[1] if isinstance(start_pos, UOp) else start_pos
      if self.has_recurrent_block:
        short_decode_len = min(8192, self.max_context)
        key = short_decode_len if pos < short_decode_len else self.max_context
      else:
        min_bucket = max(1, getenv("DECODE_BUCKET", 256))
        kv_len = key = min(self.max_context, max(min_bucket, 1 << pos.bit_length()))
      rollout_jits = self.sample_rollout_jits if sample else self.rollout_jits
      if key not in rollout_jits:
        rollout_jits[key] = TinyJit(functools.partial(self.forward_recurrent_decode, decode_len=key, sample=sample) if self.has_recurrent_block else
                                    functools.partial(self.forward, kv_len=kv_len, sample=sample))
      jit = rollout_jits[key]
    else:
      jit = self.sample_prefill_jit if sample else self.flash_prefill_jit if use_flash else self.prefill_jit
    ret = jit(tokens.contiguous(), start_pos, temperature, **jit_kwargs)
    return ret[0] if isinstance(ret, tuple) else ret

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    kv, state_dict = gguf_load(gguf)

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict)
    model = Transformer(config)
    load_device = next(iter(state_dict.values())).device
    for param in nn.state.get_parameters(model): param.replace(param.to(load_device))
    model.parameter_count = sum(int(weight.numel()) for weight in state_dict.values())
    packed_weights:set[str] = set()
    packed_linears:list[Linear] = []
    def resolve_owner(path:list[str]):
      obj = model
      for part in path: obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
      return obj
    for name, weight in state_dict.items():
      parts = name.split('.')
      quantization = get_ggml_quantization(weight)
      if quantization is not None and quantization[1] in (8, 14) and parts[-1] == "weight" and isinstance(owner:=resolve_owner(parts[:-1]), Linear):
        owner.set_quantized(*quantization)
        packed_linears.append(owner)
        state_dict[name], packed_weights = owner.weight, packed_weights | {name}
      elif len(parts) == 4 and parts[0] == "blk" and parts[2].endswith("_exps") and parts[3] == "weight" and quantization is not None:
        expert_weights = getattr(model.blk[int(parts[1])], parts[2])
        expert_weights.set_quantized(weight, *quantization)
        state_dict[name], packed_weights = expert_weights.weight, packed_weights | {name}

    state_dict = {k:v if k in packed_weights else v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    recurrent_weights = []
    for block in model.blk:
      if isinstance(block, GatedDeltaNetBlock):
        block.ssm_beta_alpha_weight = block.ssm_beta.weight.cat(block.ssm_alpha.weight).contiguous()
        recurrent_weights.append(block.ssm_beta_alpha_weight)
    Tensor.realize(*recurrent_weights)
    # Custom kernels need the shared GGUF buffer and byte offset before function tracing disables device access.
    packed_offsets = [linear._packed_offset() for linear in packed_linears]
    Tensor.realize(*packed_offsets)
    for linear,offset in zip(packed_linears, packed_offsets): linear._raw_offset_uop = offset.uop
    expert_types = {getattr(block, name).ggml_type for block in model.blk if hasattr(block, "ffn_gate_exps")
                    for name in ("ffn_gate_exps", "ffn_down_exps")}
    for ggml_type in expert_types:
      if ggml_type in (14, 21, 23) and str(model.token_embd.weight.device).startswith("AMD"):
        _expert_lut(str(model.token_embd.weight.device), ggml_type)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    # Recurrent state has no token dimension to slice. Roll back to its latest aligned checkpoint so resumed flash
    # prefill uses the same global chunk boundaries as a full prompt.
    if self.has_recurrent_block:
      return self._state_checkpoint_pos if prefix_len >= self._state_checkpoint_pos else 0
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def _recurrent_states(self) -> list[Tensor]:
    return [getattr(block, name) for block in self.blk for name in ("conv_state", "recurrent_state") if hasattr(block, name)]

  def _init_state_checkpoints(self):
    if not self._state_checkpoints:
      self._state_checkpoints = [Tensor.zeros_like(state).contiguous().realize() for state in self._recurrent_states()]
      def copy_jit(pairs:list[tuple[Tensor, Tensor]]) -> TinyJit:
        def copy_states() -> Tensor:
          copies = [dest.assign(src) for dest,src in pairs]
          Tensor.realize(*copies)
          return copies[-1]
        jit = TinyJit(copy_states)
        jit()
        jit()
        return jit
      states = self._recurrent_states()
      self._save_state_jit = copy_jit(list(zip(self._state_checkpoints, states)))
      self._restore_state_jit = copy_jit(list(zip(states, self._state_checkpoints)))

  def _save_state_checkpoint(self, pos:int):
    self._init_state_checkpoints()
    assert self._save_state_jit is not None
    self._save_state_jit()
    self._state_checkpoint_pos = pos

  def _restore_state_checkpoint(self):
    assert self._restore_state_jit is not None
    self._restore_state_jit()

  def warmup(self, chunk_size:int=256):
    device = self.token_embd.weight.device
    direct_capture = not self.has_recurrent_block and all(isinstance(block, TransformerBlock) for block in self.blk)
    if direct_capture:
      device = str(self.token_embd.weight.device)
      direct_capture = device.startswith("AMD") and Device[device].renderer.target.arch.startswith("gfx11")

    # Recurrent prefill has one fixed padded shape with symbolic valid length.
    recurrent_chunk = min(chunk_size, 256)
    warm_len = min(recurrent_chunk if self.has_recurrent_block else chunk_size * 2, self.max_context - 1)
    if warm_len > 0:
      if direct_capture:
        x = Tensor.zeros(1, 1, self.blk[0].config.dim, device=device)
        for block in self.blk: block._init_state(x)
        Tensor.realize(*[state for block in self.blk for state in (getattr(block, "cache_kv"), getattr(block, "freqs_cis"))])
        self.flash_prefill_jit.cnt = 1
        next(self.generate([0] * warm_len, chunk_size=chunk_size))
      elif self.has_recurrent_block:
        # State creation must happen outside JIT capture: capturing _init_state makes the first real request
        # reuse initialization buffers instead of the persistent recurrent/KV state.
        x = Tensor.zeros(1, 1, self.blk[0].config.dim, device=device)
        for block in self.blk: block._init_state(x)
        states = [getattr(block, name) for block in self.blk for name in ("cache_kv", "freqs_cis", "conv_state", "recurrent_state")
                  if hasattr(block, name)]
        Tensor.realize(*states)
        self._init_state_checkpoints()
        self.prefill_jit.cnt = self.flash_prefill_jit.cnt = 1
        short_decode_len = min(8192, self.max_context)
        self.rollout_jits[short_decode_len] = TinyJit(
          functools.partial(self.forward_recurrent_decode, decode_len=short_decode_len, sample=False))
        self.rollout_jits[short_decode_len].cnt = 1
        self._warming_up = True
        warm = self.generate([0] * warm_len, chunk_size=chunk_size)
        prefill_batch = getenv("PREFILL_JIT_BATCH_SIZE", 128 if str(device).startswith("AMD") else JIT_BATCH_SIZE.value)
        with Context(JIT_BATCH_SIZE=prefill_batch): next(warm)
        next(warm)
        # Long-context decode uses a larger attention partition. Capture it before listening so crossing 8K is seamless.
        if self.max_context > short_decode_len:
          self.rollout_jits[self.max_context] = TinyJit(
            functools.partial(self.forward_recurrent_decode, decode_len=self.max_context, sample=False))
          self.rollout_jits[self.max_context].cnt = 1
          long_result = self(Tensor([[0]], dtype="int32", device=device),
                             UOp.variable("start_pos", 0, self.max_context-1).bind(short_decode_len), Tensor([0.0], device=device))
          assert isinstance(long_result, Tensor)
          long_result.realize()
        self._warming_up = False
      else:
        for salt in range(2): next(self.generate([salt] + [0] * (warm_len - 1), chunk_size=chunk_size))

    # Rollout uses fixed power-of-two KV shapes. Capture every shape up front so requests never pay a JIT transition.
    if not self.has_recurrent_block:
      min_bucket = max(1, getenv("DECODE_BUCKET", 256))
      bucket_positions:dict[int, int] = {}
      for pos in [0] + [1 << i for i in range(self.max_context.bit_length())]:
        bucket = min(self.max_context, max(min_bucket, 1 << pos.bit_length()))
        bucket_positions.setdefault(bucket, pos)
      v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
      token, temperature = Tensor([[0]], dtype="int32", device=device), Tensor([0.0], device=device)
      for bucket, pos in sorted(bucket_positions.items()):
        if direct_capture:
          self.rollout_jits[bucket] = TinyJit(functools.partial(self.forward, kv_len=bucket))
          self.rollout_jits[bucket].cnt = 1
        for _ in range(1 if direct_capture else 2):
          result = self(token, v_start_pos.bind(pos), temperature)
          assert isinstance(result, Tensor)
          result.realize()

    if self._state_checkpoints:
      # Recurrent warmup starts from the zero checkpoint. Restore it directly instead of scheduling state clears and
      # then copying the same zeros back into the checkpoint.
      self._restore_state_checkpoint()
      self._state_checkpoint_pos = 0
    elif resets := [r for block in self.blk for r in block._state_reset_ops()]: Tensor.realize(*resets)
    self._cached_tokens = []

  def generate(self, tokens:list[int], chunk_size:int=256, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = min(chunk_size, 256)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    device = self.token_embd.weight.device
    temp = Tensor([temperature], device=device)
    # Dense attention needs a symbolic slice into one input buffer. Recurrent prefill instead creates fixed-size
    # chunk tensors below; allocating its input at max_context makes short requests spend most of their time converting zeros.
    t = None if self.has_recurrent_block else \
      Tensor(tokens + [0] * (self.max_context + chunk_size - len(tokens)), dtype="int32", device=device).reshape(1, self.max_context + chunk_size)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens):
      if self.has_recurrent_block and self._state_checkpoints and start_pos == self._state_checkpoint_pos:
        self._restore_state_checkpoint()
      elif resets := [r for b in self.blk for r in b._state_reset_ops()]:
        Tensor.realize(*resets)
        if self._state_checkpoints: self._save_state_checkpoint(0)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      remaining = len(tokens) - start_pos
      recurrent_prefill = self.has_recurrent_block and start_pos < prompt_len
      can_flash = bool(getenv("AMD_FLASH_ATTENTION", 1)) and chunk_size % 64 == 0 and \
                  (recurrent_prefill if self.has_recurrent_block else start_pos > 0 and remaining >= chunk_size)
      if can_flash:
        device = str(self.token_embd.weight.device)
        can_flash = device.startswith("AMD") and Device[device].renderer.target.arch.startswith("gfx11")
      use_flash = can_flash and (self.has_recurrent_block or start_pos % 64 == 0)
      sp = v_start_pos.bind(start_pos)
      # Dense attention aligns cache reuse to a 64-token flash tile. Recurrent prefill always uses its fixed,
      # padded chunk shape, and key_limit excludes the padding from attention.
      actual_nt = min(chunk_size, remaining)
      nt = chunk_size if use_flash or self.has_recurrent_block and start_pos < prompt_len else 1 if self.has_recurrent_block else \
           v_toks.bind(min(64 - start_pos % 64, remaining) if can_flash else actual_nt)
      if self.has_recurrent_block and (start_pos < prompt_len or out is None):
        assert isinstance(nt, int)
        inp = Tensor(tokens[start_pos:start_pos+actual_nt] + [0] * (nt-actual_nt), dtype="int32", device=device).reshape(1, nt)
      elif start_pos < prompt_len or out is None:
        assert t is not None
        inp = t[:, sp:sp+nt]
      else: inp = out
      valid_len = v_toks.bind(actual_nt) if recurrent_prefill else None
      # Save once immediately before a short final chunk. This is the nearest globally aligned state that can be
      # reused without changing flash-attention's numerical tile layout.
      if not self._warming_up and recurrent_prefill and remaining < chunk_size and start_pos % chunk_size == 0:
        self._save_state_checkpoint(start_pos)
      if use_flash: result = self(inp, sp, temp, use_flash=True, valid_len=valid_len, sample=temperature > 0)
      elif valid_len is not None: result = self(inp, sp, temp, valid_len=valid_len, sample=temperature > 0)
      elif temperature > 0: result = self(inp, sp, temp, sample=True)
      else: result = self(inp, sp, temp)
      out = result.realize()
      start_pos += actual_nt if self.has_recurrent_block else nt if isinstance(nt, int) else nt.val
      # Generated tool calls are reconstructed by clients and are not guaranteed token-identical on the next request.
      # Keep the reusable checkpoint at the stable prompt boundary instead of overwriting it inside generated output.
      if not self._warming_up and self.has_recurrent_block and start_pos == prompt_len and start_pos % chunk_size == 0:
        self._save_state_checkpoint(start_pos)
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
