from __future__ import annotations
import functools, itertools, math, operator, struct
from typing import Callable, Sequence, cast
from tinygrad import Tensor, UOp
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, ExpertWeights, WMMA_ARG, _is_cdna, _is_rdna4, warp_reduce, _amd_load, partial_sum, _reg, _sys, \
  keyed_empty, _carry, _unbind, amd_custom_kernels_supported, PARTIALS
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod, unwrap, round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.renderer import Estimates

GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 6, 8, 10, 11, 12, 13, 14
IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS = 17, 18, 20, 21, 22, 23
GRID_QUANTS = (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S)
QUANT_SIZES = {Q5_0: 176, Q8_0: 272, Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XS: 74,
               IQ3_XXS: 98, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136}  # bytes per 256 weights
HALFWORD_QUANTS = (Q5_0, Q8_0, Q6_K, Q2_K, Q3_K, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S)
SMALL_BLOCK_QUANTS = {Q5_0: 11, Q8_0: 17, IQ4_NL: 9}
QUANT_NAMES = {Q5_0: "q5_0", Q8_0: "q8_0", Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6_k", IQ4_XS: "iq4_xs",
               IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ4_NL: "iq4_nl", IQ3_S: "iq3_s", IQ2_S: "iq2_s"}

def q8_0_linear(w:Tensor, devices:tuple[str, ...]|str, axis:int|None=None) -> Linear:
  """w (out, in) as a Linear of ggml Q8_0 blocks (an f16 scale, 32 int8), split over devices on axis (None: a copy on each), its weight lazy"""
  out_features, in_features = cast(tuple[int, int], w.shape)
  blocks = w.float().reshape(out_features, in_features // 32, 32)
  d = blocks.abs().max(-1, keepdim=True) / 127
  q = (blocks / (d == 0).where(1, d)).round().cast(dtypes.int8)
  raw = d.half().bitcast(dtypes.uint8).cat(q.bitcast(dtypes.uint8), dim=-1)  # (out, blocks, 34)
  lin = Linear(in_features, out_features, bias=False)
  lin.ggml_type, lin.shard_axis = Q8_0, axis
  words = raw.flatten().bitcast(dtypes.uint16)  # (flat, a device's contiguous: as gguf_shard packs them)
  if not isinstance(devices, tuple): lin.weight = words.to(devices)
  elif axis is None: lin.weight = words.shard(devices)
  else:
    n = len(devices)
    local = raw.reshape(*raw.shape[:axis], n, -1, *raw.shape[axis+1:]).permute(axis, *range(axis), *range(axis+1, raw.ndim+1))
    lin.weight = local.reshape(n, -1).bitcast(dtypes.uint16).shard(devices, 0).flatten()
  return lin

def find_packed(decoded:Tensor, in_features:int) -> tuple[int, Tensor]|None:
  """(ggml type, packed words) if `decoded` is exactly the dequantization of packed ggml weights, rows of in_features"""
  if in_features % Q8_GROUP_SIZE: return None
  packed_sizes = {typ: decoded.numel() * type_size // 256 for typ,type_size in QUANT_SIZES.items()
                  if in_features % GGML_BLOCK_SIZE == 0 or typ in SMALL_BLOCK_QUANTS}
  graph = decoded.uop.toposort()
  raw = next((u for u in graph if u.op in (Ops.SHRINK, Ops.BUFFER, Ops.UNSHARD) and u.dtype == dtypes.uint8 and
              prod(u.shape) in packed_sizes.values()), None)
  if raw is None: return None
  # Only unwrap storage/order-preserving views, then require the exact dequantization expression.
  # This rejects subsequent arithmetic and permutations, including RoPE's concatenated query weights.
  # a permute keeps the order if the axes not of size 1 on a device stay sorted (gguf_shard moves the devices)
  def unwrapped(u:UOp) -> UOp:
    while u.op in (Ops.RESHAPE, Ops.STAGE) or (u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype)) or \
      (u.op is Ops.PERMUTE and (kept:=[a for a in u.marg if u.src[0].shard_shape[a] != 1]) == sorted(kept)):
      u = u.src[0]
    return u
  # Several formats have the same byte count (Q3_K/IQ3_S and Q4_K/IQ4_NL). Match the expression, not just the size.
  for ggml_type, size in packed_sizes.items():
    if size != prod(raw.shape): continue
    expected = ggml_data_to_tensor(Tensor(raw).flatten(), int(decoded.numel()), ggml_type)
    if unwrapped(decoded.uop).key == unwrapped(expected.uop).key: break
  else: return None
  # Some blocks are only halfword-aligned; keep all formats as zero-copy views of the GGUF storage.
  word_dtype = dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32
  raw_offset = raw.contiguous_view_offset()
  if raw_offset is None or raw_offset % word_dtype.itemsize or raw.buf_uop.dtype != dtypes.uint8: return None
  return ggml_type, Tensor(raw).flatten().bitcast(word_dtype).contiguous()

def _amd_dp4a(a:UOp, b:UOp, c:UOp, cdna:bool) -> UOp:
  # CDNA has no sudot4 (dot8-insts)
  return UOp.custom_function("__builtin_amdgcn_sdot4({}, {}, {}, false)" if cdna else "__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)",
                             shape=a.shape, dtype=dtypes.int32, fmt=True).call(a, b, c, inline=True)

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp.custom_function("__builtin_amdgcn_perm({}, {}, {})", shape=a.shape, dtype=dtypes.uint32,
                             fmt=True).call(*[x.cast(dtypes.uint32) for x in (a, b, selectors)], inline=True)

def _load_byte(raw:UOp, base:UOp, offset:int|UOp) -> UOp:
  size = raw.dtype.itemsize
  return (raw[base + offset//size].cast(dtypes.uint32) >> ((offset%size)*8)) & 255
def _load_u32(raw:UOp, base:UOp, offset:int|UOp, stream:bool=False) -> UOp:
  # Halfword-aligned formats cannot load u32 directly.
  lo, hi = (raw[base+offset//2+i] for i in range(2))
  if stream: lo, hi = _amd_load(lo), _amd_load(hi)
  return lo.cast(dtypes.uint32) | (hi.cast(dtypes.uint32) << 16)

def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

def _iq4_bytes(packed:UOp, shift:int|UOp) -> UOp:
  # the non-linear iq4nl table as a byte lookup: 3 byte_perms beat any arithmetic/select-tree form (~60% decode)
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _amd_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _amd_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _amd_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def _q5_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  # scales/mins (6-bit each) live in block bytes 4-15: three words total, same for the whole super-block's lanes
  w1, w2, w3 = raw[base+1].load(), raw[base+2].load(), raw[base+3].load()
  sb = (subgroup & 3) * 8  # byte within word
  byte1, byte2, byte3 = (w1 >> sb) & 255, (w2 >> sb) & 255, (w3 >> sb) & 255
  scale = (subgroup < 4).where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4))
  minimum = (subgroup < 4).where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4))
  d, dmin = (raw[base] & 0xffff).cast(dtypes.uint16), (raw[base] >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

def _iq4_scale(raw:UOp, base:UOp, subgroup:UOp) -> UOp:
  header, low = raw[base].load(), raw[base+1].load()
  scale = ((low >> (4*subgroup)) & 15) | (((header >> (16+2*subgroup)) & 3) << 4)
  return _half(header) * (scale.cast(dtypes.int32)-32).float()

def iq4_half_lut(device:str|tuple[str, ...]|None) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  return Tensor.const(tuple(x for j in range(16) for i in range(16) for x in (kvalues_iq4nl[i], kvalues_iq4nl[j])),
                      dtypes.float16).to(device, force=True).bitcast(dtypes.uint32)

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int, in_features:int, act:Callable|None=None) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  if act is None: value = x.reshape(tokens, groups, 32)[token, group, lane].load().float()
  else:  # x: a gate and up gemv's partial sums (tokens, 2*in_features, splits), the act of its halves' sums
    x = x.reshape(tokens, 2*in_features, x.shape[-1])
    gate, up = (sum((x[token, h*in_features + group*32 + lane, s].load() for s in range(1, x.shape[-1])),
                    x[token, h*in_features + group*32 + lane, 0].load()) for h in range(2))
    value = act(gate, up).float()
  # Quantize each input once, then pack four neighboring lanes into one word.
  # Keep divisions intact: reciprocal multiplication can move half-precision inputs across rounding ties.
  warp_max = warp_reduce(value.abs(), maximum=True, full_wave=True)
  d = UOp.custom_function("({0}/127.0f)", shape=warp_max.shape, dtype=dtypes.float, fmt=True).call(warp_max).maximum(1e-8)
  rounded = UOp.custom_function("__builtin_nearbyintf({0}/{1})", shape=value.shape, dtype=dtypes.float, fmt=True).call(value, d)
  quant = rounded.clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= UOp.custom_function(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f|offset<<10})", shape=word.shape, dtype=dtypes.uint32,
                                fmt=True).call(word)
  stores = (q[token, group, (lane//4).valid((lane%4).eq(0))].store(word),
            scale[token, group.valid(lane.eq(0))].store(d),
            xsum[token, group, (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant.float())))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int, act:Callable|None=None) -> tuple[Tensor, Tensor, Tensor]:
  """the q8 of x (tokens, in_features), or of act(gate, up) of a gate and up gemv's partial sums x (tokens, 2*in_features, splits). its
  buffers' slots are x's and the kernel's: the linears that read the same x share one q8"""
  if act is None: x = x.reshape(-1, x.shape[-1])
  groups, slot = in_features//Q8_GROUP_SIZE, hash((x.uop.key, tokens, in_features, act)) % (1 << 62)
  q, scale, xsum = (Tensor(UOp.alloc(shape, dt, slot+i, device=x.device, bind_on_realize=True))
                    for i, (shape, dt) in enumerate((((tokens, groups, 8), dtypes.uint32), ((tokens, groups), dtypes.float32),
                                                     ((tokens, groups, 2), dtypes.float32))))
  fxn = functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features, act=act)
  q, scale, xsum = Tensor.custom_kernel(q, scale, xsum, x, fxn=fxn)[:3]
  return q, scale, xsum

def _decode_linear(out:UOp, out_features:int, group_count:int, sets:Sequence[tuple[Callable, tuple[UOp, ...], Callable|None]], name:str,
                   share=False) -> UOp:
  # sets: (group_dot, its lookup tables, live) of each weight set, a row's output the sum of theirs in their order
  chunks = out.shape[2]
  # A wave per four outputs/chunk; group neighboring rows to amortize workgroup scheduling.
  span = 2 if group_count <= 16 and out_features % 2 == 0 else 1
  outs = math.gcd(out_features//span, 4)
  rows = math.gcd(out_features//(span*outs), 4)
  tokens = share if type(share) is int else n if share and (n:=cast(int, out.shape[0])) <= 8 else 1
  rowblock, row = UOp.range(out.shape[0]//tokens, 0, AxisType.GLOBAL), UOp.range(out_features//(rows*span*outs), 1, AxisType.GLOBAL)
  chunk, lane, half = UOp.range(chunks, 2, AxisType.GLOBAL), UOp.range(32//span, 3, AxisType.LOCAL), UOp.range(span, 4, AxisType.LOCAL)
  wave = UOp.range(rows, 5, AxisType.LOCAL)
  token, outputs = rowblock*tokens, [((row*rows+wave)*outs+o)*span+half for o in range(outs)]
  group = lane+chunk*32
  pairs = list(itertools.product(range(tokens), outputs))
  def set_values(s:int, group_dot:Callable, tables:tuple[UOp, ...], live:Callable|None) -> list[UOp]:
    if tables:  # (group_dot's lookup tables in LDS)
      lts = [UOp.alloc((int(t.numel()),), t.dtype, addrspace=AddrSpace.LOCAL) for t in tables]
      fill = UOp.group(*[lt[(i:=lane+half*(32//span)+wave*32+k*rows*32).valid(i < n)].store(t[i.minimum(n-1)].load())
                         for lt, t in zip(lts, tables) for n in [int(t.numel())] for k in range(-(-n//(rows*32)))]).barrier()
      tables = tuple(lt.after(fill) for lt in lts)
    def dot(t:int, output:UOp, g:UOp) -> UOp:
      return (g < group_count).where(group_dot(token+t, output, g.minimum(group_count-1), *tables), UOp.const(0, dtypes.float32))
    if live is None: return [dot(t, output, group) for t, output in pairs]
    # live(token, group): the group's dot is read and added at all (a loop of one or no iterations: the loads in it)
    i = UOp.range(((group < group_count) & live(token, group.minimum(group_count-1))).where(1, 0), 6+s, AxisType.LOOP)
    acc = _reg((len(pairs),), 0)
    acc = acc.after(UOp.group(*[acc[k].store(acc.after(i)[k].load() + dot(t, output, group + i)) for k, (t, output) in enumerate(pairs)]).end(i))
    return [acc[k].load() for k in range(len(pairs))]
  values = [[warp_reduce(v, full_wave=span == 1) for v in set_values(s, *ws)] for s, ws in enumerate(sets)]
  stores = [out[token+t, output, chunk.valid(lane.eq(0))].store(functools.reduce(operator.add, vs).cast(out.dtype))
            for (t, output), *vs in zip(pairs, *values)]
  return UOp.group(*stores).end(rowblock, row, wave, chunk, lane, half).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def _xor_lane(v:UOp, m:int) -> UOp:
  return UOp.custom_function(f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | m<<10}))",
                             shape=v.shape, dtype=dtypes.float, fmt=True).call(v)

def _decode_linear_dense(out:UOp, out_features:int, group_count:int, group_dot, name:str, tables:tuple[UOp, ...]=()) -> UOp:
  def waste(lanes:int) -> int: return lanes*((group_count+lanes-1)//lanes) - group_count
  lanes = 32 if group_count <= 16 else min((l for l in (32, 16, 8) if out_features % (32//l) == 0), key=lambda l: (waste(l), -l))
  outs_per_wave, slots = 32 // lanes, (group_count+lanes-1)//lanes
  per = (slots + 15) // 16
  if group_count <= 48 and out_features >= 2048: per = min(max(per, 4), slots)
  chunks = (slots + per-1) // per
  rows = math.gcd(out_features // outs_per_wave, 4)
  while rows > 1 and rows*chunks > 16: rows //= 2
  multi = int(out.shape[0]) if 1 < out.shape[0] <= 8 else 1
  BPB = out_features//(rows*outs_per_wave)
  row = UOp.range(out.shape[0]//multi*BPB, 0, AxisType.GLOBAL)
  wave, chunk, lane = UOp.range(rows, 1, AxisType.LOCAL), UOp.range(chunks, 3, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  sub, which = lane % lanes, lane // lanes
  output = ((row*rows+wave)*outs_per_wave + which) % out_features
  tokens = [((row*rows+wave)*outs_per_wave + which) // out_features] if multi == 1 else [UOp.const(t, dtypes.weakint) for t in range(multi)]
  if tables:
    tid, nthreads = (wave*chunks + chunk)*32 + lane, rows*chunks*32
    lds = []
    for tab in tables:
      n = int(tab.numel())
      l = UOp.alloc((n,), tab.dtype, addrspace=AddrSpace.LOCAL)
      idx = [tid + k*nthreads for k in range((n+nthreads-1)//nthreads)]
      lds.append(l.after(UOp.group(*[l[i.valid(i < n) if n % nthreads else i].store(tab[i.minimum(n-1)].load()) for i in idx]).barrier()))
    tables = tuple(lds)
  def dot(token:UOp, group:UOp) -> UOp:
    return (group < group_count).where(group_dot(token, output, group.minimum(group_count-1), *tables), UOp.const(0, dtypes.float32))
  totals = [functools.reduce(lambda a,s: a + dot(t, sub+(chunk*per+s)*lanes), range(1, per), dot(t, sub+chunk*per*lanes)) for t in tokens]
  for offset in (16, 8, 4, 2, 1)[(32//lanes).bit_length()-1:]: totals = [total + _xor_lane(total, offset) for total in totals]
  if chunks > 1:
    lds_t = UOp.alloc((len(totals), rows, chunks, outs_per_wave), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds_t = lds_t.after(UOp.group(*(lds_t[t, wave, chunk, which.valid(sub.eq(0))].store(total) for t, total in enumerate(totals))).barrier())
    totals = [functools.reduce(lambda a,c: a + lds_t[t, wave, c, which].load(), range(1, chunks), lds_t[t, wave, 0, which].load())
              for t in range(len(totals))]
  first = sub.eq(0) & chunk.eq(0)
  stores = [out[token, output.valid(first)].store(total.cast(out.dtype)) for token, total in zip(tokens, totals)]
  return UOp.group(*stores).end(row, wave, chunk, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def _iq_grid(device:str|tuple[str, ...]|None, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common as ggml
  grid, words = {IQ2_XS: (ggml.iq2xs_grid, 2), IQ2_S: (ggml.iq2s_grid, 2),
                 IQ3_XXS: (ggml.iq3xxs_grid, 1), IQ3_S: (ggml.iq3s_grid, 1)}[ggml_type]
  return Tensor.const(tuple((v >> (32*i)) & 0xffffffff for v in grid for i in range(words)),
                      dtypes.uint32).to(device, force=True)

def _iq_even_signs(signs:UOp) -> UOp:
  popcount = UOp.custom_function("__builtin_popcount({})", shape=signs.shape, dtype=dtypes.uint32, fmt=True)
  return signs | ((popcount.call(signs.cast(dtypes.uint32), inline=True) & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  low = ((signs & 15) * 0x00204081) & 0x01010101
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  return (word ^ _amd_byte_perm(UOp.const(0, dtypes.uint32), UOp.const(0, dtypes.uint32), low | 0x0c0c0c0c)) + low

def _quant_word(raw:UOp, base:UOp, subgroup:UOp, i:int|UOp, ggml_type:int, grid:UOp|None) -> UOp:
  # Four packed weight bytes, shared by integer-dot decode and FP16 WMMA prefill.
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset, stream=ggml_type == Q6_K)
  if ggml_type == Q8_0: return _load_u32(raw, base, 2+i*4, stream=True)
  if ggml_type == Q5_0:  # (weight j: its low nibble byte j%16 of qs (the high half past 16), its fifth bit bit j of qh)
    high = (word(2) >> (4*i)) & 15
    return ((word(6+(i%4)*4) >> (4*(i//4))) & 0x0f0f0f0f) | (((high & 1) | ((high & 2) << 7) | ((high & 4) << 14) | ((high & 8) << 21)) << 4)
  if ggml_type in (Q4_K, Q5_K):
    offset = (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8
    weights = (_amd_load(raw[base+offset], 8)[i] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
    if ggml_type == Q5_K: weights |= ((_amd_load(raw[base+4], 8)[i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
    return weights
  if ggml_type == Q6_K:
    low = word((subgroup//4)*64 + (subgroup%2)*32 + i*4) >> ((subgroup%4//2)*4).cast(dtypes.uint32)
    high = word(128 + (subgroup//4)*32 + i*4) >> ((subgroup%4)*2).cast(dtypes.uint32)
    return (low & 0x0f0f0f0f) | ((high & 0x03030303) << 4)
  if ggml_type in (Q2_K, Q3_K):
    offset = (16 if ggml_type == Q2_K else 32) + (subgroup//4)*32 + i*4
    weights = (word(offset) >> ((subgroup%4)*2)) & 0x03030303
    if ggml_type == Q3_K: weights |= ((word(i*4) >> subgroup) & 0x01010101) << 2
    return weights
  if ggml_type in (IQ4_NL, IQ4_XS):
    packed = word(2+(i%4)*4) if ggml_type == IQ4_NL else _amd_load(raw[base+2+subgroup*4+i%4])
    return _iq4_bytes(packed, 4*(i//4))
  assert grid is not None
  grid_words = 2 if ggml_type in (IQ2_XS, IQ2_S) else 1
  if ggml_type in (IQ3_S, IQ3_XXS):
    index = byte(2+subgroup*8+i)
    if ggml_type == IQ3_S:
      index |= ((byte(66+subgroup) >> i) & 1) << 8
      signs = byte(74+subgroup*4+i//2)
    else: signs = _iq_even_signs((word(66+subgroup*4) >> (7*(i//2))) & 127)
  elif ggml_type == IQ2_XS:
    packed = raw[base+1+subgroup*4+i//2].cast(dtypes.uint32)
    index, signs = packed & 511, _iq_even_signs(packed >> 9)
  else:
    index = byte(2+subgroup*4+i//2) | (((byte(66+subgroup) >> (2*(i//2))) & 3) << 8)
    signs = byte(34+subgroup*4+i//2)
  return _iq_signed_word(grid[index*grid_words+i%grid_words], signs >> (4*(i%2)))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp, out_features:int, in_features:int, ggml_type:int,
                         cdna:bool=False) -> UOp:
  return _quant_decode_body(out, xq, xd, xs, ((raw, grids, ggml_type, None, None),), out_features, in_features, cdna)

def _quant_decode_body(out:UOp, xq:UOp, xd:UOp, xs:UOp, sets:tuple[tuple[UOp, tuple[UOp, ...], int, Callable|None, UOp|None], ...],
                       out_features:int, in_features:int, cdna:bool, xtoken:Callable|None=None, name:str="linear_", w:UOp|None=None,
                       tokens:int|None=None) -> UOp:
  # sets: (raw, grids, ggml_type, gate, live) of each weight set, a row's output the sum of theirs. gate(token, base) -> base: moves the weight
  # rows, xtoken(token) -> the activation row of an output row. w: output row t the sum over the rows k = j*rows + t (out's rows) of their
  # outputs weighted by w[k] (a row's groups go over its k)
  gpk = in_features//32
  def group_dot(raw:UOp, ggml_type:int, gate:Callable|None, token:UOp, output:UOp, group:UOp, *grid:UOp) -> UOp:
    if w is None: return dot(raw, ggml_type, gate, token, output, group, *grid)
    return dot(raw, ggml_type, gate, k:=group//gpk*out.shape[0] + token, output, group%gpk, *grid) * w[k].load()
  def dot(raw:UOp, ggml_type:int, gate:Callable|None, token:UOp, output:UOp, group:UOp, grid:UOp|None=None) -> UOp:
    block, subgroup = group//8, group%8
    base = (output*in_features//256 + block) * (QUANT_SIZES[ggml_type]//raw.dtype.itemsize)
    if ggml_type in SMALL_BLOCK_QUANTS: base = (output*in_features//32 + group)*SMALL_BLOCK_QUANTS[ggml_type]
    if gate is not None: base = gate(token, base)
    if xtoken is not None: token = xtoken(token)
    def byte(offset): return _load_byte(raw, base, offset)
    def word(offset): return _load_u32(raw, base, offset)
    xwords = _amd_load(xq[token, group, 0], 8)
    # One accumulator per scale group: 32 weights for Q4/Q5/IQ4_XS, two groups of 16 otherwise.
    dots = [UOp.const(0, dtypes.int32)] * (1 if ggml_type in (Q4_K, Q5_K, IQ4_XS) else 2)
    for i in range(8):
      weights = _quant_word(raw, base, subgroup, i, ggml_type, grid)
      acc = i//(8//len(dots))
      dots[acc] = _amd_dp4a(weights, xwords[i], dots[acc], cdna)
    if ggml_type in (Q4_K, Q5_K):
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      total = dots[0].float()*d*scale - (xs[token, group, 0].load()+xs[token, group, 1].load())*dmin*minimum
    elif ggml_type == IQ4_XS:
      total = dots[0].float() * _iq4_scale(raw, base, subgroup)
    elif ggml_type == Q6_K:
      # Subtract the quant offset via the activation sums instead of unpacking signed bytes.
      scales = [(raw[base+96+subgroup] >> (h*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for h in range(2)]
      total = sum((dots[h].float()-32*xs[token, group, h].load())*scales[h] for h in range(2))
      return total * xd[token, group] * _half(raw[base+104])
    elif ggml_type in (Q2_K, Q3_K):
      total = UOp.const(0, dtypes.float32)
      for half in range(2):
        j = subgroup*2+half
        if ggml_type == Q2_K:
          scale = byte(j)
          total += dots[half].float() * (scale & 15).float() * _half(raw[base+40]) - \
                   xs[token, group, half] * (scale >> 4).float() * _half(raw[base+41])
        else:
          scale = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
          total += (dots[half].float() - 4*xs[token, group, half]) * (scale.cast(dtypes.int32)-32).float() * _half(raw[base+54])
    elif ggml_type == Q5_0:  # (the offset 16 via the activation sums)
      total = ((dots[0]+dots[1]).float() - 16*(xs[token, group, 0].load()+xs[token, group, 1].load())) * _half(raw[base])
    elif ggml_type in (IQ2_XS, IQ2_S):
      scales = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
      total = sum(dots[h].float() * (((scales >> (h*4)) & 15).float()+0.5) for h in range(2)) * 0.25 * _half(raw[base])
    else:
      total = (dots[0]+dots[1]).float() * _half(raw[base])
      if ggml_type == IQ3_S: total *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      if ggml_type == IQ3_XXS: total *= ((word(66+subgroup*4) >> 28).float()+0.5)*0.5
    return total * xd[token, group]
  # (an expert gemv's tokens a slot share its weight loads)
  share = sets[0][3] is None if tokens is None else tokens
  groups = gpk * (1 if w is None else cast(int, w.shape[0])//cast(int, out.shape[0]))
  # live: a row's weights (w's layout), a group skipped where every token's is 0 (a table's padding, a token's left out experts)
  def is_live(live:UOp, token:UOp, group:UOp) -> UOp:
    base = group//gpk*out.shape[0] + token if w is not None else token
    return functools.reduce(lambda a, b: a | b, [live[base + t].load().ne(0) for t in range(cast(int, tokens))])
  name += "_".join(QUANT_NAMES[ggml_type] for _, _, ggml_type, _, _ in sets)
  if len(out.shape) == 2:
    (raw, grids, ggml_type, _, _), = sets
    return _decode_linear_dense(out, out_features, groups, functools.partial(group_dot, raw, ggml_type, None), name, grids)
  dots = [(functools.partial(group_dot, raw, ggml_type, gate), grids, None if live is None else functools.partial(is_live, live))
          for raw, grids, ggml_type, gate, live in sets]
  return _decode_linear(out, out_features, groups, dots, name, share)

@functools.cache
def _moe_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, *grids:UOp,
                       out_features:int, in_features:int, ggml_type:int, x_share:int, cdna:bool=False, ndev:int=1) -> UOp:
  # out rows are (token, slot): the slot's expert reads its weight rows, x_share slots share an activation row
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  local_experts = raw.shape[0]//expert_words
  def expert(row:UOp) -> UOp: return sel[row].cast(dtypes.weakint) - UOp.range(ndev, 4, AxisType.DEVICE)*local_experts
  def gate(row:UOp, base:UOp) -> UOp:
    base = base + expert(row).maximum(0).minimum(local_experts-1) * expert_words
    return base.valid((expert(row) >= 0) & (expert(row) < local_experts)) if ndev > 1 else base
  def xtoken(row:UOp) -> UOp: return row // x_share
  return _quant_decode_body(out, xq, xd, xs, ((raw, grids, ggml_type, gate, None),), out_features, in_features, cdna, xtoken, "moe_")

def moe_linear(layer:ExpertWeights, sel:Tensor, x:Tensor) -> Tensor:
  if not isinstance(T:=sel.shape[1], int): return moe_linear(layer, sel.pad_to(sel.max_shape, value=-1), x.pad_to(x.max_shape))[..., :T, :, :]
  B, T, slots = cast(tuple[int, int, int], sel.shape[-3:]) # sel: (B, T, k), weight: (E, O, I)
  in_features, x_slots = int(x.uop.shard_shape[-1]), int(x.shape[-2]) # gate/up: (B, T, 1, D), down: (B, T, k, H)
  assert x_slots in (1, slots), f"x has {x_slots} rows per token, expected 1 or {slots}"
  local_out, global_out, ggml_type = int(layer.weight.uop.max_shard_shape[1]), int(layer.weight.shape[1]), unwrap(layer.ggml_type)
  splits, ndev = layer.in_features // in_features, int(layer.weight.shape[0]) // int(layer.weight.uop.shard_shape[0])
  assert (layer.weight.uop.axis == 2) == (splits > 1), f"input features split over {splits} devices, weight split on axis {layer.weight.uop.axis}"
  xq, xd, xs = q8_quantize(x.contiguous(), B*T*x_slots, in_features)
  out = Tensor.empty(ndev*B*T*slots, global_out, splits*((in_features+1023)//1024), dtype=dtypes.float32, device=x.device, axis=layer.weight.uop.axis)
  fxn = functools.partial(_moe_decode_kernel, out_features=local_out, in_features=in_features, ggml_type=ggml_type, x_share=slots//x_slots,
                          cdna=_is_cdna(x.device), ndev=ndev)
  extra = (_iq_grid(x.device, ggml_type),) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  result = Tensor.custom_kernel(out, layer.packed, xq, xd, xs, sel.cast(dtypes.int32).flatten(), *extra, fxn=fxn)[0].sum(-1)
  return result.reshape(ndev, B, T, slots, global_out) if ndev > 1 else result.reshape(B, T, slots, global_out)

@functools.cache
def _expert_decode_kernel(out:UOp, xq:UOp, xd:UOp, xs:UOp, *srcs:UOp, out_features:int, in_features:int, sets:tuple[tuple[int, int, int], ...],
                          cdna:bool=False, weighted:bool=False, per:int=1, live:bool=False) -> UOp:
  # sets: (ggml_type, stride, offset) of each weight set, srcs its raw, slots, (weighted or live) the rows' weights, its grid. row k reads the
  # weight rows at slots[k // per]*stride + offset words (64 bit: a pool can pass 2**31 words) and the activation row k % rows (a slot's per
  # rows share their weight loads). weighted: the rows' weights, their weighted sums out's rows. live: the rows' weights (unweighted), a slot's
  # rows skipped where they're all 0
  def xtoken(row:UOp) -> UOp: return row % xq.shape[0]
  parts, at, w = [], 0, None
  for ggml_type, stride, offset in sets:
    count = 2 + (weighted or live) + (ggml_type in GRID_QUANTS)
    raw, slots, *rest = srcs[at:at+count]
    at += count
    def gate(row:UOp, base:UOp, slots=slots, stride=stride, offset=offset) -> UOp:
      return base + slots[row // per].cast(dtypes.weakint)*stride + offset
    w, grids = (rest[0], tuple(rest[1:])) if weighted or live else (None, tuple(rest))
    parts.append((raw, grids, ggml_type, gate, w))
  return _quant_decode_body(out, xq, xd, xs, tuple(parts), out_features, in_features, cdna, xtoken, "expert_sum_" if weighted else "expert_",
                            w if weighted else None, per)

def expert_linear(pool:Tensor, slots:Tensor, x:Tensor, out_features:int, in_features:int, ggml_type:int, stride:int, offset:int=0,
                  act:Callable|None=None, split:bool=False, w:Tensor|None=None, rows:int=1, per:int=1, live:Tensor|None=None) -> Tensor:
  """out[k] = W[slots[k // per]] @ x[k % rows]: the quantized experts in the slots of a pool of raw gguf blocks, a slot every stride bytes, W at
  offset bytes in it (per: a slot's rows, e.g. the tokens', their weights loaded once). x: (rows, in_features), out: (per*len(slots),
  out_features). a pool sharded over devices is a pool on each, of its own experts or parts of them: out is (ndev, per*len(slots), out_features),
  each device's, and x is the same on each or each device's (ndev, rows, in_features). act: x is a gate and up expert_linear's split output,
  the gemv's input act(gate, up) (in its q8, one kernel). split: out's partial sums over the input's 1024 chunks, a last axis. w: out (rows,
  out_features) the sums over the rows j*rows + t of their outputs weighted by w (w[j*rows + t]), the rows of weight 0 skipped. live: the rows'
  weights (w's layout) of an unweighted one, a slot skipped where all of its rows' are 0"""
  return copies_linear([(pool, slots, ggml_type, stride, offset, w if w is not None else live)], x, out_features, in_features, act, split,
                       w is not None, rows, per)

def copies_linear(sets:list[tuple[Tensor, Tensor, int, int, int, Tensor|None]], x:Tensor, out_features:int, in_features:int,
                  act:Callable|None=None, split:bool=False, weighted:bool=False, rows:int=1, per:int=1) -> Tensor:
  """expert_linear over the weight sets (pool, slots, ggml_type, stride, offset, live) of a table's copies in one kernel: out[k] the sum of the
  sets' where their live[k] isn't 0 (a row read from the copy that has it). weighted: one set, live its w"""
  sets = sorted(sets, key=lambda s: s[2] not in GRID_QUANTS)
  pool, slots = sets[0][:2]
  words = [dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32 for _, _, ggml_type, _, _, _ in sets]
  for (_, _, ggml_type, stride, offset, _), word in zip(sets, words):
    assert stride % word.itemsize == 0 and offset % word.itemsize == 0, f"{QUANT_NAMES[ggml_type]} slots must be {word.itemsize} byte aligned"
  ndev = int(pool.shape[0]) // int(pool.uop.shard_shape[0])
  xrows = int(x.uop.max_shard_shape[-3]) if act is not None else prod(x.uop.max_shard_shape) // in_features
  n, groups = slots.shape[0]*per if not weighted else rows, in_features//32 * (1 if not weighted else slots.shape[0]*per//rows)
  out = Tensor.empty(ndev*n, out_features, (groups+31)//32, dtype=dtypes.float32, device=pool.device, axis=0 if ndev > 1 else None)
  fxn = functools.partial(_expert_decode_kernel, out_features=out_features, in_features=in_features, cdna=_is_cdna(x.device), weighted=weighted,
                          sets=tuple((ggml_type, stride//word.itemsize, offset//word.itemsize) for (_, _, ggml_type, stride, offset, _), word
                                     in zip(sets, words)), per=per, live=not weighted and sets[0][5] is not None)
  srcs = [src for (pool, slots, ggml_type, _, _, live), word in zip(sets, words)
          for src in (pool.bitcast(word), slots.cast(dtypes.int32), *([] if live is None else [live]),
                      *([_iq_grid(x.device, ggml_type)] if ggml_type in GRID_QUANTS else []))]
  ret = Tensor.custom_kernel(out, *q8_quantize(x, xrows, in_features, act), *srcs, fxn=fxn)[0]
  ret = ret if split else ret[..., 0] if ret.shape[-1] == 1 else ret.sum(-1)  # (a sum of one is a copy)
  return ret.reshape(ndev, n, *ret.shape[1:]) if ndev > 1 else ret

# kgroups: groups of 16 lanes that split k in a 16x16x16 fragment, RDNA3 copies a and b in both halves
def _wmma_words(lane_group:UOp, kgroups:int) -> tuple[UOp|int, ...]:
  return tuple(range(4)) if kgroups == 1 else tuple(lane_group*(4//kgroups) + j for j in range(4//kgroups))

def _output_waves(out_features:int, output_tiles:int) -> int: return 2 if output_tiles <= 2 and out_features % (32*output_tiles) == 0 else 1

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int, lanes:int, heads:int=1, live:UOp|None=None):
  # heads: out's rows a head's in turn (a workgroup's one head's). live: a head's rows to compute (a symbolic count: the workgroups of those,
  # out the max's)
  if out_features % (16*output_tiles): output_tiles = 1
  output_waves = _output_waves(out_features, output_tiles)
  # the waves on more tokens (up to 8) share the workgroup's weights: a fragment dequantized once, through LDS
  token_waves = next(n for n in (8, 4, 2, 1) if out.shape[0] // heads % (token_tile*n) == 0 and (2*output_tiles) % n == 0)
  per, rows_head = token_tile*token_waves, out.shape[0] // heads
  blocks = rows_head // per if live is None else (live + per - 1) // per  # (a head's workgroups)
  token_block = UOp.range(heads*blocks, 0)
  output_block = UOp.range(out_features//(16*output_tiles*output_waves), 1)
  # lane is a hardware WARP range (like the flash kernel): the fragment math stays visible without being
  # range-split into nested loops, which would scramble the WMMA fragment layout
  lane, wave = UOp.range(lanes, -1, axis_type=AxisType.WARP), UOp.range(output_waves*token_waves, 3, axis_type=AxisType.LOCAL)
  col, lane_group, rows = lane % 16, lane // 16, 256 // lanes
  first = token_block // blocks * rows_head + token_block % blocks * per + wave // output_waves * token_tile
  outputs = tuple((output_block*output_waves + wave % output_waves)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(first + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(first + tile*16 + lane_group*rows + i for i in range(rows)) for tile in range(token_tile//16))
  return output_waves*token_waves, token_block, output_block, lane, wave, lane_group, outputs, inputs, tokens

def _wmma_stores(out, outputs, tokens, accs, update, lane, wave, waves, kgroups):
  # RDNA4 and CDNA (kgroups > 1) own consecutive rows per lane group, matching tokens directly.
  if kgroups > 1:
    return [out[token, output].store(acc.after(update)[i].load()) for output,output_accs in zip(outputs, accs)
            for tile_tokens,acc in zip(tokens, output_accs) for i,token in enumerate(tile_tokens)]
  # the accumulator fragment halves are exchanged between lane pairs (l, l^16) through LDS (a ds_swizzle without CUSTOM)
  flat_accs = [acc for output_accs in accs for acc in output_accs]
  lds = UOp.alloc((waves, 32, len(flat_accs)*8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  stores = [lds[wave, lane, a*8+i].store(acc.after(update)[i].load()) for a,acc in enumerate(flat_accs) for i in range(8)]
  lds = lds.after(*stores)
  def values(ai:int) -> tuple[UOp, ...]:
    own = tuple(lds[wave, lane, ai*8+i].load() for i in range(8))
    peer = tuple(lds[wave, lane ^ 16, ai*8+i].load() for i in range(8))
    low = (lane // 16).eq(0)
    return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
  tt = len(tokens)
  return [out[token, output].store(value) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
          for tile,(tile_tokens,_acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, values(ot*tt+tile))]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, kgroups, lanes, heads=1):
  # heads: the rows a head's in turn, each its own weights (a workgroup's rows one head's)
  x = x.reshape(out.shape[0], in_features)
  waves, token_block, output_block, lane, wave, lane_group, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  # IQ4_NL blocks are 32 wide: a device can hold part of a 256 weight row group
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  accs = tuple(tuple(UOp.alloc((len(tokens[0]),), dtypes.float32, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  # a step: steps groups of 32 inputs. the workgroup's waves on more tokens share its weight fragments (an output tile's half of a group, a
  # lane's ks): each dequantized by one wave into LDS (two buffers by the step's parity: one barrier a step)
  steps = next(n for n in (2, 1) if (in_features // Q8_GROUP_SIZE) % n == 0)
  step = UOp.range(in_features // Q8_GROUP_SIZE // steps, 4, AxisType.LOOP)
  ks = tuple(word*4 + b for word in _wmma_words(lane_group, kgroups) for b in range(4))
  output_waves = _output_waves(out_features, output_tiles)
  frag_count, head = 2*output_tiles*output_waves, input_tokens[0] // (out.shape[0] // heads)  # (a workgroup's rows one head's)
  def frag(g:int, f:UOp|int) -> tuple[UOp, ...]:  # (fragment f of group g of the step: f // 2's output tile of the workgroup, f % 2 its half)
    row = head*out_features + (output_block*output_waves + f // (2*output_tiles))*(16*output_tiles) + f // 2 % output_tiles*16 + lane % 16
    return dequant(row*output_words + (step*steps + g)//8*type_words, (step*steps + g) % 8, f % 2)
  if waves > output_waves:
    lds = UOp.alloc((2, steps, frag_count, lanes, len(ks)), dtypes.float16, addrspace=AddrSpace.LOCAL)
    staged = [lds[step % 2, g, f + wave, lane, i].store(v) for g, f in itertools.product(range(steps), range(0, frag_count, waves))
              for i, v in enumerate(frag(g, f + wave))]
    lds = lds.after(UOp.group(*staged).barrier())
  wmma_accs = [list(output_accs) for output_accs in accs]
  for g, half in itertools.product(range(steps), range(2)):
    afrags = tuple(UOp.stack(*(x[input_token, (step*steps + g)*32 + half*16 + i].cast(dtypes.float16) for i in ks))
                   for input_token in input_tokens)
    for output_tile in range(output_tiles):
      f = ((wave % output_waves)*output_tiles + output_tile)*2 + half
      bfrag = UOp.stack(*(lds[step % 2, g, f, lane, i].load() for i in range(len(ks)))) if waves > output_waves else UOp.stack(*frag(g, f))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(step) if g == half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, WMMA_ARG[0], lanes)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(step)
  stores = _wmma_stores(out, outputs, tokens, accs, update, lane, wave, waves, kgroups)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int, kgroups:int, lanes:int,
                               heads:int=1, live:UOp|None=None) -> UOp:
  token_tile, output_tiles = (32, 4) if out.shape[0] % 256 == 0 and out_features % 64 == 0 else \
    (64, 1 if out_features <= 1024 else 2) if out.shape[0] % 64 == 0 else (32 if out.shape[0] % 32 == 0 else 16, 2)
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes, heads, live)
  word_indices = _wmma_words(layout[5], kgroups)
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
    words = tuple((raw[qs_base+i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      (((raw[base+4+half*4+i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if ggml_type == Q5_K else 0) for i in word_indices)
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//4,
                            layout, dequant, f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma", kgroups, lanes, heads)

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int, kgroups:int, lanes:int,
                                heads:int=1, live:UOp|None=None) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    out_features <= 6144 else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes, heads, live)
  output_waves, _, _, lane, wave, lane_group, _, _, _ = layout
  word_indices = _wmma_words(lane_group, kgroups)
  local_lut = UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*lanes+lane, 256//(lanes*output_waves)
  lut = local_lut.after(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    scale = _iq4_scale(raw, base, subgroup)
    pairs = tuple(lut[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                  for word in word_indices for byte in range(4))
    return tuple((_half((pair >> (half*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in pairs)
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[IQ4_XS]//4, layout, dequant, "linear_iq4_xs_f16_wmma", kgroups, lanes,
                            heads)

def _wmma_scale(raw:UOp, base:UOp, subgroup:UOp, half:int, ggml_type:int) -> tuple[UOp, int, UOp]:
  def byte(offset): return _load_byte(raw, base, offset)
  zero, minimum = 0, UOp.const(0, dtypes.float)
  if ggml_type in (Q4_K, Q5_K):
    d, dmin, sc, mn = _q5_scales(raw, base, subgroup)
    scale, minimum = d*sc, dmin*mn
  elif ggml_type == Q2_K:
    sc = byte(subgroup*2+half)
    scale, minimum = _half(raw[base+40])*(sc & 15).float(), _half(raw[base+41])*(sc >> 4).float()
  elif ggml_type == Q3_K:
    j = subgroup*2+half
    sc = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
    scale, zero = _half(raw[base+54])*(sc.cast(dtypes.int32)-32).float(), 4
  elif ggml_type == Q6_K:
    scale, zero = _half(raw[base+104])*byte(192+subgroup*2+half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
  elif ggml_type == Q5_0: scale, zero = _half(raw[base]), 16
  else:
    scale = _half(raw[base])
    if ggml_type in (IQ2_XS, IQ2_S):
      sc = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
      scale *= (((sc >> (half*4)) & 15).float()+0.5)*0.25
    elif ggml_type == IQ3_S: scale *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
    elif ggml_type == IQ3_XXS: scale *= ((_load_u32(raw, base, 66+subgroup*4) >> 28).float()+0.5)*0.5
  return scale, zero, minimum

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp,
                                  out_features:int, in_features:int, ggml_type:int, kgroups:int, lanes:int, heads:int=1,
                                  live:UOp|None=None) -> UOp:
  token_tile, output_tiles = (32, 4) if out.shape[0] % 256 == 0 and out_features % 64 == 0 else \
    (64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16, 2)
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes, heads, live)
  output_waves, _, _, lane, wave, lane_group, _, _, _ = layout
  word_indices = _wmma_words(lane_group, kgroups)
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*lanes+lane, output_waves*lanes
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    if ggml_type in SMALL_BLOCK_QUANTS: base += subgroup*SMALL_BLOCK_QUANTS[ggml_type]  # eight independent blocks per 256 weights
    scale, zero, minimum = _wmma_scale(raw, base, subgroup, half, ggml_type)
    words = tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in word_indices)
    return tuple((((word >> (b*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()-zero)*scale-minimum).cast(dtypes.half)
                 for word in words for b in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//raw.dtype.itemsize,
                            layout, dequant, f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", kgroups, lanes, heads)

def _iq3_signs(device:str|tuple[str, ...]|None) -> Tensor:
  """the sign masks of the halves of an IQ3_XXS grid word: (its 7 sign bits, the word's half of the 8, a pair of its bytes) a uint32's two"""
  def mask(bits:int, half:int, pair:int) -> int:
    signs = bits | (bin(bits).count("1") & 1) << 7
    return sum(0x8000 << (16*b) for b in range(2) if signs >> (4*half + 2*pair + b) & 1)
  return Tensor.const(tuple(mask(i, h, p) for i in range(128) for h in range(2) for p in range(2)), dtypes.uint32).to(device, force=True)

def _iq3_grid_halves(device:str|tuple[str, ...]|None) -> Tensor:
  """the IQ3_XXS grid's words' byte pairs as halves: a word's two uint32s"""
  from tinygrad.runtime.autogen import ggml_common as ggml
  def halves(a:int, b:int) -> int: return struct.unpack("<I", struct.pack("<ee", a, b))[0]
  return Tensor.const(tuple(halves((v >> (16*p)) & 255, (v >> (16*p + 8)) & 255) for v in ggml.iq3xxs_grid for p in range(2)),
                      dtypes.uint32).to(device, force=True)

# a pair of a word's bytes b as halves 1024+b (a byte_perm under 0x64) less the zero, times the scale plus the bias
_F16_PAIR = UOp.custom_function("({{ typedef _Float16 h2 __attribute__((ext_vector_type(2))); __builtin_bit_cast(unsigned, "
  "(__builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, {0}, {4})) - __builtin_bit_cast(h2, {1})) * __builtin_bit_cast(h2, {2}) + "
  "__builtin_bit_cast(h2, {3})); }})", dtype=dtypes.uint32, fmt=True)
# a pair of halves times the scale, then their signs
_F16_SIGNED = UOp.custom_function("({{ typedef _Float16 h2 __attribute__((ext_vector_type(2))); __builtin_bit_cast(unsigned, "
  "__builtin_bit_cast(h2, (unsigned)({0})) * __builtin_bit_cast(h2, (unsigned)({1}))) ^ {2}; }})", dtype=dtypes.uint32, fmt=True)
TILE_QUANTS = (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_XXS)

@functools.cache
def _expert_f16_kernel(out:UOp, raw:UOp, x:UOp, tiles:UOp, *extra:UOp, out_features:int, in_features:int, ggml_type:int, stride:int,
                       offset:int, act:Callable|None, tile:int, gather:bool, skip:bool) -> UOp:
  # a workgroup: a tile's rows (16, 32 or 64) and 256 outputs (act: 128, a wave's gate tile and its up tile), 8 waves of 2 output tiles. a group
  # of 32 inputs at a time the tile's rows go through LDS (two buffers by its parity) and a wave decodes its outputs' weights to halves, the
  # scales in them (a tile of 16 rows: from LDS a block at a time, its lanes' reads of a row's words in turn, where they fit, its rows read from x
  # as they are). skip: a tile's 16 rows whose first is padding (live -1) left out
  rows_src, tables = extra[:gather + skip], extra[gather + skip:]
  xrow, live = rows_src[0] if gather else None, rows_src[-1] if skip else None
  waves, type_words = 8, QUANT_SIZES[ggml_type]//raw.dtype.itemsize
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  token_block, output_block = UOp.range(out.shape[0]//tile, 0), UOp.range(out_features//(32*waves), 1)
  lane, wave = UOp.range(32, -1, axis_type=AxisType.WARP), UOp.range(waves, 3, axis_type=AxisType.LOCAL)
  ph, col, tid = lane//16, lane%16, wave*32 + lane
  def output(o:int|UOp, w:UOp, c:UOp|int) -> UOp:
    return (output_block*waves + w)*16 + c + o*(out_features//2) if act is not None else (output_block*waves + w)*32 + o*16 + c
  outputs = tuple(output(o, wave, col) for o in range(2))
  if tables:
    (grid_src, sign_src), grid, signs = tables, UOp.alloc((512,), dtypes.uint32, addrspace=AddrSpace.LOCAL), \
      UOp.alloc((512,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    grid = grid.after(*(grid[tid*2 + i].store(grid_src[tid*2 + i]) for i in range(2)))
    signs = signs.after(*(signs[tid*2 + i].store(sign_src[tid*2 + i]) for i in range(2)))
  slot = tiles[token_block].cast(dtypes.int32)
  # a step: steps groups (their rows' words staged, their weights decoded together: more in flight between two barriers)
  steps = next(n for n in (2, 1) if (in_features // Q8_GROUP_SIZE) % n == 0)
  blk, sib = UOp.range((slot >= 0).where(in_features//GGML_BLOCK_SIZE, 0), 4, AxisType.LOOP), UOp.range(8//steps, 20, AxisType.LOOP)
  step = blk*(8//steps) + sib
  def row(o:int, c:UOp|int) -> UOp: return slot.maximum(0).cast(dtypes.int64)*stride + offset + output(o, wave, c)*output_words + blk*type_words
  if (small:=tile == 16 and waves*32*QUANT_SIZES[ggml_type] <= 32768):
    src = UOp.alloc((waves*32*type_words,), raw.dtype, addrspace=AddrSpace.LOCAL)
    src = src.after(UOp.group(*(src[((o*waves + wave)*16 + c)*type_words + w].store(raw[row(o, c) + w]) for o in range(2) for c in range(16)
                                for w in ((lane + i*32).minimum(type_words - 1) for i in range(-(-type_words//32))))).barrier())
    bases = [((o*waves + wave)*16 + col)*type_words for o in range(2)]
  else: src, bases = raw, [row(o, col) for o in range(2)]
  def xr(r:UOp) -> UOp: return r if xrow is None else xrow[r].cast(dtypes.int32).maximum(0).cast(dtypes.weakint)
  # a row's 16 words of the group, rows 20 words apart (a lane's 4 words of a fragment in one read, the 16 rows' on different banks)
  lds = UOp.alloc((2*steps*tile*20,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  staged = [lds[(((step % 2)*steps + k)*tile + r//16)*20 + r%16].store(_amd_load(x[xr(token_block*tile + r//16), ((step*steps + k)*16 + r%16)*2], 2)
                                                                     .bitcast(dtypes.uint32))
            for k in range(steps) for r in (tid + i*waves*32 for i in range(tile*16 // (waves*32)))]
  lds = src if small else lds.after(UOp.group(*staged).barrier())
  def halves(w:UOp) -> list[UOp]: return [(w >> (16*i)).cast(dtypes.uint16).bitcast(dtypes.half) for i in range(2)]
  def pair(v:UOp) -> UOp: return v.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32) * 0x10001
  # a lane's inputs of a 16 step h: the first 8 (ph 0) or the last 8, its weights' words 2ph and 2ph+1 of the step's 4
  def afrags(buf:UOp, k:int, t:int) -> list[UOp]:
    if small:
      r, i0 = xr(token_block*tile + t*16 + col), (step*steps + k)*32
      return [UOp.stack(*(x.after(buf)[r, i0 + h*16 + 8*ph + j] for j in range(8))) for h in range(2)]
    return [UOp.stack(*(v for w in range(4) for v in halves(buf[(((step % 2)*steps + k)*tile + t*16 + col)*20 + h*8 + 4*ph + w].load())))
            for h in range(2)]
  accs = [[UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(tile//16)] for _ in outputs]
  accs = [[acc.after(acc.store(acc.const_like(0))) for acc in row] for row in accs]
  bfrags:list[list[list[UOp]]] = []
  for k in range(steps):
    sub = sib*steps + k
    bfrags.append([])
    for o, base in enumerate(bases):
      bfrags[k].append([])
      for h in range(2):
        scale, zero, minimum = _wmma_scale(src, base, sub, h, ggml_type)
        scale = scale.cast(dtypes.half)
        zero2, bias2 = UOp.const((0x6400 + zero) * 0x10001, dtypes.uint32), pair(-minimum)
        def word(i:UOp, half:int) -> list[UOp]:  # (the lane's word i of the group: its four weights, half of its signs' group)
          if tables:
            q, bits = _load_byte(src, base, 2+sub*8+i)*2, ((_load_u32(src, base, 66+sub*4) >> (7*(i//2))) & 127)*4 + half*2
            return [v for p in range(2) for v in halves(_F16_SIGNED.call(grid[q + p].load(), pair(scale), signs[bits + p].load()))]
          q = _quant_word(src, base, sub, i, ggml_type, None)
          return [v for sel in (0x04010400, 0x04030402) for v in halves(_F16_PAIR.call(q, zero2, pair(scale), bias2, UOp.const(sel, dtypes.uint32)))]
        bfrags[k][o].append(UOp.stack(*(v for j in range(2) for v in word(h*4 + 2*ph + j, j))))
  updates = []
  for t in range(tile//16):
    on = sib if live is None else UOp.range((live[token_block*tile + t*16] >= 0).where(1, 0).after(lds), 5 + t, AxisType.LOOP)
    vals = [accs[o][t].after(on) for o in range(len(outputs))]
    for k in range(steps):
      for o in range(len(outputs)):
        for afrag, bfrag in zip(afrags(lds if live is None else lds.after(on), k, t), bfrags[k][o]):
          vals[o] = UOp.wmma(afrag, bfrag, vals[o], WMMA_ARG[0], 32)
    stored = UOp.group(*(accs[o][t].store(v) for o, v in enumerate(vals)))
    updates.append(stored if live is None else stored.end(on))
  update = UOp.group(*updates).end(sib, blk)
  def value(o:int, t:int, j:int) -> UOp: return accs[o][t].after(update)[j].load()
  rows = [[(token_block*tile + t*16 + ph*8 + j).valid(slot >= 0) for j in range(8)] for t in range(tile//16)]
  if act is not None: stores = [out[r, outputs[0]].store(act(value(0, t, j), value(1, t, j)).cast(out.dtype)) for t, rs in enumerate(rows)
                                for j, r in enumerate(rs)]
  else: stores = [out[r, output].store(value(o, t, j).cast(out.dtype)) for o, output in enumerate(outputs) for t, rs in enumerate(rows)
                  for j, r in enumerate(rs)]
  weights = cast(int, out.shape[0]) * out_features * in_features
  estimates = Estimates(2*weights, weights*QUANT_SIZES[ggml_type]//GGML_BLOCK_SIZE, weights*QUANT_SIZES[ggml_type]//GGML_BLOCK_SIZE)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(
    arg=KernelInfo(name=f"expert_{QUANT_NAMES[ggml_type]}_f16_wmma", opts_to_apply=(), estimates=estimates))

def expert_tiles(pool:Tensor, tiles:Tensor, x:Tensor, out_features:int, in_features:int, ggml_type:int, stride:int, offset:int=0,
                 xrow:Tensor|None=None, act:Callable|None=None, out:Tensor|None=None, tile:int=64, live:Tensor|None=None) -> Tensor:
  """out[r] = W[tiles[r // tile]] @ x[xrow[r]] (or x[r]) in half: an expert's tile of rows (16, 32 or 64) of a quant type of TILE_QUANTS on the
  f16 tensor cores (RDNA4), x in half (act: act(gate, up) of the outputs' two halves). a tile -1 is skipped (its rows not written), a row -1 of
  xrow reads x[0]. out: the
  buffer of the output. live: the rows' padding (-1), a tile's 16 rows whose first is padding skipped (their outputs 0)"""
  ndev, n = int(pool.shape[0]) // int(pool.uop.shard_shape[0]), tile * tiles.shape[0]
  if out is None:
    out = Tensor.empty(ndev*n, out_features // (1 if act is None else 2), dtype=dtypes.half, device=pool.device, axis=0 if ndev > 1 else None)
  word = dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32
  assert stride % word.itemsize == 0 and offset % word.itemsize == 0, f"{QUANT_NAMES[ggml_type]} slots must be {word.itemsize} byte aligned"
  fxn = functools.partial(_expert_f16_kernel, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                          stride=stride//word.itemsize, offset=offset//word.itemsize, act=act, tile=tile, gather=xrow is not None,
                          skip=live is not None)
  tables = (_iq3_grid_halves(x.device), _iq3_signs(x.device)) if ggml_type == IQ3_XXS else ()
  srcs = (pool.bitcast(word), x.reshape(-1, in_features).cast(dtypes.half).contiguous(), tiles.cast(dtypes.int32),
          *(t.cast(dtypes.int32) for t in (xrow, live) if t is not None), *tables)
  ret = Tensor.custom_kernel(out, *srcs, fxn=fxn)[0]
  return ret.reshape(ndev, n, -1) if ndev > 1 else ret

def _wmma_kernel(ggml_type:int, device) -> tuple[Callable[..., UOp], tuple[Tensor, ...]]:
  """the f16 tensor cores' kernel of a quant type on device and its tables"""
  cdna, fxn = _is_cdna(device), cast(Callable[..., UOp], _iq4_linear_f16_wmma_kernel)
  extra:tuple[Tensor, ...] = (_iq_grid(device, ggml_type),) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  if ggml_type == IQ4_XS: extra = (iq4_half_lut(device),)
  else: fxn = functools.partial(_q5_linear_f16_wmma_kernel if ggml_type in (Q4_K, Q5_K) else _quant_linear_f16_wmma_kernel, ggml_type=ggml_type)
  return functools.partial(fxn, kgroups=4 if cdna else 2 if _is_rdna4(device) else 1, lanes=64 if cdna else 32), extra

def head_linear(x:Tensor, w:Tensor) -> Tensor|None:
  """x (1, H, T, in) @ w.T of a head's own weights w (H, out, in) of packed ggml weights, on the f16 tensor cores (T a multiple of 64), a few
  tokens on the expert gemv (a head a slot of the weights), else None"""
  B, H, T, I = x.shape
  N = x.max_shape[2]  # (a symbolic T: the kernel's rows those there are a head, its buffers the max's)
  if B != 1 or not isinstance(I, int) or not amd_custom_kernels_supported(x.device): return None
  if (found:=find_packed(w, I)) is None: return None
  ggml_type, raw = found
  heads, out_features = (x.uop.shard_shape[1] if x.uop.axis == 1 else H), int(w.shape[1])
  if N % 64:
    if not isinstance(T, int) or T > 8: return None
    slots = Tensor.arange(heads, dtype=dtypes.int32).to(x.device)
    y = expert_linear(raw, slots, x.reshape(H*T, I), out_features, I, ggml_type, out_features*I*QUANT_SIZES[ggml_type]//256, per=T)
    return y.reshape(B, H, T, out_features)
  fxn, extra = _wmma_kernel(ggml_type, x.device)
  out = Tensor.empty(H*N, out_features, dtype=dtypes.float32, device=x.device, axis=None if x.uop.axis is None else 0)
  fxn = functools.partial(fxn, out_features=out_features, in_features=I, heads=heads, **({} if isinstance(T, int) else {"live": _unbind(T)}))
  x = _carry(x.cast(dtypes.float16).pad_to(x.max_shape).reshape(H*N, I).contiguous(), *([] if isinstance(T, int) else [T]))
  return Tensor.custom_kernel(out, raw, x, *extra, fxn=fxn)[0].reshape(B, H, N, out_features)[:, :, :T]

def q8_linear(layer:Linear, x:Tensor, xq:tuple[Tensor, ...]=(), rows:int|UOp|None=None) -> Tensor:
  """layer(x) on the q8 of x (xq: its q8, from the kernel that made x). rows: a symbolic count of x's rows (its max shape's) computed (the
  tensor cores' kernel; the rest garbage)"""
  assert layer.ggml_type in QUANT_SIZES
  tokens = int(x.numel()) // layer.in_features
  out_features, in_features = layer.out_features, int(x.uop.shard_shape[-1])
  splits = layer.in_features // in_features
  assert (layer.shard_axis == 1) == (splits > 1), f"input features split over {splits} devices, weight split on axis {layer.shard_axis}"
  out_shape:tuple[int, ...] = (splits*tokens, out_features)
  fxn:Callable[..., UOp]
  extra:tuple[Tensor, ...] = (_iq_grid(x.device, layer.ggml_type),) if layer.ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  # the kernels write the output features of their own device
  local_out = out_features // len(dev) if isinstance(dev:=layer.weight.device, tuple) and layer.shard_axis == 0 else out_features
  cdna = _is_cdna(x.device)
  if tokens % 16 == 0 and local_out % 16 == 0:
    fxn, extra = _wmma_kernel(layer.ggml_type, x.device)
    if rows is not None: fxn = functools.partial(fxn, live=_unbind(rows))
    srcs = (_carry(x.cast(dtypes.float16).contiguous(), *([] if rows is None else [rows])), *extra)
  else:
    fxn = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type, cdna=cdna)
    srcs = (*(xq or q8_quantize(x, tokens, in_features)), *extra)
  out = keyed_empty((layer.weight.uop.key, *(s.uop.key for s in srcs)), out_shape, dtypes.float32, x.device,
                    None if layer.shard_axis is None else 1-layer.shard_axis)
  result = Tensor.custom_kernel(out, layer.weight, *srcs, fxn=functools.partial(fxn, out_features=local_out, in_features=in_features))[0]
  # row parallel: every device wrote the partial sum of its input features, the sum over the devices is the allreduce (PARTIALS: its caller's)
  if splits > 1 and PARTIALS[0]: return result.reshape(splits, *x.shape[:-1], out_features)
  if splits > 1: result = partial_sum(result.reshape(splits, tokens, out_features))
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias


# *** routing of many tokens: a token's top k in a wave, then a counting sort of the selections into tiles of an expert's rows

@functools.cache
def _route_topk_kernel(sel:UOp, w:UOp, logits:UOp, bias:UOp, k:int, live:UOp|int) -> UOp:
  # a wave a token: the k experts of the largest sigmoid + bias, the largest first (ties: the lower expert), and their sigmoids over their sum
  # added in that order: the host's exact route(). the tokens from live on: expert -1, weight 0
  T, E = cast(tuple[int, int], logits.shape)
  token, lane = UOp.range(T, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  def sigmoid(e:UOp) -> UOp: return _sys("(1.0f / (1.0f + __builtin_expf(-{0})))", dtypes.float, logits[token, e].load())
  experts = [lane + 32*i for i in range(E // 32)]
  keys, picks, total = [sigmoid(e) + bias[e].load() for e in experts], [], UOp.const(0.0, dtypes.float)
  bad = functools.reduce(UOp.__or__, [_sys("__builtin_isnan({0})", dtypes.bool, key) for key in keys])
  valid = warp_reduce(bad.cast(dtypes.float), maximum=True, full_wave=True).eq(0)
  for _ in range(k):
    top = warp_reduce(functools.reduce(UOp.maximum, keys), maximum=True, full_wave=True)
    lowest = functools.reduce(UOp.minimum, [key.eq(top).where(e, E) for key, e in zip(keys, experts)]).cast(dtypes.float)
    e = (-warp_reduce(-lowest, maximum=True, full_wave=True)).cast(dtypes.int32).cast(dtypes.weakint)
    keys = [x.eq(e).where(-math.inf, key) for key, x in zip(keys, experts)]
    valid = valid & (e < E)
    picks.append((e, sigmoid(e.minimum(E - 1))))
    total = total + picks[-1][1]
  ok, stores = (token < live) & valid & (total > 0), []
  for r, (e, sg) in enumerate(picks):
    at = UOp.const(r, dtypes.weakint).valid(lane.eq(0))
    stores += [sel[token, at].store(ok.where(e.cast(dtypes.int32), -1)), w[token, at].store(ok.where(sg / total, 0.0))]
  return UOp.group(*stores).end(token, lane).sink(arg=KernelInfo(name="route_topk", opts_to_apply=()))

ROUTE_G, ROUTE_T = 64, 256  # the counting sort's workgroups (up to) and threads
def _route_grid(N:int) -> tuple[int, int]:  # (the workgroups of N selections, a thread's: none without a selection)
  G = min(ROUTE_G, (N + ROUTE_T - 1) // ROUTE_T)
  return G, (N + G*ROUTE_T - 1) // (G*ROUTE_T)

@functools.cache
def _route_init_kernel(counts:UOp, entries:UOp, tiles:UOp) -> UOp:  # (the counts 0, the rows and the tiles -1)
  i = UOp.range(max(int(prod(b.shape)) for b in (counts, entries, tiles)), 0, AxisType.GLOBAL)
  return UOp.group(*(b[i.valid(i < int(prod(b.shape)))].store(UOp.const(v, dtypes.int32)) for b, v in ((counts, 0), (entries, -1), (tiles, -1)))) \
    .end(i).sink(arg=KernelInfo(name="route_init", opts_to_apply=()))

def _wave_scan(x:UOp, lane:UOp) -> UOp:  # (the inclusive prefix sums of x over the lanes of a wave)
  for d in (1, 2, 4, 8, 16):
    other = _sys("__builtin_amdgcn_ds_bpermute({1}, {0})", dtypes.int32, x, ((lane - d).maximum(0)*4).cast(dtypes.int32))
    x = x + (lane >= d).where(other, UOp.const(0, dtypes.int32))
  return x

@functools.cache
def _route_count_kernel(rank:UOp, counts:UOp, sel:UOp, experts:int) -> UOp:
  # the rank of every selection among its expert's: a workgroup's in LDS, its base in the expert's global count
  N, per = int(prod(rank.shape)), (experts + ROUTE_T - 1) // ROUTE_T
  G, Q = _route_grid(N)
  wg, t = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(ROUTE_T, 1, AxisType.LOCAL)
  hist = UOp.alloc((per*ROUTE_T,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  hist = hist.after(UOp.group(*(hist[t*per + m].store(UOp.const(0, dtypes.int32)) for m in range(per))).barrier())
  items = [(i, (e:=sel.reshape(-1)[i.minimum(N-1)].load()).maximum(0).cast(dtypes.weakint), (i < N) & (e >= 0))
           for i in ((wg*Q + q)*ROUTE_T + t for q in range(Q))]
  locs = [_sys("__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)", dtypes.int32, hist[e], ok.cast(dtypes.int32))
          for i, e, ok in items]
  hist = hist.after(UOp.group(*locs).barrier())
  base = UOp.alloc((per*ROUTE_T,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  base = base.after(UOp.group(*(base[t*per + m].store(_sys("__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)",
    dtypes.int32, counts[(t*per + m).minimum(experts-1)], (t*per + m < experts).where(hist[t*per + m].load(), 0))) for m in range(per))).barrier())
  stores = [rank.reshape(-1)[i.valid(i < N)].store(base[e].load() + loc) for (i, e, _), loc in zip(items, locs)]
  return UOp.group(*stores).end(wg, t).sink(arg=KernelInfo(name="route_count", opts_to_apply=()))

@functools.cache
def _route_place_kernel(pos:UOp, entries:UOp, tiles:UOp, rank:UOp, counts:UOp, sel:UOp, experts:int, tile:int) -> UOp:
  # the experts' first rows (every workgroup scans the counts padded to tiles), every selection its row, the first workgroup the tiles' experts
  N, per = int(prod(pos.shape)), (experts + ROUTE_T - 1) // ROUTE_T
  G, Q = _route_grid(N)
  wg, wave, lane = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(ROUTE_T // 32, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  t, zero = wave*32 + lane, UOp.const(0, dtypes.int32)
  pads = [(t*per + m < experts).where((counts[(t*per + m).minimum(experts-1)].load() + (tile-1)) // tile * tile, zero) for m in range(per)]
  total = functools.reduce(UOp.__add__, pads, zero)
  incl = _wave_scan(total, lane)
  sums = UOp.alloc((ROUTE_T // 32,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  sums = sums.after(sums[wave.valid(lane.eq(31))].store(incl).barrier())
  before = functools.reduce(UOp.__add__, [(v < wave).where(sums[v].load(), zero) for v in range(ROUTE_T // 32)], zero) + incl - total
  firsts = [before + functools.reduce(UOp.__add__, pads[:m], zero) for m in range(per)]
  first = UOp.alloc((per*ROUTE_T,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  first = first.after(UOp.group(*(first[t*per + m].store(f) for m, f in enumerate(firsts))).barrier())
  stores = []
  for i in ((wg*Q + q)*ROUTE_T + t for q in range(Q)):
    e = sel.reshape(-1)[i.minimum(N-1)].load()
    p = first[e.maximum(0).cast(dtypes.weakint)].load() + rank.reshape(-1)[i.minimum(N-1)].load()
    stores += [pos.reshape(-1)[i.valid(i < N)].store((e >= 0).where(p, -1)),
               entries[p.cast(dtypes.weakint).valid((i < N) & (e >= 0))].store(i.cast(dtypes.int32))]
  for m, f in enumerate(firsts):
    nt = UOp.range(((t*per + m < experts) & wg.eq(0)).where(pads[m] // tile, 0), 2 + m, AxisType.LOOP)
    stores.append(tiles[(f // tile + nt).cast(dtypes.weakint)].store((t*per + m).cast(dtypes.int32)).end(nt))
  return UOp.group(*stores).end(wg, wave, lane).sink(arg=KernelInfo(name="route_place", opts_to_apply=()))

def _moe_route_tensors(logits:Tensor, bias:Tensor, k:int, tile:int, live:UOp|None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
  # moe_route in tensor ops: a selection's row its expert's tiles' start and the selections of the expert before it (in t*k + j order)
  T, E = cast(tuple[int, int], logits.shape)
  N, R = T*k, round_up(T*k + (tile-1)*E, tile)
  sig = logits.float().sigmoid()
  keys = sig + bias.float()
  sel = keys.topk(k)[1]
  weights = sig.gather(-1, sel)
  total = weights.sum(-1, keepdim=True)
  w = weights / total
  valid = (total > 0) & ~keys.isnan().any(-1, keepdim=True)
  ok = ((Tensor.arange(T).reshape(T, 1) < (T if live is None else Tensor(live))) & valid).expand(T, k).reshape(N)
  onehot = ok.unsqueeze(-1) & (sel.reshape(N, 1) == Tensor.arange(E).reshape(1, E))
  counts = onehot.sum(0, dtype=dtypes.int32)
  padded = (counts + tile - 1) // tile * tile
  first = padded.cumsum() - padded
  pos = ok.where((onehot.where(first.reshape(1, E) + onehot.cast(dtypes.int32).cumsum(0) - 1, 0)).sum(-1), -1).cast(dtypes.int32)
  entries = Tensor.full((R + 1,), -1, dtype=dtypes.int32, device=logits.device).contiguous()
  entries = entries.scatter(0, ok.where(pos, R), Tensor.arange(N, dtype=dtypes.int32).to(logits.device))[:R]
  start = Tensor.arange(R // tile).reshape(-1, 1) * tile
  tiles = ((first.reshape(1, E) <= start) & (start < (first + padded).reshape(1, E))).where(Tensor.arange(E).reshape(1, E) + 1, 0).sum(-1) - 1
  return ok.reshape(T, k).where(w, 0.0), pos, entries, tiles.cast(dtypes.int32), counts

def moe_counts(logits:Tensor, bias:Tensor, k:int, live:UOp) -> Tensor:
  T, E = cast(tuple[int, int], logits.shape)
  if amd_custom_kernels_supported(logits.device) and E % 32 == 0:
    sel, _ = _moe_topk(logits, bias, k, live)
  else:
    sig = logits.float().sigmoid()
    keys = sig + bias.float()
    sel = keys.topk(k)[1]
    valid = (Tensor.arange(T) < Tensor(live)) & (sig.gather(-1, sel).sum(-1) > 0) & ~keys.isnan().any(-1)
    sel = valid.unsqueeze(-1).where(sel, -1)
  return (sel.reshape(-1, 1) == Tensor.arange(E)).sum(0, dtype=dtypes.int32)

def _moe_topk(logits:Tensor, bias:Tensor, k:int, live:UOp|None) -> tuple[Tensor, Tensor]:
  T = logits.shape[0]
  sel, w = Tensor.empty(T, k, dtype=dtypes.int32, device=logits.device), Tensor.empty(T, k, dtype=dtypes.float32, device=logits.device)
  sel, w = Tensor.custom_kernel(sel, w, _carry(logits.float().contiguous(), *([] if live is None else [live])), bias.float().contiguous(),
                                fxn=functools.partial(_route_topk_kernel, k=k, live=T if live is None else _unbind(live)))[:2]
  return sel, w

def moe_route(logits:Tensor, bias:Tensor, k:int, tile:int=64, live:UOp|None=None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
  """the tokens' (T, E) router logits routed exactly (route_topk) and sorted by expert into tiles of an expert's rows: their weights (T, k), pos
  (T*k): the row of token t's j-th expert (t*k + j, -1 past live), entries (R): a row's selection t*k + j (-1: padding), tiles (R/tile): a tile's
  expert (-1), counts (E): an expert's rows. live: the tokens routed (the rest weight 0)"""
  T, E = cast(tuple[int, int], logits.shape)
  N, R = T*k, round_up(T*k + (tile-1)*E, tile)
  if not amd_custom_kernels_supported(logits.device) or E % 32: return _moe_route_tensors(logits, bias, k, tile, live)
  sel, w = _moe_topk(logits, bias, k, live)
  counts, entries, tiles = Tensor.custom_kernel(*(Tensor.empty(n, dtype=dtypes.int32, device=logits.device) for n in (E, R, R // tile)),
                                                fxn=_route_init_kernel)
  rank = Tensor.empty(N, dtype=dtypes.int32, device=logits.device)
  rank, counts = Tensor.custom_kernel(rank, counts, sel, fxn=functools.partial(_route_count_kernel, experts=E))[:2]
  pos = Tensor.empty(N, dtype=dtypes.int32, device=logits.device)
  pos, entries, tiles = Tensor.custom_kernel(pos, entries, tiles, rank, counts, sel,
                                             fxn=functools.partial(_route_place_kernel, experts=E, tile=tile))[:3]
  return w, pos, entries, tiles, counts
