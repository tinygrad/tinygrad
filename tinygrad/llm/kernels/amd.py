from __future__ import annotations
import functools, math, weakref
from typing import Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve, sint
from tinygrad.renderer.cstyle import HIPRenderer

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)
GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 10, 11, 12, 13, 14
IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS = 17, 18, 20, 21, 22, 23
QUANT_SIZES = {Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XS: 74,
               IQ3_XXS: 98, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136}  # bytes per 256 weights
HALFWORD_QUANTS = (Q6_K, Q2_K, Q3_K, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S)
QUANT_NAMES = {Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6", IQ4_XS: "iq4_xs",
               IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ4_NL: "iq4_nl", IQ3_S: "iq3_s", IQ2_S: "iq2_s"}

def _unbind(v:int|UOp) -> int|UOp: return v.unbind_all()[0] if isinstance(v, UOp) else v

@functools.cache
def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if getenv("DISABLE_AMD_KERNELS"): return False
  # Wave32 and WMMA kernels support RDNA3/4. CDNA's wave64/MFMA is not supported.
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  # @function contexts set ALLOW_DEVICE_USAGE=0 (scheduling must not open devices); the device is always open here
  with Context(ALLOW_DEVICE_USAGE=1):
    return (t:=getattr(Device[device], "target", None)) is not None and t[0] in (11, 12) and isinstance(Device[device].renderer, HIPRenderer)

@functools.cache
def _wmma_rdna4(device:str|tuple[str, ...]) -> bool:
  if isinstance(device, tuple): device = device[0]
  with Context(ALLOW_DEVICE_USAGE=1): return getattr(Device[device], "target")[0] == 12

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=
      (f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

def _reg(shape:tuple[int, ...], value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.alloc(shape, dtypes.float, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

# ******** quant linear: q8-activation kernels over packed ggml weights ********

class Linear(nn.Linear):
  ggml_type:int|None = None
  use_custom_quant = True
  shard_axis:int|None = None
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
  def set_quantized(self, decoded:Tensor):
    if self.in_features % GGML_BLOCK_SIZE: return
    packed_sizes = {typ: decoded.numel() // 256 * type_size for typ,type_size in QUANT_SIZES.items()}
    graph = decoded.uop.toposort()
    raw = next((u for u in graph if u.op in (Ops.SHRINK, Ops.BUFFER, Ops.UNSHARD) and u.dtype == dtypes.uint8 and
                prod(u.shape) in packed_sizes.values()), None)
    if raw is None: return
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
      expected = ggml_data_to_tensor(Tensor(raw).flatten(), self.in_features * self.out_features, ggml_type)
      if unwrapped(decoded.uop).key == unwrapped(expected.uop).key: break
    else: return
    # Some blocks are only halfword-aligned; keep all formats as zero-copy views of the GGUF storage.
    word_dtype = dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32
    raw_offset = raw.contiguous_view_offset()
    if raw_offset is None or raw_offset % word_dtype.itemsize or raw.buf_uop.dtype != dtypes.uint8: return
    self.ggml_type = ggml_type
    self.shard_axis = decoded.uop.axis
    self.weight = Tensor(raw).flatten().bitcast(word_dtype).contiguous()
  def __call__(self, x:Tensor) -> Tensor:
    supported = self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized(self.weight)
      if self.ggml_type is None:
        # tiny dense fp16 matmul (e.g. the ssm beta/alpha head rows): single fp16 gemv kernel instead of a
        # generic matmul schedule, and realize the densely packed weight once if it is still a lazy ggml view
        if self.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and self.out_features <= 2048 \
          and self.in_features % (WARP_SIZE*4) == 0 and self.weight.uop.axis in (None, 0):
          numel, max_shape = x.numel(), x.max_shape
          if isinstance(numel, int) or prod(max_shape) // self.in_features <= 32:
            out = f16_gemv(self, x if isinstance(numel, int) else x.pad_to(max_shape))
            return out if isinstance(numel, int) else out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
        self.use_custom_quant = supported = False  # not a supported quant format
    if self.ggml_type in QUANT_SIZES and supported:
      if isinstance(x.numel(), int): return q8_linear(self, x)
      # symbolic token count: pad to the max chunk size so the kernels see static shapes, garbage rows are sliced off
      out = q8_linear(self, x.pad_to(x.max_shape))
      return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
    return super().__call__(x)

def _amd_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=(a, b, c), arg=("__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)", dtypes.int32))

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=tuple(x.cast(dtypes.uint32) for x in (a, b, selectors)), arg=("__builtin_amdgcn_perm({}, {}, {})", dtypes.uint32))

def _amd_load(ptr:UOp, lanes:int|None=None, stream:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  # nontemporal scalar load: streamed weights must not evict the activations/KV cache from L2
  if lanes is None: return ptr.load(arg="nontemporal")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(arg="nontemporal" if stream else None)

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
  w1, w2, w3 = _amd_load(raw[base+1]), _amd_load(raw[base+2]), _amd_load(raw[base+3])
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

def _q8_pack(value:UOp, lane:UOp, q:UOp, scale:UOp, xsum:UOp, token:UOp, group:UOp) -> tuple[UOp, ...]:
  # Quantize each input once, then pack four neighboring lanes into one word.
  # Keep divisions intact: reciprocal multiplication can move half-precision inputs across rounding ties.
  d = UOp(Ops.CUSTOM, src=(warp_reduce(value.abs(), maximum=True, full_wave=True),), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  rounded = UOp(Ops.CUSTOM, src=(value, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float))
  quant = rounded.clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= UOp(Ops.CUSTOM, src=(word,), arg=(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | offset<<10})", dtypes.uint32))
  return (q[token, group, (lane//4).valid((lane%4).eq(0))].store(word),
          scale[token, group.valid(lane.eq(0))].store(d),
          xsum[token, group, (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant.float())))

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  value = x.reshape(tokens, groups, 32)[token, group, lane].load().float()
  stores = _q8_pack(value, lane, q, scale, xsum, token, group)
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

# linears sharing an input (q/k/v, gate/up) reuse its quantization
_q8_cache:weakref.WeakKeyDictionary[UOp, tuple[Tensor, Tensor, Tensor]] = weakref.WeakKeyDictionary()

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  if (cached:=_q8_cache.get(x.uop)) is not None and cached[1].uop.max_shard_shape == (tokens, in_features//Q8_GROUP_SIZE): return cached
  _q8_cache[x.uop] = ret = _q8_quantize(x, tokens, in_features)
  return ret

def _q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  groups = in_features//Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  xsum = Tensor.empty(tokens, groups, 2, dtype=dtypes.float32, device=x.device)
  q, scale, xsum = Tensor.custom_kernel(q, scale, xsum, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:3]
  return q, scale, xsum

# ******** fused residual add + rmsnorm + q8 quantize ********

def _norm_waves(D:int) -> int: return next((w for w in (32, 16, 8, 4, 2, 1) if D % (w*32) == 0), 0)

# LL words are 32 bits (a peer write of 32 aligned bits is atomic, 64 bits may arrive in halves). a value is one word, the float
# rounded to 15 mantissa bits (relative error < 2^-16) with the low byte holding the epoch. a wide value is exact in two words,
# each holding 16 bits of the float above 16 bits of the epoch
def _ll_load(p:str) -> str: return f"__hip_atomic_load((unsigned*)({p}), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)"
def _ll_store(p:str, w:str) -> str: return f"__hip_atomic_store((unsigned*)({p}), {w}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);"

def _ll_write(ok:UOp, ptr:UOp, epoch:UOp, v:UOp, deps:tuple[UOp, ...], wide:bool=False) -> UOp:
  bits = "__builtin_bit_cast(unsigned, (float){3})"
  code = f"if ({{0}}) {{{{ {_ll_store('(unsigned*){1}', f'({bits} & 0xffff0000u) | ((unsigned){{2}} & 0xffffu)')} " \
         f"{_ll_store('(unsigned*){1}+1', f'({bits} << 16) | ((unsigned){{2}} & 0xffffu)')} }}}}" if wide else \
         f"if ({{0}}) {_ll_store('{1}', f'(({bits} + 0x80u) & ~0xffu) | ((unsigned){{2}} & 0xffu)')}"
  return UOp(Ops.CUSTOM, src=(ok, ptr, epoch, v, *deps), arg=(code, dtypes.void))

def _ll_round(v:UOp, wide:bool=False) -> UOp:
  # what a peer reads of v: every device sums the same rounded values, so they all get the same bits
  if wide: return v
  return UOp(Ops.CUSTOM, src=(v,), arg=("__builtin_bit_cast(float, (__builtin_bit_cast(unsigned, (float){0}) + 0x80u) & ~0xffu)", dtypes.float32))

def _ll_read(ok:UOp, ptrs:list[UOp], epoch:UOp, deps:tuple[UOp, ...], wide:bool=False) -> list[UOp]:
  # spin until every word shows the epoch, polling all of them at once (each poll is a trip to VRAM), then read the values
  words = [(i, k) for i in range(len(ptrs)) for k in (range(2) if wide else range(1))]
  loads = " ".join(f"_w[{j}] = {_ll_load('(unsigned*){'+str(i+2)+'}+'+str(k))};" for j,(i,k) in enumerate(words))
  mask = "0xffffu" if wide else "0xffu"
  ready = " && ".join(f"(_w[{j}] & {mask}) == ((unsigned){{1}} & {mask})" for j in range(len(words)))
  def value(w0:str, w1:str) -> str: return f"__builtin_bit_cast(float, ({w0} & 0xffff0000u) | ({w1} >> 16))" if wide else \
                                            f"__builtin_bit_cast(float, {w0} & ~0xffu)"
  wait = UOp(Ops.CUSTOM, src=(ok, epoch, *ptrs, *deps), arg=(
    f"({{{{ unsigned _w[{len(words)}]; if ({{0}}) for (;;) {{{{ {loads} if ({ready}) break; __builtin_amdgcn_s_sleep(1); }}}} "
    f"else {{{{ {' '.join(f'_w[{j}] = 0;' for j in range(len(words)))} }}}} {value('_w[0]', '_w[1]' if wide else '_w[0]')}; }}}})", dtypes.float32))
  return [wait] + [UOp(Ops.CUSTOM, src=(ok, p, wait), arg=(
    f"({{0}} ? {value(_ll_load('(unsigned*){1}'), _ll_load('(unsigned*){1}+1') if wide else '')} : 0.0f)", dtypes.float32)) for p in ptrs[1:]]

def _p2p_slot(epoch:UOp, s:int|UOp, wg:UOp, e:UOp, N:int, G:int, chunk:int, wide:bool=False) -> UOp:
  return (((epoch & 1)*N + s)*G + wg)*chunk + e*(2 if wide else 1)

def _p2p_exchange(vals:list[UOp], elems:list[UOp], wg:UOp, counter:UOp, mbox:UOp, peers:tuple[UOp, ...], N:int, G:int, chunk:int,
                  valid:UOp|None=None, wide:bool=False) -> tuple[list[UOp], UOp]:
  # LL protocol: every value goes to the peers as one word with the epoch in it (see _ll_write), so it carries its own ready flag.
  # a reader spins on its words until they show the current epoch: no fences, no flags, no waiting for the writes to land.
  # workgroup wg of every device uses its own epoch counter and mailbox region, the regions alternate with the epoch parity
  # (a device can only be one exchange ahead of its peers). a lane writes its words before it waits for any: a peer lane may
  # be waiting for them. a GPU writing to several peers at once is slow when they sit behind different PCIe root complexes,
  # so a round only writes to one peer. returns the sums over the devices (the same bits on every device) and the counter update
  epoch, me = counter[wg].load() + 1, counter[G].load()
  ok = UOp.const(True, dtypes.bool) if valid is None else valid
  def slot(s, e:UOp) -> UOp: return _p2p_slot(epoch, s, wg, e, N, G, chunk, wide)
  if N & (N-1) == 0:
    # recursive doubling: in round r, exchange the running sum with device me^2^r, log2(N) rounds. a+b == b+a
    accs, deps = list(vals), cast(tuple[UOp, ...], ())
    for r in range(N.bit_length()-1):
      # the peer at offset j is device me+j: write through the alias that maps it
      ws = tuple(_ll_write(ok & (((me ^ (1 << r)) - me + N) % N).eq(j), peer[slot(r, e)], epoch, acc, deps, wide)
                 for j,peer in enumerate(peers, start=1) for acc,e in zip(accs, elems))
      accs = [_ll_round(acc, wide) + got for acc,got in zip(accs, _ll_read(ok, [mbox[slot(r, e)] for e in elems], epoch, ws, wide))]
      deps = tuple(accs)
    return accs, epoch
  # round j: write to device me+j (its slot N-j) and read what device me-j wrote to our slot N-j. sum in device order
  got:dict[int, list[UOp]] = {}
  deps = cast(tuple[UOp, ...], ())
  for j,peer in enumerate(peers, start=1):
    ws = tuple(_ll_write(ok, peer[slot(N-j, e)], epoch, v, deps, wide) for v,e in zip(vals, elems))
    got[N-j] = _ll_read(ok, [mbox[slot(N-j, e)] for e in elems], epoch, ws, wide)
    deps = tuple(got[N-j])
  def peer_val(a:int, i:int) -> UOp:
    s = (a - me + N) % N
    return functools.reduce(lambda x,y: s.eq(y[0]).where(y[1], x), ((k, got[k][i]) for k in range(1, N)), _ll_round(vals[i], wide))
  return [functools.reduce(lambda x,y: x+y, (peer_val(a, i) for a in range(N))) for i in range(len(vals))], epoch

@functools.cache
def _rms_norm_q8_kernel(xn:UOp, q:UOp, scale:UOp, xsum:UOp, *srcs:UOp, eps:float, add:bool, p2p:int=0, chunk:int=0) -> UOp:
  # one workgroup per token. h = x (+ a), xn = rmsnorm(h) * w, then xn (in its storage dtype) is quantized like q8_quantize.
  # with p2p, a is the sum over the devices of their partials, exchanged here
  h, x, a, w = (srcs[0], srcs[1], srcs[2], srcs[3]) if add else (None, srcs[0], None, srcs[1])
  D, tokens = cast(int, xn.shape[-1]), cast(int, xn.numel()) // cast(int, xn.shape[-1])
  W = _norm_waves(D)
  xn, x = xn.reshape(tokens, D), x.reshape(tokens, D)
  token, wave, lane = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  groups = [i*W + wave for i in range(D//(32*W))]
  elems = [g*32+lane for g in groups]
  vals = [x[token, e].load().float() for e in elems]
  stores:list[UOp] = []
  counter_update:tuple[UOp, UOp]|None = None
  if a is not None and h is not None:
    a, h = a.reshape(tokens, D), h.reshape(tokens, D)
    adds = [a[token, e].load().float() for e in elems]
    if p2p:
      counter, mbox, peers = srcs[4], srcs[5], srcs[6:]
      adds, epoch = _p2p_exchange(adds, elems, token, counter, mbox, peers, p2p, P2P_GROUPS, chunk)
      counter_update = (counter, epoch)
    vals = [v + ad for v,ad in zip(vals, adds)]
    stores += [h[token, e].store(v.cast(h.dtype)) for v,e in zip(vals, elems)]
  ss = warp_reduce(sum((v*v for v in vals[1:]), vals[0]*vals[0]), full_wave=True)
  if W > 1:
    lds = UOp.alloc((W,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds = lds.after(lds[wave.valid(lane.eq(0))].store(ss))
    ss = sum((lds[i].load() for i in range(1, W)), lds[0].load())
  rinv = ((ss * (1/D)) + eps).sqrt().reciprocal()
  for v,g,e in zip(vals, groups, elems):
    out = ((v * rinv) * w[e].load().float()).cast(xn.dtype)
    stores.append(xn[token, e].store(out))
    stores += _q8_pack(out.float(), lane, q, scale, xsum, token, g)
  if counter_update is not None:
    # the next epoch is stored once every lane of the workgroup is past the exchange (they all read the counter first)
    counter, epoch = counter_update
    stores = [counter.after(UOp.group(*stores).barrier())[token.valid(wave.eq(0) & lane.eq(0))].store(epoch)]
  return UOp.group(*stores).end(token, wave, lane).sink(arg=KernelInfo(name="rms_norm_q8" + ("_p2p" if p2p else ""), opts_to_apply=()))

def fused_norm_supported(norm:nn.RMSNorm, D:sint) -> bool:
  return isinstance(D, int) and _norm_waves(D) > 0 and norm.weight is not None and norm.weight.shape == (D,)

def p2p_partial(a:Tensor) -> Tensor:
  """the unreduced partials (sharded on a new leading axis) of a p2p_allreduce output, else a"""
  return pending[1] if (pending:=_p2p_pending.get(a.uop)) is not None else a

def rms_norm_q8(norm:nn.RMSNorm, x:Tensor, a:Tensor|None=None, dtype=dtypes.float32) -> tuple[Tensor, Tensor]:
  """returns (x + a, rmsnorm(x + a)) and registers the q8 quantization of the normed output for the linears that consume it.
  `a` can be the partials of a p2p_allreduce (see p2p_partial), they are summed over the devices here"""
  tokens, D = prod(x.shape[:-1]), x.shape[-1]
  assert isinstance(tokens, int) and isinstance(D, int) and _norm_waves(D) and norm.weight is not None
  xn = Tensor.empty(*x.shape, dtype=dtype, device=x.device)
  q = Tensor.empty(tokens, D//Q8_GROUP_SIZE, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, D//Q8_GROUP_SIZE, dtype=dtypes.float32, device=x.device)
  xsum = Tensor.empty(tokens, D//Q8_GROUP_SIZE, 2, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_rms_norm_q8_kernel, eps=norm.eps, add=a is not None)
  x = x.contiguous()
  if a is None: xn, q, scale, xsum = Tensor.custom_kernel(xn, q, scale, xsum, x, norm.weight, fxn=fxn)[:4]
  else:
    h = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device)
    extra:tuple[Tensor, ...] = ()
    # the row parallel output a is still an unreduced partial: the exchange happens in this kernel
    a = p2p_partial(a)
    if isinstance(a.device, tuple) and a.uop.axis == 0 and x.uop.axis is None and a.shape[0] == (N:=len(a.device)) and a.numel() == N*x.numel():
      a = a.reshape(N, tokens, D)
      if tokens <= P2P_WIDE_GROUP and (comm:=p2p_comm(a.device)) is not None:
        fxn = functools.partial(fxn, p2p=N, chunk=comm.chunk)
        extra = (comm.counter, comm.local, *comm.peers)
      else: a = a.sum(0)
    xn, q, scale, xsum, x = Tensor.custom_kernel(xn, q, scale, xsum, h, x, a.contiguous(), norm.weight, *extra, fxn=fxn)[:5]
  _q8_cache[xn.uop] = (q, scale, xsum)
  return x, xn

def _decode_rows(out_features:int, chunks:int) -> int:
  return next((r for r in (4, 2, 1) if out_features % r == 0 and r*chunks*WARP_SIZE <= 1024), 0)

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str) -> UOp:
  chunks = (group_count+WARP_SIZE-1)//WARP_SIZE
  # One wave per output row and 1024 inputs; the waves of a row sum their partials through LDS.
  # group neighboring rows in a workgroup to amortize workgroup scheduling.
  rows = _decode_rows(out_features, chunks)
  row = UOp.range(out.shape[0]*out_features//rows, 0, AxisType.GLOBAL)
  lane, wave, chunk = UOp.range(32, 2, AxisType.LOCAL), UOp.range(rows, 3, AxisType.LOCAL), UOp.range(chunks, 4, AxisType.LOCAL)
  token_output = row*rows+wave
  token, output = token_output // out_features, token_output % out_features
  group = lane+chunk*32
  value = (group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
  total = warp_reduce(value, full_wave=True)
  if chunks > 1:
    lds = UOp.alloc((rows, chunks), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds = lds.after(lds[wave, chunk.valid(lane.eq(0))].store(total))
    total = sum((lds[wave, c].load() for c in range(1, chunks)), lds[wave, 0].load())
  return out[token, output.valid(lane.eq(0) & chunk.eq(0))].store(total.cast(out.dtype)).end(row, wave, chunk, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

def _iq_grid(device:str|tuple[str, ...]|None, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common as ggml
  grid, words = {IQ2_XS: (ggml.iq2xs_grid, 2), IQ2_S: (ggml.iq2s_grid, 2),
                 IQ3_XXS: (ggml.iq3xxs_grid, 1), IQ3_S: (ggml.iq3s_grid, 1)}[ggml_type]
  return Tensor.const(tuple((v >> (32*i)) & 0xffffffff for v in grid for i in range(words)),
                      dtypes.uint32).to(device, force=True)

def _iq_even_signs(signs:UOp) -> UOp:
  parity = signs ^ (signs >> 4)
  parity ^= parity >> 2
  parity ^= parity >> 1
  return signs | ((parity & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  mask = sum(((signs >> i) & 1) * (255 << (8*i)) for i in range(4))
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  return (word ^ mask) + (mask & 0x01010101)

def _quant_word(raw:UOp, base:UOp, subgroup:UOp, i:int|UOp, ggml_type:int, grid:UOp|None) -> UOp:
  # Four packed weight bytes, shared by integer-dot decode and FP16 WMMA prefill.
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset, stream=ggml_type == Q6_K)
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
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp,
                         out_features:int, in_features:int, ggml_type:int) -> UOp:
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group//8, group%8
    base = (output*in_features//256 + block) * (QUANT_SIZES[ggml_type]//raw.dtype.itemsize)
    if ggml_type == IQ4_NL: base = (output*in_features//32 + group)*9
    def byte(offset): return _load_byte(raw, base, offset)
    def word(offset): return _load_u32(raw, base, offset)
    xwords = _amd_load(xq[token, group, 0], 8)
    # One accumulator per scale group: 32 weights for Q4/Q5/IQ4_XS, two groups of 16 otherwise.
    dots = [UOp.const(0, dtypes.int32)] * (1 if ggml_type in (Q4_K, Q5_K, IQ4_XS) else 2)
    for i in range(8):
      weights = _quant_word(raw, base, subgroup, i, ggml_type, grids[0] if grids else None)
      acc = i//(8//len(dots))
      dots[acc] = _amd_dp4a(weights, xwords[i], dots[acc])
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
    elif ggml_type in (IQ2_XS, IQ2_S):
      scales = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
      total = sum(dots[h].float() * (((scales >> (h*4)) & 15).float()+0.5) for h in range(2)) * 0.25 * _half(raw[base])
    else:
      total = (dots[0]+dots[1]).float() * _half(raw[base])
      if ggml_type == IQ3_S: total *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      if ggml_type == IQ3_XXS: total *= ((word(66+subgroup*4) >> 28).float()+0.5)*0.5
    return total * xd[token, group]
  return _decode_linear(out, out_features, in_features//32, group_dot, "linear_"+QUANT_NAMES[ggml_type])

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int):
  if out_features % (16*output_tiles): output_tiles = 1
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  token_block, output_block = UOp.range(out.shape[0]//token_tile, 0), UOp.range(out_features//(16*output_tiles*output_waves), 1)
  # lane is a hardware WARP range (like the flash kernel): the fragment math stays visible without being
  # range-split into nested loops, which would scramble the WMMA fragment layout
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  col, half = lane % 16, lane // 16
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(token_block*token_tile + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + half*8 + i for i in range(8)) for tile in range(token_tile//16))
  return output_waves, token_block, output_block, lane, wave, half, outputs, inputs, tokens

def _wmma_stores(out, outputs, tokens, accs, update, half, lane, wave, output_waves, rdna4):
  # RDNA4 owns eight consecutive rows per half-wave, matching tokens directly.
  if rdna4:
    return [out[token, output].store(acc.after(update)[i].load()) for output,output_accs in zip(outputs, accs)
            for tile_tokens,acc in zip(tokens, output_accs) for i,token in enumerate(tile_tokens)]
  # the accumulator fragment halves are exchanged between lane pairs (l, l^16) through LDS (a ds_swizzle without CUSTOM)
  flat_accs = [acc for output_accs in accs for acc in output_accs]
  lds = UOp.alloc((output_waves, 32, len(flat_accs)*8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  stores = [lds[wave, lane, a*8+i].store(acc.after(update)[i].load()) for a,acc in enumerate(flat_accs) for i in range(8)]
  lds = lds.after(*stores)
  def values(ai:int) -> tuple[UOp, ...]:
    own = tuple(lds[wave, lane, ai*8+i].load() for i in range(8))
    peer = tuple(lds[wave, lane ^ 16, ai*8+i].load() for i in range(8))
    low = half.eq(0)
    return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
  tt = len(tokens)
  return [out[token, output].store(value) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
          for tile,(tile_tokens,_acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, values(ot*tt+tile))]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, rdna4):
  x = x.reshape(out.shape[0], in_features)
  output_waves, token_block, output_block, lane, wave, physical_half, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  # IQ4_NL blocks are 32 wide: a device can hold part of a 256 weight row group
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  accs = tuple(tuple(UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.LOOP)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  # gfx12 moves k2 from the element index to lane bit 4: each lane supplies two groups of four.
  ks = tuple(i%4 + physical_half*4 + (i//4)*8 for i in range(8)) if rdna4 else tuple(range(16))
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in ks))
                   for input_token in input_tokens)
    for output_tile,output in enumerate(outputs):
      bfrag = UOp.stack(*dequant(output*output_words + block*type_words, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, *WMMA_ARG)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)
  stores = _wmma_stores(out, outputs, tokens, accs, update, physical_half, lane, wave, output_waves, rdna4)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int, rdna4:bool=False) -> UOp:
  token_tile, output_tiles = (64, 1 if out_features <= 1024 else 2) if out.shape[0] % 64 == 0 else \
    (32 if out.shape[0] % 32 == 0 else 16, 2)
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  word_indices = (layout[5], layout[5]+2) if rdna4 else tuple(range(4))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
    words = tuple((raw[qs_base+i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      (((raw[base+4+half*4+i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if ggml_type == Q5_K else 0) for i in word_indices)
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//4,
                            layout, dequant, f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma", rdna4)

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int, rdna4:bool=False) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    out_features <= 6144 else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  output_waves, _, _, lane, wave, half, _, _, _ = layout
  word_indices = (half, half+2) if rdna4 else tuple(range(4))
  local_lut = UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*32+lane, 256//(32*output_waves)
  lut = local_lut.after(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    scale = _iq4_scale(raw, base, subgroup)
    pairs = tuple(lut[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                  for word in word_indices for byte in range(4))
    return tuple((_half((pair >> (half*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in pairs)
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[IQ4_XS]//4, layout, dequant, "linear_iq4_xs_f16_wmma", rdna4)

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp,
                                  out_features:int, in_features:int, ggml_type:int, rdna4:bool=False) -> UOp:
  token_tile = 64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16
  layout = _wmma_layout(out, out_features, token_tile, 2)
  output_waves, _, _, lane, wave, physical_half, _, _, _ = layout
  word_indices = (physical_half, physical_half+2) if rdna4 else tuple(range(4))
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*32+lane, output_waves*32
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    if ggml_type == IQ4_NL: base += subgroup*9  # eight independent 18-byte blocks per 256 weights
    def byte(offset): return _load_byte(raw, base, offset)
    zero, minimum = 0, UOp.const(0, dtypes.float)
    if ggml_type == Q2_K:
      sc = byte(subgroup*2+half)
      scale, minimum = _half(raw[base+40])*(sc & 15).float(), _half(raw[base+41])*(sc >> 4).float()
    elif ggml_type == Q3_K:
      j = subgroup*2+half
      sc = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
      scale, zero = _half(raw[base+54])*(sc.cast(dtypes.int32)-32).float(), 4
    elif ggml_type == Q6_K:
      scale, zero = _half(raw[base+104])*byte(192+subgroup*2+half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
    else:
      scale = _half(raw[base])
      if ggml_type in (IQ2_XS, IQ2_S):
        sc = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
        scale *= (((sc >> (half*4)) & 15).float()+0.5)*0.25
      elif ggml_type == IQ3_S: scale *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      elif ggml_type == IQ3_XXS: scale *= ((_load_u32(raw, base, 66+subgroup*4) >> 28).float()+0.5)*0.5
    words = tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in word_indices)
    return tuple((((word >> (b*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()-zero)*scale-minimum).cast(dtypes.half)
                 for word in words for b in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//raw.dtype.itemsize,
                            layout, dequant, f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", rdna4)

def q8_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in QUANT_SIZES
  tokens = int(x.numel()) // layer.in_features
  out_features, in_features = layer.out_features, int(x.uop.shard_shape[-1])
  splits = layer.in_features // in_features
  assert (layer.shard_axis == 1) == (splits > 1), f"input features split over {splits} devices, weight split on axis {layer.shard_axis}"
  out_shape:tuple[int, ...] = (splits*tokens, out_features)
  fxn:Callable[..., UOp]
  extra = (_iq_grid(x.device, layer.ggml_type),) if layer.ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  # the kernels write the output features of their own device
  local_out = out_features // len(dev) if isinstance(dev:=layer.weight.device, tuple) and layer.shard_axis == 0 else out_features
  if tokens % 16 == 0 and local_out % 16 == 0:
    if layer.ggml_type == IQ4_XS:
      fxn, extra = _iq4_linear_f16_wmma_kernel, (iq4_half_lut(x.device),)
    else:
      fxn = functools.partial(_q5_linear_f16_wmma_kernel if layer.ggml_type in (Q4_K, Q5_K) else _quant_linear_f16_wmma_kernel,
                              ggml_type=layer.ggml_type)
    fxn = functools.partial(fxn, rdna4=_wmma_rdna4(x.device))
    srcs = (x.cast(dtypes.float16).contiguous(), *extra)
  else:
    fxn = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
    srcs = (*q8_quantize(x, tokens, in_features), *extra)
    assert _decode_rows(local_out, (in_features+1023)//1024), f"{in_features} input features don't fit a decode workgroup"
  out = Tensor.empty(out_shape, dtype=dtypes.float32, device=x.device, axis=None if layer.shard_axis is None else 1-layer.shard_axis)
  result = Tensor.custom_kernel(out, layer.weight, *srcs, fxn=functools.partial(fxn, out_features=local_out, in_features=in_features))[0]
  # row parallel: every device wrote the partial sum of its input features, the sum over the devices is the allreduce
  if splits > 1 and isinstance(x.device, tuple) and p2p_supported(comm:=p2p_comm(x.device), (tokens, out_features)) and comm is not None:
    result = p2p_allreduce(comm, result, (tokens, out_features))
  else:
    if len(result.shape) == 3: result = result.sum(-1)
    if splits > 1: result = result.reshape(splits, tokens, out_features).sum(0)
  out = result.reshape(*x.shape[:-1], out_features)
  if layer.bias is not None: return out + layer.bias
  if (pending:=_p2p_pending.get(result.uop)) is not None: _p2p_pending[out.uop] = pending
  return out

# ******** tiny dense fp16 gemv ********

@functools.cache
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int) -> UOp:
  bias: UOp|None = rest[0] if rest else None
  # one block per (token, output row), 32 lanes accumulate 4-wide chunks of the row
  lanes, val_chunk = WARP_SIZE, 4
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL)
  per = in_features // (lanes * val_chunk)
  assert per * lanes * val_chunk == in_features
  w = w.reshape((out_features, per, lanes*val_chunk))
  x = x.reshape((tokens, per, lanes*val_chunk))
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    for j in range(val_chunk):
      acc = acc + w[out_row, i, lane*val_chunk + j].load().float() * x[token, i, lane*val_chunk + j].load().float()
  total = warp_reduce(acc, full_wave=True)
  if bias is not None: total = total + bias[out_row].load().float()
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

def _view_back(t:Tensor) -> Tensor:
  # Widening half to float is exact; preserve casts that round or change the values.
  uop = t.uop
  while uop.op is Ops.CAST and uop.dtype == dtypes.float32 and uop.src[0].dtype in (dtypes.half, dtypes.bfloat16): uop = uop.src[0]
  return Tensor(uop).reshape(t.shape)

def f16_gemv(layer:Linear, x:Tensor) -> Tensor:
  tokens = prod(x.shape[:-1])
  assert isinstance(tokens, int)
  weight = _view_back(layer.weight)
  x = x.contiguous()
  # column parallel: every device computes the rows of its own weight shard
  sharded = isinstance(wdev:=layer.weight.device, tuple) and layer.weight.uop.axis == 0
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device, axis=1 if sharded else None)
  local_out = layer.out_features // len(wdev) if isinstance(wdev, tuple) and sharded else layer.out_features
  fxn = functools.partial(_amd_f16_gemv_kernel, in_features=layer.in_features, out_features=local_out, tokens=tokens)
  srcs = (out, weight.reshape(-1), x.reshape(tokens, layer.in_features)) + (() if layer.bias is None else (_view_back(layer.bias),))
  return Tensor.custom_kernel(*srcs, fxn=fxn)[0].reshape(*x.shape[:-1], layer.out_features)

# ******** flash attention on the KV cache ********

def _vec_load(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  if lanes == 1: return (ptr.load().float(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i].float() for i in range(lanes))

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, valid_kv_len, max_kv_len, block_n, waves=4):
  valid_kv_len = _unbind(valid_kv_len)
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DPL, WAVES, PARTIALS = H // H_KV, block_n, D // WARP_SIZE, waves, out.shape[2]
  assert CHUNK % WAVES == 0
  SEC = CHUNK // WAVES  # keys each wave scans independently
  total_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  live_chunks = min(total_chunks, PARTIALS) if isinstance(total_chunks, int) else total_chunks.minimum(PARTIALS)
  block_bhkv, block_chunk = UOp.range(B*H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  # per-lane query fragments for every GQA head, kept packed in registers; unpacked at use
  qf = tuple(_vec_load(q[b, kv_head*G+h, 0, lane*DPL], DPL) for h in range(G))
  zerof = UOp.const(0, dtypes.float)
  # Each block scans every PARTIALS-th chunk, keeping an online softmax across rounds.
  chunk_round = UOp.range((total_chunks-1-block_chunk)//PARTIALS+1, 4, AxisType.LOOP)
  chunk_id = block_chunk + chunk_round*PARTIALS
  valids: list[UOp] = []
  scores: list[list[UOp]] = [[zerof]*G for _ in range(SEC)]
  vfrags: list[tuple[UOp, ...]] = [()]*SEC
  for j in range(SEC):
    key = chunk_id*CHUNK + wave*SEC + j
    valid = key < valid_kv_len
    valids.append(valid)
    kfrag = _vec_load(cache_kv[0, b, kv_head, key, lane*DPL], DPL)
    # V is prefetched in the score pass so both streams are in flight together
    vfrags[j] = tuple(valid.where(v, zerof) for v in _vec_load(cache_kv[1, b, kv_head, key, lane*DPL], DPL))
    for h in range(G):
      s = warp_reduce(sum((qf[h][i]*kfrag[i] for i in range(DPL)), UOp.const(0, dtypes.float)), full_wave=True) * (1/math.sqrt(D))
      scores[j][h] = valid.where(s, UOp.const(-1e30, dtypes.float))
  # A finite initial max keeps fully masked waves from computing exp(-inf - -inf).
  acc_reg, max_reg, sum_reg = _reg((G, DPL), 0), _reg((G,), -1e30), _reg((G,), 0)
  prev_acc, prev_max, prev_sum = acc_reg.after(chunk_round), max_reg.after(chunk_round), sum_reg.after(chunk_round)
  row_max = [functools.reduce(UOp.maximum, (scores[j][h] for j in range(SEC)), prev_max[h].load()) for h in range(G)]
  # Rescale the previous rounds to the new max, then accumulate this round's keys.
  alpha = [((prev_max[h].load()-row_max[h])*LOG2E).exp2() for h in range(G)]
  accs = [[alpha[h]*prev_acc[h, i].load() for i in range(DPL)] for h in range(G)]
  row_sums = [alpha[h]*prev_sum[h].load() for h in range(G)]
  for j in range(SEC):
    for h in range(G):
      beta = valids[j].where(((scores[j][h]-row_max[h])*LOG2E).exp2(), zerof)
      accs[h] = [a + beta*v for a, v in zip(accs[h], vfrags[j])]
      row_sums[h] = row_sums[h] + beta
  update = UOp.group(acc_reg.store(UOp.stack(*(x for acc in accs for x in acc)).reshape(G, DPL)),
                     max_reg.store(UOp.stack(*row_max)), sum_reg.store(UOp.stack(*row_sums))).end(chunk_round)
  acc_reg, max_reg, sum_reg = acc_reg.after(update), max_reg.after(update), sum_reg.after(update)
  # exchange across the block's waves through LDS (fp16 halves LDS so more blocks fit per CU)
  # Matching cache/LDS strides can reuse a loop-local cache index outside the loop. Pad that layout.
  acc_lds = UOp.alloc((WAVES, G, D + (LDS_PAD if G == SEC else 0)), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :, :D]
  ml_lds = UOp.alloc((WAVES, G, 2), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_acc = acc_lds.reshape(WAVES, G, WARP_SIZE, DPL)
  # Normalize before fp16 to avoid overflow. Nonempty waves have sum >= 1; empty waves keep their zero accumulator.
  stores = [lds_acc[wave, h, lane].store((acc_reg[h].load() / sum_reg[h].load().maximum(1)).cast(dtypes.half)) for h in range(G)]
  # NOTE: duplicate stores of the same value from every lane are harmless here
  stores += [ml_lds[wave, h, i].store(x) for h in range(G) for i, x in enumerate((max_reg[h].load(), sum_reg[h].load()))]
  acc_lds, ml_lds = acc_lds.after(*stores), ml_lds.after(*stores)
  tid = wave*WARP_SIZE + lane
  final_stores:list[UOp] = []
  for i in range(-(-G*D//(WAVES*WARP_SIZE))):
    flat = tid + i*WAVES*WARP_SIZE
    h, d = flat // D, flat % D
    M = functools.reduce(UOp.maximum, (ml_lds[w, h, 0].load() for w in range(WAVES)))
    # LDS holds normalized values; restore each wave's sum before combining.
    val = sum((((ml_lds[w, h, 0].load()-M)*LOG2E).exp2() * ml_lds[w, h, 1].load() * acc_lds[w, h, d].load().float()
               for w in range(WAVES)), zerof)
    oidx = out[b, kv_head*G + h, block_chunk, d]
    if G*D % (WAVES*WARP_SIZE): oidx = out[b, (kv_head*G + h).valid(flat < G*D), block_chunk, d]
    final_stores.append(oidx.store(val))
  hstat = tid
  M = functools.reduce(UOp.maximum, (ml_lds[w, hstat, 0].load() for w in range(WAVES)))
  L = sum((((ml_lds[w, hstat, 0].load()-M)*LOG2E).exp2() * ml_lds[w, hstat, 1].load() for w in range(WAVES)), zerof)
  q_head = (kv_head*G + hstat).valid(hstat < G) if WAVES*WARP_SIZE > G else kv_head*G + hstat
  final_stores += [stats[b, q_head, block_chunk, 0].store(M), stats[b, q_head, block_chunk, 1].store(L)]
  return UOp.group(*final_stores).end(lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, live:int|UOp) -> UOp:
  # one wave per (batch, head, 64-dim tile): every lane redundantly weights its chunks; no cross-lane traffic
  live = _unbind(live)
  B, H, C, D = cast(tuple[int, int, int, int], partial.shape)
  DT = 64 if D % 64 == 0 else WARP_SIZE  # dims per block
  assert D % DT == 0
  block_bh, block_dt = UOp.range(B*H, 0, AxisType.GLOBAL), UOp.range(D//DT, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  b, h = block_bh // H, block_bh % H
  NPD = DT // WARP_SIZE  # output dims per lane
  dims = tuple(block_dt*DT + lane*NPD + i for i in range(NPD))
  chunk = UOp.range(live, 100, AxisType.LOOP)
  def iloop(ph, val): return ph.store(ph.const_like(val))
  chunk_max = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  chunk_max_i = chunk_max.after(iloop(chunk_max, -math.inf))
  update0 = chunk_max_i.store(chunk_max_i.after(chunk).maximum(stats[b, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max_i.after(update0)
  chunk2 = UOp.range(live, 101, AxisType.LOOP)
  acc = UOp.alloc((NPD,), dtypes.float, addrspace=AddrSpace.REG)
  weight_sum = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  acc_i, weight_sum_i = acc.after(iloop(acc, 0)), weight_sum.after(iloop(weight_sum, 0))
  w = ((stats[b, h, chunk2, 0].load()-chunk_max)*LOG2E).exp2()
  update1 = UOp.group(*[acc_i[i].store(acc_i.after(chunk2)[i].load() + w*partial[b, h, chunk2, d].load()) for i, d in enumerate(dims)],
                      weight_sum_i[0].store(weight_sum_i.after(chunk2)[0].load() + w*stats[b, h, chunk2, 1].load())).end(chunk2)
  acc, weight_sum = acc_i.after(update1), weight_sum_i.after(update1)
  inv = 1 / weight_sum[0].load()
  return UOp.group(*[o[b, h, 0, d].store(acc[i].load() * inv) for i, d in enumerate(dims)]) \
    .end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine_gated(o:UOp, xq:UOp, xscale:UOp, xsum:UOp, partial:UOp, stats:UOp, gate:UOp, live:int|UOp) -> UOp:
  # one wave per (batch, head, 32 dims), a lane per dim: o = attention * sigmoid(gate), quantized for the output projection.
  # gate is the query projection (B, H, 2, D): the gate is the second half of every head
  live = _unbind(live)
  B, H, C, D = cast(tuple[int, int, int, int], partial.shape)
  o, gate = o.reshape(B, H*D), gate.reshape(B, H*2*D)
  block_bh, block_dt = UOp.range(B*H, 0, AxisType.GLOBAL), UOp.range(D//32, 1, AxisType.GLOBAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  b, h = block_bh // H, block_bh % H
  d = block_dt*32 + lane
  chunk = UOp.range(live, 100, AxisType.LOOP)
  mx = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  mx = mx.after(mx.store(mx.const_like(-math.inf)))
  mx = mx.after(mx.store(mx.after(chunk).maximum(stats[b, h, chunk, 0].load())).end(chunk))
  chunk2 = UOp.range(live, 101, AxisType.LOOP)
  acc = UOp.alloc((2,), dtypes.float, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  w = ((stats[b, h, chunk2, 0].load()-mx[0].load())*LOG2E).exp2()
  cur = acc.after(chunk2)
  acc = acc.after(acc.store(UOp.stack(cur[0].load() + w*partial[b, h, chunk2, d].load(), cur[1].load() + w*stats[b, h, chunk2, 1].load()))
                  .end(chunk2))
  val = acc[0].load() / acc[1].load() * _sigmoid(gate[b, h*2*D + D + d].load().float())
  stores = (o[b, h*D + d].store(val),) + _q8_pack(val, lane, xq, xscale, xsum, b, h*(D//32) + block_dt)
  return UOp.group(*stores).end(block_dt, block_bh, lane).sink(arg=KernelInfo(name="flash_decode_combine_gated", opts_to_apply=()))

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int, gate:Tensor|None=None) -> Tensor:
  """attention of one query token on the half KV cache. with gate (the gated query projection (B, 1, H*2*D)), returns the gated
  output (B, 1, H*D) with its q8 quantization registered"""
  # Carry length bindings even when the cache was populated independently of this call.
  if isinstance(valid_kv_len, UOp): cache_kv = Tensor(cache_kv.uop.after(valid_kv_len))
  B, H, D = cache_kv.shape[1], q.shape[1], cache_kv.shape[4]
  # small blocks: few keys per wave keep a short context fast, more blocks keep a long one parallel
  BN = 16
  chunks, axis = min(64, max_kv_len // BN), q.uop.axis
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device, axis=axis)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device, axis=axis)
  waves, group = 8, H // cache_kv.shape[2]
  while waves * group * ((D+LDS_PAD)*2 + 8) > 65536: waves //= 2
  assert waves > 0, "attention head group exceeds shared memory capacity"
  fxn = functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=BN, waves=waves)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv, fxn=fxn)[:2]
  live = (valid_kv_len+BN-1)//BN
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  if gate is not None:
    # the gated output (B, 1, H*D) with its quantization registered for the output projection
    gaxis = None if axis is None else 2
    out = Tensor.empty(B, 1, H*D, dtype="float32", device=q.device, axis=gaxis)
    xq = Tensor.empty(B, H*D//Q8_GROUP_SIZE, 8, dtype=dtypes.uint32, device=q.device, axis=None if axis is None else 1)
    xscale = Tensor.empty(B, H*D//Q8_GROUP_SIZE, dtype=dtypes.float32, device=q.device, axis=None if axis is None else 1)
    xsum = Tensor.empty(B, H*D//Q8_GROUP_SIZE, 2, dtype=dtypes.float32, device=q.device, axis=None if axis is None else 1)
    fxn = functools.partial(_amd_flash_decode_combine_gated, live=live)
    out, xq, xscale, xsum = Tensor.custom_kernel(out, xq, xscale, xsum, partial, stats, gate.contiguous(), fxn=fxn)[:4]
    _q8_cache[out.uop] = (xq, xscale, xsum)
    return out
  out = Tensor.empty(B, H, 1, D, dtype="float32", device=q.device, axis=axis)
  fxn = functools.partial(_amd_flash_decode_combine, live=live)
  return Tensor.custom_kernel(out, partial, stats, fxn=fxn)[0]

def _wmma_fragment(fragment:UOp, lane:UOp, rdna4:bool) -> UOp:
  return fragment.reshape(2, 2, 4)[:, lane//16, :].reshape(8) if rdna4 else fragment

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, valid_kv_len:int|UOp, q_start:int|UOp|None=None, rdna4:bool=False) -> UOp:
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start) if q_start is not None else None
  BH, M, D = q.shape
  _, B, H_KV, physical_n, cache_dim = cache.shape
  k, v = cache[0].reshape(B*H_KV, physical_n, cache_dim), cache[1].reshape(B*H_KV, physical_n, cache_dim)
  assert k.shape == v.shape and BH % k.shape[0] == 0 and k.shape[2] == D
  gqa_group = BH // k.shape[0]
  if isinstance(M, int): assert M % BLOCK_M == 0
  assert isinstance(D, int) and D % WMMA_K == 0 and D % LANES_PER_WAVE_N == 0
  TM, TN, TD, SCALE = BLOCK_M//(WAVES_M*LANES_PER_WAVE_M), BLOCK_N//LANES_PER_WAVE_N, D//(WAVES_N*LANES_PER_WAVE_N), 1/math.sqrt(D)
  # query row 0 sits at sequence position q_base (the queries may be padded beyond valid_kv_len - q_base rows)
  q_base = valid_kv_len - M if q_start is None else q_start
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)
  kv_head = block_bh // gqa_group
  q, o = (x.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m] for x in (q, o))
  k, v = k[kv_head], v[kv_head]
  wave_m, wave_n, lane = UOp.range(WAVES_M, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, lane_m, lane_n = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane, lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  Q_ELEMS_PER_THREAD, KV_ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK, BLOCK_N * D // THREADS_PER_BLOCK
  QP_lds = UOp.alloc((BLOCK_M, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)
  KV_lds = UOp.alloc((BLOCK_N, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :D]
  acc, m_i, l_i = _reg((TM, TD), 0), _reg((TM,), -math.inf), _reg((TM,), 0)
  n_tile = UOp.range(((q_base + (block_m + 1) * BLOCK_M).minimum(valid_kv_len) + BLOCK_N - 1) // BLOCK_N, 100, AxisType.LOOP)
  Q_lds = QP_lds[:, :D]
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
  load_k = UOp.range(KV_ELEMS_PER_THREAD, 90)
  kval = k.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_k].float()
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_k].store(kval).end(load_k)
  Q_lds, KV_lds_k = Q_lds.after(Q_store, K_store), KV_lds.after(Q_store, K_store)
  S_reg = _reg((TM, TN), 0, n_tile)
  k_qk, tm1, tn1 = UOp.range(D//WMMA_K, 101, AxisType.LOOP), UOp.range(TM//WMMA_ACC, 200), UOp.range(TN, 201)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
  # All waves must finish reading Q/K before their shared memory is reused for P/V.
  q_frag, k_frag = (_wmma_fragment(f, lane, rdna4) for f in (q_frag, k_frag))
  qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk).barrier()
  S_reg = S_reg.after(qk_done, S_reg.store(S_reg * SCALE))
  rm, rn = UOp.range(TM, 250), UOp.range(TN, 251)
  q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + (lane_m*TM + rm if rdna4 else rm*LANES_PER_WAVE_M + lane_m)
  k_idx = n_tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
  causal = (k_idx <= q_idx) & (k_idx < valid_kv_len)
  S_reg = S_reg.after(S_reg[rm, rn].store(causal.where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))).end(rm, rn))
  m_ij, rm2 = _reg((TM,), -math.inf, n_tile), UOp.range(TN, 261, AxisType.LOOP)
  m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
  ri_w = UOp.range(TM, 270)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], maximum=True)).end(ri_w))
  tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
  S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
  p_local, ri_ws = _reg((TM,), 0, n_tile), UOp.range(TM, 295)
  p_sum = p_local.after(p_local[ri_ws].store(sum((warp_reduce(S_reg[ri_ws, rn]) for rn in range(TN)), S_reg.const_like(0))).end(ri_ws))
  P_lds = QP_lds.flatten()[:WAVES_N * BLOCK_M * BLOCK_N].reshape(WAVES_N, BLOCK_M, BLOCK_N)
  # gfx11 distributes even/odd rows between half-waves; gfx12 distributes the low/high eight rows.
  row_shape = (LANES_PER_WAVE_M, TM) if rdna4 else (TM, LANES_PER_WAVE_M)
  row_lane, row_elem = (2, 3) if rdna4 else (3, 2)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, *row_shape, TN, LANES_PER_WAVE_N).permute(1, 0, row_lane, 5, row_elem, 4) \
    .reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))
  beta_i, ri4, rj4 = UOp.alloc((TM,), dtypes.float, addrspace=AddrSpace.REG), UOp.range(TM, 330), UOp.range(TD, 331)
  m_new = m_i[ri4].maximum(m_ij[ri4])
  alpha_val, beta_val = ((m_i[ri4] - m_new) * LOG2E).exp2(), ((m_ij[ri4] - m_new) * LOG2E).exp2()
  correction = UOp.group(acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
                         l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
                         m_i[ri4].store(m_new), beta_i[ri4].store(beta_val)).end(ri4)
  acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  V_lds = UOp.alloc((D, BLOCK_N + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_copy, load_v = V_lds.after(qk_done).permute(1, 0), UOp.range(KV_ELEMS_PER_THREAD, 390)
  v_pos = n_tile*BLOCK_N + (tid*KV_ELEMS_PER_THREAD + load_v)//D
  vval = (v_pos < valid_kv_len).where(v.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_v].float(), 0)
  V_store = V_copy.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_v].store(vval).end(load_v)
  P_lds, V_lds = P_lds.after(P_store, V_store), V_lds.after(P_store, V_store)
  pv_acc = _reg((TM, TD), 0, n_tile)
  k_pv, tm2, tn2 = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.LOOP), UOp.range(TM//WMMA_ACC, 401), UOp.range(TD, 402)
  pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  p_frag, v_frag = (_wmma_fragment(f, lane, rdna4) for f in (p_frag, v_frag))
  pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)
  ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
  n_tile_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).end(n_tile)
  acc, l_i, m_i = acc.after(n_tile_end), l_i.after(n_tile_end), m_i.after(n_tile_end)
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))
  o = o.reshape(WAVES_M, *row_shape, WAVES_N, TD, LANES_PER_WAVE_N) \
    .permute(0, 3, row_lane-1, 5, row_elem-1, 4).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

def flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  # cached flash attention on the half KV cache (already written through assigned_kv); valid_end stays bound at the graph level
  T_real, q_start = q.shape[2], None
  D, N, group = q.shape[3], assigned_kv.shape[3], q.shape[1] // assigned_kv.shape[2]
  decode = resolve(T_real == 1, False)
  # Non-power-of-two decode dimensions can lose tail-store masks. Q/P, K, and V use separate LDS allocations.
  supported = D % 32 == 0 and (D & (D-1) == 0 and N % 64 == 0 and group*((D+LDS_PAD)*2+8) <= 65536 if decode else
    D >= 64 and 2*(2*BLOCK_M*(D+LDS_PAD) + D*(BLOCK_N+LDS_PAD)) <= 65536 and N % BLOCK_N == 0 and q.max_shape[2] % BLOCK_M == 0)
  if not supported:
    k, v = (assigned_kv[i, :, :, :valid_end].float() for i in range(2))
    mask = None if decode else Tensor.full((T_real, valid_end), -math.inf, dtype=dtypes.float32, device=q.device).triu(valid_end-T_real+1)
    return q.float().scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
  if decode: return amd_flash_attention_decode(q.half(), assigned_kv, valid_end, cast(int, N))
  if isinstance(T_real, UOp):
    # symbolic chunk: pad the queries to the static tile; garbage rows are sliced off
    T_pad = q.max_shape[2]
    assert T_pad % BLOCK_M == 0, "chunk_size must be a multiple of 32"
    q, q_start = q.pad_to((*q.shape[:2], T_pad, q.shape[3])), valid_end - T_real
  B, H, T, D = q.shape
  out = Tensor.empty(B, H, T, D, dtype="float32", device=q.device, axis=q.uop.axis).reshape(B*H, T, D)
  fxn = functools.partial(_amd_flash_attention, valid_kv_len=valid_end, q_start=q_start, rdna4=_wmma_rdna4(q.device))
  if isinstance(valid_end, UOp): assigned_kv = Tensor(assigned_kv.uop.after(valid_end))
  out = Tensor.custom_kernel(out, q.half().reshape(B*H, T, D), assigned_kv, fxn=fxn)[0].reshape(B, H, T, D)
  return out if q_start is None else out[:, :, :T_real]

# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows, cols = tuple(row_base+i for i in range(row_tile)), tuple(lane + i*32 for i in range(key_dim//32))
  current = UOp.alloc((row_tile*key_dim//32,), dtypes.float32, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(state[bh, row, col].float() if initial is None else
    initial.where(0, state[bh, row, col].float()) for row in rows for col in cols))))
  token = UOp.range(tokens, 2, AxisType.LOOP)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_dim//32+i].load() for i in range(key_dim//32))
    decayed, bv = tuple(x * alpha[bh, token, col if alpha_dim > 1 else 0].load() for x,col in zip(previous, cols)), beta[bh, token].load()
    state_k = warp_reduce(sum((x*y for x,y in zip(decayed, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
    state_q = warp_reduce(sum((x*y for x,y in zip(decayed, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k) * bv
    updates += [x + delta*y for x,y in zip(decayed, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens) and state.shape == (batch, heads, value_dim, key_dim)
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, key_dim))
  assert key_dim % 32 == 0 and value_dim % 4 == 0
  assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
  assert state.uop.contiguous_view_offset() is not None, "recurrent state must be contiguous"
  if start_pos is not None:
    assert start_pos.uop.is_bound_var
    state = Tensor(state.uop.after(start_pos.uop))
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  call = _gated_delta_prefill_kernel(*params, None if start_pos is None else start_pos.uop.unbound()).call(*contig)
  return Tensor(contig[0].after(call))

# ******** p2p allreduce: kernels write their partials straight into the peers' memory ********

# The queue level allreduce costs a copy plus two cross device signals per call (~25us on PCIe). Here every device pushes its partial
# into a mailbox in the VRAM of each peer, then spins on the flags the peers set. Workgroup w only exchanges with workgroup w of the
# peers, each through its own mailbox region, flag and epoch counter, so no workgroup waits for another one on its device.
# The data slots alternate with the epoch parity: a device can only be one allreduce ahead of its peers.
# workgroup regions 0..14 carry narrow words (a row each), 15 the wide ones
P2P_GROUPS, P2P_WIDE_GROUP, P2P_MAXN = 16, 15, 1 << 18

class P2PComm:
  def __init__(self, devs:tuple[str, ...]):
    from tinygrad.device import Buffer, BufferSpec, BufferStorage, MultiBuffer
    N, self.devs, self.chunk = len(devs), devs, P2P_MAXN // P2P_GROUPS
    nbytes = 2*N*P2P_MAXN*4  # two parities, a region per device, 32-bit words
    def multi(bufs:list, dtype) -> Tensor:
      mb = MultiBuffer.__new__(MultiBuffer)
      mb.bufs = bufs
      return Tensor(UOp.from_buffer(mb, dtype, devs))
    with Context(ALLOW_DEVICE_USAGE=1):
      # uncached: the peers write these over PCIe, so they must not sit stale in L2
      self.mbox = [Buffer(d, nbytes, options=BufferSpec(uncached=True)).ensure_allocated() for d in devs]
      for b in self.mbox: b.copy_from(Buffer("PYTHON", nbytes, initial_value=bytes(nbytes)))
      self.counters = [Buffer(d, (P2P_GROUPS+1)*4).ensure_allocated() for d in devs]
      for i,b in enumerate(self.counters): b.copy_from(Buffer("PYTHON", b.nbytes, initial_value=bytes(P2P_GROUPS*4)+i.to_bytes(4, "little")))
      # peer j of device d is the mailbox of device (d+j)%N, mapped into the address space of d. it is a buffer of d, so the graph
      # doesn't add cross device waits for it
      self.aliases = [[Buffer(d, nbytes, opaque=BufferStorage(self.mbox[(i+j)%N].get_buf(d))) for i,d in enumerate(devs)] for j in range(1, N)]
    self.local, self.counter = multi(self.mbox, dtypes.float32), multi(self.counters, dtypes.int32)
    self.peers = [multi(a, dtypes.float32) for a in self.aliases]

@functools.cache
def p2p_comm(devs:tuple[str, ...]) -> P2PComm|None:
  if getenv("P2P_ALLREDUCE", 1) == 0 or len(devs) < 2 or not amd_custom_kernels_supported(devs): return None
  with Context(ALLOW_DEVICE_USAGE=1):
    # the peers' VRAM must be mappable: tinygrad's PCI interface on one PCIe root with large BARs
    ifaces:list = [getattr(Device[d], "iface", None) for d in devs]
    if any(type(i).__name__ != "PCIIface" for i in ifaces): return None
    if len({i.peer_group for i in ifaces}) != 1 or any(i.is_bar_small() for i in ifaces): return None
  return P2PComm(devs)

@functools.cache
def _p2p_allreduce_kernel(out:UOp, partial:UOp, counter:UOp, mbox:UOp, *peers:UOp, N:int, chunk:int, exact:bool=False) -> UOp:
  # a workgroup per row, like the exchange in rms_norm_q8
  D = cast(int, out.shape[-1])
  rows = cast(int, out.numel()) // D
  W = _norm_waves(D)
  assert W and rows <= (1 if exact else P2P_WIDE_GROUP) and D*(2 if exact else 1) <= chunk, f"p2p allreduce of {rows}x{D} doesn't fit"
  out, partial = out.reshape(rows, D), partial.reshape(rows, D)
  row, wave, lane = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  # the 64-bit words of an exact exchange get their own workgroup region: a stale wide word must never pass for a narrow one
  wg = row + P2P_WIDE_GROUP if exact else row
  elems = [(i*W + wave)*32 + lane for i in range(D//(32*W))]
  totals, epoch = _p2p_exchange([partial[row, e].load().float() for e in elems], elems, wg, counter, mbox, peers, N, P2P_GROUPS, chunk,
                                wide=exact)
  done = UOp.group(*[out[row, e].store(t) for t,e in zip(totals, elems)]).barrier()
  return counter.after(done)[wg.valid(wave.eq(0) & lane.eq(0))].store(epoch).end(row, wave, lane).sink(
    arg=KernelInfo(name=f"p2p_allreduce_{D}", opts_to_apply=()))

# a consumer that fuses the exchange (rms_norm_q8) takes the partial of a p2p_allreduce output instead: (comm, partial)
_p2p_pending:weakref.WeakKeyDictionary[UOp, tuple[P2PComm, Tensor]] = weakref.WeakKeyDictionary()

def p2p_supported(comm:P2PComm|None, shape:tuple[int, ...]) -> bool:
  return comm is not None and isinstance(D:=shape[-1], int) and _norm_waves(D) > 0 and prod(shape[:-1]) <= P2P_WIDE_GROUP and D <= comm.chunk

def p2p_allreduce(comm:P2PComm, partial:Tensor, shape:tuple[int, ...], exact:bool=False) -> Tensor:
  """sum over the devices of `partial` (sharded on axis 0, each device holds `shape`) -> replicated `shape` (rows of the last axis).
  the values travel with 15 mantissa bits unless exact"""
  out = Tensor.empty(*shape, dtype=dtypes.float32, device=comm.devs)
  fxn = functools.partial(_p2p_allreduce_kernel, N=len(comm.devs), chunk=comm.chunk, exact=exact)
  out = Tensor.custom_kernel(out, partial.float(), comm.counter, comm.local, *comm.peers, fxn=fxn)[0]
  _p2p_pending[out.uop] = (comm, partial.reshape(len(comm.devs), *shape))
  return out

# ******** gated delta net: fused single token step ********

GDN_WAVES = 16
def _sigmoid(x:UOp) -> UOp: return ((x * -LOG2E).exp2() + 1).reciprocal()
def _silu(x:UOp) -> UOp: return x * _sigmoid(x)
def _softplus(x:UOp) -> UOp: return (x > 20).where(x, ((x * LOG2E).exp2() + 1).log2() * (1/LOG2E))

@functools.cache
def _gated_delta_decode_kernel(z:UOp, zq:UOp, zscale:UOp, zsum:UOp, conv_state:UOp, state:UOp, qkv:UOp, gate:UOp, beta:UOp, alpha:UOp,
                               conv_w:UOp, dt_bias:UOp, ssm_a:UOp, norm_w:UOp, start_pos:UOp, *, k_heads:int, eps:float, qk_eps:float) -> UOp:
  # one workgroup per value head: the conv of its q/k/v channels, the l2 norms, the delta rule step on its (V, K) state,
  # the gated rmsnorm and the q8 quantization of the output for ssm_out
  v_heads, dv, dk = state.shape[-3], state.shape[-2], state.shape[-1]
  C, taps = qkv.numel(), conv_w.shape[-1]
  assert dk == dv == 128 and C == 2*k_heads*dk + v_heads*dv and taps == conv_state.numel() // C + 1
  conv_state, state, qkv, gate = conv_state.reshape(taps-1, C), state.reshape(v_heads, dv, dk), qkv.reshape(C), gate.reshape(v_heads*dv)
  z, conv_w, beta, alpha = z.reshape(v_heads*dv), conv_w.reshape(C, taps), beta.reshape(v_heads), alpha.reshape(v_heads)
  head, wave, lane = UOp.range(v_heads, 0, AxisType.GLOBAL), UOp.range(GDN_WAVES, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  initial = start_pos.eq(0)
  zero = UOp.const(0, dtypes.float32)
  # conv: thread (wave%4, lane) owns element e of the head's q, k and v channels
  e = (wave % 4)*32 + lane
  kh = head % k_heads
  chans = (kh*dk + e, k_heads*dk + kh*dk + e, 2*k_heads*dk + head*dv + e)
  stores:list[UOp] = []
  conv = []
  for ci, c in enumerate(chans):
    window = [initial.where(zero, conv_state[i, c].load()) for i in range(taps-1)] + [qkv[c].load().float()]
    conv.append(_silu(sum((window[i]*conv_w[c, i].load().float() for i in range(1, taps)), window[0]*conv_w[c, 0].load().float())))
    # the next state drops the oldest tap. the q/k channels are shared by the value heads of a key head: the first one writes them
    owner = (wave < 4) if ci == 2 else (wave < 4) & (head < k_heads)
    stores += [conv_state[i, c.valid(owner)].store(window[i+1]) for i in range(taps-1)]
  # l2 norms of q and k across the four waves
  def head_sum(val:UOp, slot:int) -> UOp:
    lds = UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds = lds.after(lds[wave.valid(lane.eq(0) & (wave < 4))].store(warp_reduce(val, full_wave=True)))
    return lds[0].load() + lds[1].load() + lds[2].load() + lds[3].load()
  qn = conv[0] * (head_sum(conv[0]*conv[0], 0).sqrt().maximum(qk_eps).reciprocal() * dk**-0.5)
  kn = conv[1] * head_sum(conv[1]*conv[1], 1).sqrt().maximum(qk_eps).reciprocal()
  vec = UOp.alloc((3, 128), dtypes.float32, addrspace=AddrSpace.LOCAL)
  vec = vec.after(*(vec[i, e.valid(wave < 4)].store(x) for i,x in enumerate((qn, kn, conv[2]))))
  b = _sigmoid(beta[head].load().float())
  a = (_softplus(alpha[head].load().float() + dt_bias[head].load().float()) * ssm_a[head].load().float() * LOG2E).exp2()
  # delta rule: lanes hold 4 key columns, every wave walks its 16 rows
  cols = tuple(lane + i*32 for i in range(4))
  ks, qs = tuple(vec[1, c].load() for c in cols), tuple(vec[0, c].load() for c in cols)
  outs = UOp.alloc((128,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  out_stores = []
  for r in range(dv // GDN_WAVES):
    row = wave*(dv // GDN_WAVES) + r
    decayed = tuple(initial.where(zero, state[head, row, c].load()) * a for c in cols)
    kv = warp_reduce(sum((x*y for x,y in zip(decayed[1:], ks[1:])), decayed[0]*ks[0]), full_wave=True)
    delta = (vec[2, row].load() - kv) * b
    new = tuple(x + delta*y for x,y in zip(decayed, ks))
    stores += [state[head, row, c].store(x) for x,c in zip(new, cols)]
    out_stores.append(outs[row.valid(lane.eq(0))].store(warp_reduce(sum((x*y for x,y in zip(new[1:], qs[1:])), new[0]*qs[0]), full_wave=True)))
  outs = outs.after(*out_stores)
  # gated rmsnorm over the head, then quantize: wave w < 4 owns q8 group w of the head
  o = outs[e].load()
  rinv = (head_sum(o*o, 2) * (1/dv) + eps).sqrt().reciprocal()
  zv = (o * rinv * norm_w[e].load().float() * _silu(gate[head*dv + e].load().float())).cast(z.dtype)
  stores.append(z[(head*dv + e).valid(wave < 4)].store(zv))
  group = (head*(dv//32) + wave).valid(wave < 4)
  stores += _q8_pack(zv.float(), lane, zq, zscale, zsum, UOp.const(0, dtypes.int), group)
  return UOp.group(*stores).end(head, wave, lane).sink(arg=KernelInfo(name="gated_delta_decode", opts_to_apply=()))

def gated_delta_decode(qkv:Tensor, gate:Tensor, beta:Tensor, alpha:Tensor, conv_state:Tensor, state:Tensor, conv_w:Tensor, dt_bias:Tensor,
                       ssm_a:Tensor, norm:nn.RMSNorm, start_pos:Tensor, k_heads:int, qk_eps:float, dtype=dtypes.half) -> Tensor:
  """one token step of the gated delta net, updates conv_state and state in place. returns the gated, normed output (B, T, H*V),
  with its q8 quantization registered for ssm_out"""
  v_heads, dv = state.shape[1], state.shape[2]
  assert start_pos.uop.is_bound_var and norm.weight is not None
  state = Tensor(state.uop.after(start_pos.uop))
  z = Tensor.empty(1, 1, v_heads*dv, dtype=dtype, device=qkv.device, axis=None if qkv.uop.axis is None else 2)
  groups = v_heads*dv//Q8_GROUP_SIZE
  zq = Tensor.empty(1, groups, 8, dtype=dtypes.uint32, device=qkv.device, axis=None if qkv.uop.axis is None else 1)
  zscale = Tensor.empty(1, groups, dtype=dtypes.float32, device=qkv.device, axis=None if qkv.uop.axis is None else 1)
  zsum = Tensor.empty(1, groups, 2, dtype=dtypes.float32, device=qkv.device, axis=None if qkv.uop.axis is None else 1)
  srcs = (z, zq, zscale, zsum, conv_state, state, qkv.contiguous(), gate.contiguous(), beta.contiguous(), alpha.contiguous(), conv_w,
          dt_bias, ssm_a, norm.weight)
  contig = tuple(x.uop if x.uop.op is Ops.AFTER or x.uop.is_realized else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  fxn = functools.partial(_gated_delta_decode_kernel, k_heads=k_heads, eps=norm.eps, qk_eps=qk_eps)
  call = fxn(*params, start_pos.uop.unbound()).call(*contig)
  z, zq, zscale, zsum = (Tensor(u.after(call)) for u in contig[:4])
  _q8_cache[z.uop] = (zq, zscale, zsum)
  return z

# ******** sampling: gumbel-max argmax over the (vocab sharded) logits ********

SAMPLE_WAVES, SAMPLE_GROUPS = 8, 16

def _argmax_pair(v:UOp, i:UOp, ov:UOp, oi:UOp) -> tuple[UOp, UOp]:
  # the larger value wins, ties go to the smaller index (like argmax)
  take = (ov > v) | (ov.eq(v) & (oi < i))
  return take.where(ov, v), take.where(oi, i)

def _hash_uniform(x:UOp) -> UOp:
  # a uniform in (0, 1) from a 32-bit hash (lowbias32)
  for shift, mul in ((16, 0x7feb352d), (15, 0x846ca68b)): x = (x ^ (x >> shift)) * mul
  x = x ^ (x >> 16)
  # 23 bits: (k + 0.5) / 2^23 is exact in float32, so u never rounds to 0 or 1 (log(-log(u)) stays finite)
  return ((x >> 9).cast(dtypes.float32) + 0.5) * (1/(1 << 23))

@functools.cache
def _sample_kernel(out:UOp, seed:UOp, logits:UOp, temp:UOp, counter:UOp|None=None, *, slots:int) -> UOp:
  # every workgroup scans its part of the (local) vocab and writes its (value, index) candidate. with a vocab split over the
  # devices, the candidate goes into the slot of this device and workgroup (the other slots are zero, so a sum gathers them)
  V, GS, W = logits.numel(), SAMPLE_GROUPS, SAMPLE_WAVES
  logits, T = logits.reshape(V), W*32
  wg, wave, lane = UOp.range(GS, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  tid = wave*32 + lane
  per_wg = (V + GS - 1) // GS
  it = UOp.range((per_wg + T - 1)//T, 2, AxisType.LOOP)
  idx = wg*per_wg + it*T + tid
  valid = (it*T + tid < per_wg) & (idx < V)
  me = UOp.const(0, dtypes.int) if counter is None else counter[P2P_GROUPS].load()
  # every workgroup advances its own seed (they may run at different times)
  step = seed[wg].load()
  gidx = (idx + me*V).cast(dtypes.uint32)
  u = _hash_uniform(gidx * 0x9E3779B1 ^ (step.cast(dtypes.uint32) * 0x85EBCA77 + 0x165667B1))
  # Gumbel-max trick: argmax(logits/temp - log(-log(u))) samples from softmax(logits/temp)
  inv_temp = temp[0].load().float().maximum(1e-12).reciprocal()
  g = logits[idx.valid(valid)].load().float() * inv_temp - (-(u.log2() * (1/LOG2E))).log2() * (1/LOG2E)
  g = valid.where(g, -math.inf)
  best = UOp.alloc((2,), dtypes.float32, addrspace=AddrSpace.REG)
  best = best.after(best.store(UOp.stack(UOp.const(-math.inf, dtypes.float32), UOp.const(0, dtypes.float32))))
  cur = best.after(it)
  nv, ni = _argmax_pair(cur[0].load(), cur[1].load(), g, gidx.cast(dtypes.float32))
  best = best.after(best.store(UOp.stack(nv, ni)).end(it))
  v, i = best[0].load(), best[1].load()
  for off in (16, 8, 4, 2, 1):
    sw = [UOp(Ops.CUSTOM, src=(x,), arg=(f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | off<<10}))",
                                         dtypes.float)) for x in (v, i)]
    v, i = _argmax_pair(v, i, sw[0], sw[1])
  lds = UOp.alloc((W, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(lds[wave.valid(lane.eq(0)), 0].store(v), lds[wave.valid(lane.eq(0)), 1].store(i))
  v, i = lds[0, 0].load(), lds[0, 1].load()
  for w in range(1, W): v, i = _argmax_pair(v, i, lds[w, 0].load(), lds[w, 1].load())
  out, devs = out.reshape(slots), slots // (2*GS)
  stores = [out[((d*GS + wg)*2 + k).valid(tid.eq(0))].store(me.eq(d).where(v if k == 0 else i, UOp.const(0, dtypes.float32)))
            for d in range(devs) for k in range(2)]
  stores = [seed.after(UOp.group(*stores).barrier())[wg.valid(tid.eq(0))].store(step + 1)]
  return UOp.group(*stores).end(wg, wave, lane).sink(arg=KernelInfo(name="sample_argmax", opts_to_apply=()))

@functools.cache
def _sample_seed(device:str|tuple[str, ...]) -> Tensor:
  from tinygrad.device import Buffer, MultiBuffer
  init = (Tensor._seed * 1000003 + 12345) & 0x7fffffff
  with Context(ALLOW_DEVICE_USAGE=1):
    bufs = [Buffer(d, SAMPLE_GROUPS*4).ensure_allocated() for d in (device if isinstance(device, tuple) else (device,))]
    for i,b in enumerate(bufs): b.copy_from(Buffer("PYTHON", b.nbytes, initial_value=b"".join(((init+i*7919+w) & 0x7fffffff).to_bytes(4, "little")
                                                                                          for w in range(SAMPLE_GROUPS))))
  if isinstance(device, str): return Tensor(UOp.from_buffer(bufs[0], dtypes.int32))
  mb = MultiBuffer.__new__(MultiBuffer)
  mb.bufs = bufs
  return Tensor(UOp.from_buffer(mb, dtypes.int32, device))

def amd_sample(logits:Tensor, temperature:Tensor, device:str|tuple[str, ...]) -> Tensor|None:
  """argmax(logits/temp - log(-log(uniform))) as a (1, 1) int32 on `device`. logits are (1, V), maybe sharded on the vocab"""
  ldev = logits.device
  sharded = isinstance(ldev, tuple) and logits.uop.axis == 1
  if isinstance(ldev, tuple) != isinstance(device, tuple) or (isinstance(ldev, tuple) and not sharded): return None
  comm = p2p_comm(ldev) if isinstance(ldev, tuple) and sharded else None
  if sharded and comm is None: return None
  N = len(ldev) if isinstance(ldev, tuple) and sharded else 1
  slots = 2*N*SAMPLE_GROUPS
  out = Tensor.empty(slots, dtype=dtypes.float32, device=ldev) if comm is None else Tensor.empty(N, slots, dtype=dtypes.float32, device=ldev, axis=0)
  srcs = (out, _sample_seed(ldev), logits.contiguous(), temperature.to(ldev).contiguous()) + ((comm.counter,) if comm is not None else ())
  pairs = Tensor.custom_kernel(*srcs, fxn=functools.partial(_sample_kernel, slots=slots))[0]
  if comm is not None: pairs = p2p_allreduce(comm, pairs, (1, slots), exact=True)
  pairs = pairs.reshape(N*SAMPLE_GROUPS, 2)
  return pairs[pairs[:, 0].argmax(), 1].reshape(1, 1).cast(dtypes.int32)

# ******** swiglu + q8 quantize for the down projection ********

@functools.cache
def _swiglu_q8_kernel(act:UOp, q:UOp, scale:UOp, xsum:UOp, gate:UOp, up:UOp) -> UOp:
  features = act.shape[-1]
  tokens, groups = act.numel() // features, features // Q8_GROUP_SIZE
  act, gate, up = (x.reshape(tokens, groups, 32) for x in (act, gate, up))
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  value = (_silu(gate[token, group, lane].load().float()) * up[token, group, lane].load().float()).cast(act.dtype)
  stores = (act[token, group, lane].store(value),) + _q8_pack(value.float(), lane, q, scale, xsum, token, group)
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="swiglu_q8", opts_to_apply=()))

def swiglu_q8(gate:Tensor, up:Tensor) -> Tensor:
  """silu(gate) * up, with its q8 quantization registered for the linear that consumes it"""
  assert gate.shape == up.shape and gate.uop.axis == up.uop.axis
  features, axis = gate.shape[-1], gate.uop.axis
  tokens = prod(gate.shape[:-1])
  assert isinstance(tokens, int) and isinstance(features, int) and axis in (None, len(gate.shape)-1)
  qaxis = None if axis is None else 1
  act = Tensor.empty(*gate.shape, dtype=dtypes.float32, device=gate.device, axis=axis)
  q = Tensor.empty(tokens, features//Q8_GROUP_SIZE, 8, dtype=dtypes.uint32, device=gate.device, axis=qaxis)
  scale = Tensor.empty(tokens, features//Q8_GROUP_SIZE, dtype=dtypes.float32, device=gate.device, axis=qaxis)
  xsum = Tensor.empty(tokens, features//Q8_GROUP_SIZE, 2, dtype=dtypes.float32, device=gate.device, axis=qaxis)
  act, q, scale, xsum = Tensor.custom_kernel(act, q, scale, xsum, gate.contiguous(), up.contiguous(), fxn=_swiglu_q8_kernel)[:4]
  _q8_cache[act.uop] = (q, scale, xsum)
  return act

# ******** embedding: copy the packed rows of the tokens, dequantize only those ********

@functools.cache
def _embedding_rows_kernel(out:UOp, raw:UOp, tokens:UOp, counter:UOp|None=None, *, rows:int) -> UOp:
  # with the vocab split over the devices, a device copies the row if it holds it, else zeros (the rows sum to the embedding)
  T, words = tokens.numel(), out.numel() // tokens.numel()
  out, raw, tokens = out.flatten(), raw.flatten(), tokens.reshape(T)
  tj = UOp.range(T*words, 0, AxisType.GLOBAL)
  me = UOp.const(0, dtypes.int) if counter is None else counter[P2P_GROUPS].load()
  row = tokens[tj // words].load().cast(dtypes.int) - me*rows
  mine = (row >= 0) & (row < rows)
  return out[tj].store(mine.where(raw[(row*words + tj % words).valid(mine)].load(), UOp.const(0, raw.dtype))).end(tj).sink(
    arg=KernelInfo(name="embedding_rows", opts_to_apply=()))

def amd_embedding(embedding:nn.Embedding, tokens:Tensor) -> tuple[Tensor, bool]|None:
  """the embedding of tokens (B, T) without decoding the table. with a vocab split over the devices, returns the per device
  partials (devices, B*T, dim) to be summed and True"""
  V, D = cast(tuple[int, int], embedding.weight.shape)
  if not hasattr(embedding, "_packed"):
    lin = Linear(D, V, bias=False)
    with Context(ALLOW_DEVICE_USAGE=1): lin.set_quantized(embedding.weight)
    embedding._packed = None if lin.ggml_type is None else lin  # type: ignore[attr-defined]
  if (lin:=getattr(embedding, "_packed")) is None or not isinstance(T:=prod(tokens.shape), int): return None
  wdev, typ = lin.weight.device, cast(int, lin.ggml_type)
  sharded = isinstance(wdev, tuple) and lin.shard_axis == 0
  comm = p2p_comm(wdev) if isinstance(wdev, tuple) and sharded else None
  if (sharded and comm is None) or (isinstance(wdev, tuple) and not sharded): return None
  N = len(wdev) if isinstance(wdev, tuple) else 1
  words = D//256 * QUANT_SIZES[typ] // lin.weight.dtype.itemsize
  out = Tensor.empty(N*T, words, dtype=lin.weight.dtype, device=wdev, axis=0 if sharded else None)
  srcs = (out, lin.weight, tokens.to(wdev).contiguous()) + ((comm.counter,) if comm is not None else ())
  rows = Tensor.custom_kernel(*srcs, fxn=functools.partial(_embedding_rows_kernel, rows=V//N))[0]
  decoded = ggml_data_to_tensor(rows.flatten().bitcast(dtypes.uint8), N*T*D, typ).reshape(N, T, D) if sharded else \
    ggml_data_to_tensor(rows.flatten().bitcast(dtypes.uint8), T*D, typ).reshape(T, D)
  return decoded.cast(embedding.weight.dtype), sharded

# ******** decode attention: the q/k norm, rope and kv cache write, and the gated output with its q8 quantization ********

@functools.cache
def _attn_prep_kernel(out:UOp, x:UOp, w:UOp, freqs:UOp, start_pos:UOp, *rest:UOp, eps:float, stride:int, kv:bool) -> UOp:
  # a workgroup per head, a lane per head dim: rmsnorm, then rope on the first rope_dim dims (pairs j and j+rope_dim/2).
  # queries (rows `stride` heads apart in x) go to out (H, D) half, keys and values (rest[0]) to the cache (out) at start_pos.
  # NOTE: two kernels instead of one with gated stores: with one kv head on the device the gates of the cache writes were lost
  D, R = w.shape[0], freqs.shape[-1]
  H = x.numel() // (D*stride)
  assert D % 32 == 0 and D <= 1024 and R % 2 == 0 and R <= D
  x, freqs = x.reshape(H*stride, D), freqs.reshape(-1, R)
  h, wave, lane = UOp.range(H, 0, AxisType.GLOBAL), UOp.range(D//32, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  d = wave*32 + lane
  val = x[h*stride, d].load().float()
  lds = UOp.alloc((D//32,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(lds[wave.valid(lane.eq(0))].store(warp_reduce(val*val, full_wave=True)))
  ss = sum((lds[i].load() for i in range(1, D//32)), lds[0].load())
  xn = val * ((ss * (1/D)) + eps).sqrt().reciprocal() * w[d].load().float()
  vals = UOp.alloc((D,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  vals = vals.after(vals[d].store(xn))
  j = d % (R//2)
  cos, sin = freqs[start_pos, j.valid(d < R)].load(), freqs[start_pos, (j + R//2).valid(d < R)].load()
  partner = (d < R//2).where(vals[(d + R//2).minimum(D-1)].load(), vals[(d - R//2).maximum(0)].load())
  roped = ((d < R//2).where(xn*cos - partner*sin, xn*cos + partner*sin))
  res = (d < R).where(roped, xn).cast(dtypes.half)
  if not kv: stores:tuple[UOp, ...] = (out.reshape(H, D)[h, d].store(res),)
  else:
    v = rest[0].reshape(H, D)
    stores = (out[0, 0, h, start_pos, d].store(res), out[1, 0, h, start_pos, d].store(v[h, d].load().cast(dtypes.half)))
  return UOp.group(*stores).end(h, wave, lane).sink(arg=KernelInfo(name="attn_prep_" + ("kv" if kv else "q"), opts_to_apply=()))

def flash_decode_supported(D:int, N:int, group:int) -> bool:
  return D % 32 == 0 and D & (D-1) == 0 and N % 64 == 0 and group*((D+LDS_PAD)*2+8) <= 65536

def attn_decode_prep(q:Tensor, k:Tensor, v:Tensor, q_norm:nn.RMSNorm, k_norm:nn.RMSNorm, freqs:Tensor, cache_kv:Tensor, start_pos:UOp,
                     n_heads:int, gated:bool) -> tuple[Tensor, Tensor]:
  """one token: rmsnorm and rope q and k, write k and v to the cache at start_pos. returns (q (B, H, 1, D) half, the cache after the write)"""
  assert q_norm.weight is not None and k_norm.weight is not None and start_pos.is_bound_var and cache_kv.shape[1] == 1
  D = cache_kv.shape[-1]
  qr = Tensor.empty(1, n_heads, 1, D, dtype=dtypes.half, device=q.device, axis=None if q.uop.axis is None else 1)
  def run(srcs:tuple[Tensor, ...], **kw) -> UOp:
    contig = tuple(x.uop if x.uop.op is Ops.AFTER or x.uop.is_realized else x.uop.contiguous() for x in srcs)
    params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
    p = params[:4] + (start_pos.unbound(),) + params[4:]
    return contig[0].after(_attn_prep_kernel(*p, eps=q_norm.eps, **kw).call(*contig))
  qr_out = run((qr, q, q_norm.weight, freqs), stride=2 if gated else 1, kv=False)
  cache = run((Tensor(cache_kv.uop.after(start_pos)), k, k_norm.weight, freqs, v), stride=1, kv=True)
  return Tensor(qr_out), Tensor(cache)
