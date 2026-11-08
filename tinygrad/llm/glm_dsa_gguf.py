import weakref, functools, io, itertools, pathlib, re, struct
from typing import Any, Callable, cast

from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp, Ops
from tinygrad.device import Device
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import prod, round_up, getenv, Context, unwrap, dedup
from tinygrad.nn.state import TensorIO

DEBUG_LOAD = getenv("DEBUG_LOAD", 1)

# ggml packs each iq grid entry as N bytes (N=4 for uint32 grids, N=8 for uint64 grids) in a single word. See ggml-common.h.
def _ggml_iq_grid(device: str|tuple[str, ...]|None, grid: tuple[int, ...], grid_shape: tuple[int, int]) -> Tensor:
  # Use 32-bit words for devices without uint64; keep word columns separate to avoid doubling the constant lookup depth.
  words = [Tensor.const(tuple((v >> shift) & 0xffffffff for v in grid), dtypes.uint32) for shift in range(0, 8*grid_shape[1], 32)]
  return Tensor.stack(*words, dim=1).to(device, force=True).bitcast(dtypes.uint8).float().reshape(grid_shape)

def _ggml_iq_signs(device: str|tuple[str, ...]|None) -> Tensor:
  return Tensor.const(tuple(i | (0x80 if i.bit_count() % 2 else 0) for i in range(128)), dtypes.uint8).to(device, force=True)

# native types {ggml_type: dtype}
_GGML_NATIVE = {0: dtypes.float32, 1: dtypes.float16, 24: dtypes.int8, 25: dtypes.int16,
                26: dtypes.int32, 27: dtypes.int64, 28: dtypes.float64, 30: dtypes.bfloat16}

# quant types {ggml_type: (number of elements, number of bytes)}
_GGML_QUANT = {2:(32,18), 3:(32,20), 6:(32,22), 7:(32,24), 8:(32,34),
               10:(256,84), 11:(256,110), 12:(256,144), 13:(256,176), 14:(256,210),
               16:(256,66), 17:(256,74), 18:(256,98), 19:(256,50), 20:(32,18), 21:(256,110), 22:(256,82), 23:(256,136),
               29:(256,56), 39:(32,17), 41:(128,18)}

# the dequantized weights the loader made: the expression (its storage views unwrapped) -> (ggml type, the packed bytes). the quantized
# kernels use the packed bytes (find_packed) without making the expression again to match it
decoded_weights:weakref.WeakKeyDictionary[UOp, tuple[int, UOp]] = weakref.WeakKeyDictionary()
def _remember_decoded(decoded:Tensor, typ:int, packed:UOp):
  from tinygrad.llm.kernels.amd_glm_dsa import unwrap_storage_views
  decoded_weights[unwrap_storage_views(decoded.uop)] = (typ, packed)

def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int) -> Tensor:
  """
  Converts ggml tensor data to a tinygrad tensor.

  Supported native types: float32 (id: 0), float16 (id: 1), int8 (id: 24),
  int16 (id: 25), int32 (id: 26), int64 (id: 27), float64 (id: 28), bfloat16 (id: 30)
  Supported quantized types: Q4_0 (id: 2), Q4_1 (id: 3), Q5_0 (id: 6),
  Q5_1 (id: 7), Q8_0 (id: 8), Q2_K (id: 10), Q3_K (id: 11), Q4_K (id: 12), Q5_K (id: 13),
  Q6_K (id: 14), IQ2_XXS (id: 16), IQ2_XS (id: 17), IQ3_XXS (id: 18), IQ1_S (id: 19),
  IQ4_NL (id: 20), IQ3_S (id: 21), IQ2_S (id: 22), IQ4_XS (id: 23), IQ1_M (id: 29), MXFP4 (id: 39), Q1_0 (id: 41)
  """
  # https://github.com/ggerganov/ggml/blob/323951f1bdcdfbd5b5ff3a9a7c3770e63b1a560e/include/ggml.h#L356

  if (dtype := _GGML_NATIVE.get(ggml_type)) is not None:
    return t[:dtype.itemsize * n].contiguous().bitcast(dtype)

  def q_to_uint8(t: Tensor, b: int) -> Tensor:
    # TODO: rewrite with arange?
    shift_tensor, bitmask = Tensor.const(tuple(2**(i*b) for i in range(8//b)), t.dtype), 0xff >> (8 - b)
    return t.unsqueeze(-1).div(shift_tensor, rounding_mode="trunc").bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

  if (nelements_nbytes := _GGML_QUANT.get(ggml_type)) is not None:
    from tinygrad.runtime.autogen import ggml_common as _ggml
    blocks = t[:(n//nelements_nbytes[0])*nelements_nbytes[1]].reshape((-1, nelements_nbytes[1])).contiguous()
    if ggml_type == 2: return (q_to_uint8(blocks[:,2:], 4).bitcast(dtypes.int8) - 8) * blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
    if ggml_type == 3:
      d, m = (blocks[:,s:s+2].bitcast(dtypes.float16).cast(dtypes.float32) for s in [ 0, 2 ])
      return q_to_uint8(blocks[:,4:], 4).bitcast(dtypes.int8) * d + m
    if ggml_type in (6, 7):
      d = blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
      qh_off = 2 if ggml_type == 6 else 4
      qh = q_to_uint8(blocks[:,qh_off:qh_off+4], 1).reshape((-1, 8, 4)).transpose(-1, -2).flatten(-2).bitcast(dtypes.int8)
      q = q_to_uint8(blocks[:,qh_off+4:], 4).bitcast(dtypes.int8) + qh * 16
      return q * d + (blocks[:,2:4].bitcast(dtypes.float16).cast(dtypes.float32) if ggml_type == 7 else -16 * d)
    if ggml_type == 8: return blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32) * blocks[:,2:].bitcast(dtypes.int8)
    # Q2_K: 256 elements per 84-byte block (scales:16, qs:64, d:2, dmin:2)
    if ggml_type == 10:
      d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [80, 82])
      sc = blocks[:, :16]
      q = q_to_uint8(blocks[:, 16:80].reshape((-1, 2, 32)), 2).reshape((-1, 16, 16))
      return (d * sc.bitwise_and(0xF).unsqueeze(-1) * q - dmin * sc.rshift(4).unsqueeze(-1)).flatten(-2)
    # Q3_K: 256 elements per 110-byte block (hmask:32, qs:64, scales:12, d:2)
    if ggml_type == 11:
      d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1)
      sc = q_to_uint8(blocks[:,96:104], 4).bitwise_or(q_to_uint8(blocks[:,104:108], 2).lshift(4)).bitcast(dtypes.int8) - 32
      q = q_to_uint8(blocks[:,32:96].reshape((-1, 2, 32)), 2).reshape((-1, 16, 16))
      qh = q_to_uint8(blocks[:,:32], 1).reshape((-1, 16, 16))
      return (d * sc.unsqueeze(-1) * (q.bitcast(dtypes.int8) - qh.bitwise_xor(1).lshift(2).bitcast(dtypes.int8))).flatten(-2)
     # Q4_K: 256 elements per 144-byte block (d:2, dmin:2, scales:12, qs:128)
     # Q5_K: 256 elements per 176-byte block (d:2, dmin:2, scales:12, qh:32, qs:128)
    if ggml_type in (12, 13):
      d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [0, 2])
      s = blocks[:,4:16]  # 12 bytes: 6-bit scales[0-3], 6-bit mins[0-3], high bits[4-7]
      sc = s[:,0:4].bitwise_and(63).cat(s[:,8:12].bitwise_and(0xF).bitwise_or(s[:,0:4].rshift(6).lshift(4)), dim=-1)
      mn = s[:,4:8].bitwise_and(63).cat(s[:,8:12].rshift(4).bitwise_or(s[:,4:8].rshift(6).lshift(4)), dim=-1)
      qs_off = 48 if ggml_type == 13 else 16
      q = Tensor.stack((qs:=blocks[:,qs_off:qs_off+128].reshape(-1,4,32)).bitwise_and(0xF), qs.rshift(4), dim=2).reshape(-1,8,32)
      if ggml_type == 13: q = q + q_to_uint8(blocks[:,16:48], 1).reshape(-1, 8, 32) * 16
      return (d * sc.unsqueeze(-1) * q - dmin * mn.unsqueeze(-1)).flatten(-2)
    if ggml_type == 14:
      xl, xh = q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
      scales = blocks[:,192:208].bitcast(dtypes.int8).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
      d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32)
      return d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales
    if ggml_type == 18:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scale_words = blocks[:, 66:98].bitcast(dtypes.uint32)
      db = d * (scale_words.rshift(28).cast(dtypes.float32) + 0.5).reshape((-1, 8, 1, 1)) * 0.5
      sign_idx = scale_words.unsqueeze(-1).rshift(Tensor.const((0, 7, 14, 21), dtypes.uint32)).bitwise_and(0x7F).reshape((-1, 32)).cast(dtypes.int32)
      signs = (q_to_uint8(_ggml_iq_signs(t.device)[sign_idx].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      grid = _ggml_iq_grid(t.device, _ggml.iq3xxs_grid, (256, 4))[blocks[:, 2:66]].reshape((-1, 8, 4, 8))
      return (db * grid * signs).flatten(-3)
    # IQ2_XXS: 256 elements per 66-byte block (d:2, qs:64). 8 groups of 32: 4 grid bytes + packed signs/scale.
    if ggml_type == 16:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      qs_u32 = blocks[:, 2:].bitcast(dtypes.uint32).reshape((-1, 8, 2))
      db = d * (qs_u32[:, :, 1].rshift(28).cast(dtypes.float32) + 0.5).reshape((-1, 8, 1, 1)) * 0.25
      sign_idx = qs_u32[:, :, 1].unsqueeze(-1).rshift(Tensor.const((0, 7, 14, 21), dtypes.uint32))
      sign_idx = sign_idx.bitwise_and(0x7F).reshape((-1, 32)).cast(dtypes.int32)
      signs = (q_to_uint8(_ggml_iq_signs(t.device)[sign_idx].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      grid = _ggml_iq_grid(t.device, _ggml.iq2xxs_grid, (256, 8))[blocks[:, 2:].reshape((-1, 8, 8))[:, :, :4]].reshape((-1, 8, 4, 8))
      return (db * grid * signs).flatten(-3)
    # IQ2_XS: 256 elements per 74-byte block (d:2, qs:64 as uint16, scales:8)
    if ggml_type == 17:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      db = d * (q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 4).reshape((-1, 16)).cast(dtypes.float32) + 0.5).reshape((-1, 16, 1, 1)) * 0.25
      qs = blocks[:, 2:66].bitcast(dtypes.uint16)
      sign_idx = qs.rshift(9).cast(dtypes.int32)
      signs = (q_to_uint8(_ggml_iq_signs(t.device)[sign_idx].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 16, 2, 8))
      grid = _ggml_iq_grid(t.device, _ggml.iq2xs_grid, (512, 8))[qs.bitwise_and(511)].reshape((-1, 16, 2, 8))
      return (db * grid * signs).flatten(-3)
    # IQ1_S: 256 elements per 50-byte block (d:2, qs:32, qh:16). grid bytes are int8 {-1,0,1}.
    if ggml_type == 19:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      qh = blocks[:, 34:50].bitcast(dtypes.uint16)
      dl = d * (qh.rshift(12).bitwise_and(7).cast(dtypes.float32) * 2 + 1).reshape((-1, 8, 1, 1))
      delta = (qh.bitwise_and(0x8000) == 0).where(0.125, -0.125).reshape((-1, 8, 1, 1))
      qh_hi = qh.unsqueeze(-1).rshift(Tensor.const((0, 3, 6, 9), dtypes.uint16)).bitwise_and(7).lshift(8)
      q = blocks[:, 2:34].cast(dtypes.uint16) + qh_hi.reshape((-1, 32))
      grid = _ggml_iq_grid(t.device, _ggml.iq1s_grid, (2048, 8))[q].reshape((-1, 8, 4, 8))
      grid = (grid > 127).where(grid - 256, grid)
      return (dl * (grid + delta)).flatten(-3)
    if ggml_type == 20:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32)
      return d * Tensor.const(tuple(_ggml.kvalues_iq4nl), dtypes.float32)[q_to_uint8(blocks[:, 2:], 4)]
    if ggml_type == 21:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      scales = (1 + 2 * q_to_uint8(blocks[:, 106:110].reshape((-1, 4, 1)), 4).reshape((-1, 8))).cast(dtypes.float32).reshape((-1, 8, 1, 1))
      qh = q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 1).reshape((-1, 64)).cast(dtypes.uint16)
      signs = (q_to_uint8(blocks[:, 74:106].reshape((-1, 32, 1)), 1).reshape((-1, 256)) == 0).where(1.0, -1.0).reshape((-1, 8, 4, 8))
      q = blocks[:, 2:66].cast(dtypes.uint16) + qh.lshift(8)
      return (d * scales * _ggml_iq_grid(t.device, _ggml.iq3s_grid, (512, 4))[q].reshape((-1, 8, 4, 8)) * signs).flatten(-3)
    if ggml_type == 22:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1))
      db = d * (q_to_uint8(blocks[:, 74:82].reshape((-1, 8, 1)), 4).reshape((-1, 16)).cast(dtypes.float32) + 0.5).reshape((-1, 16, 1, 1)) * 0.25
      signs = (q_to_uint8(blocks[:, 34:66].reshape((-1, 32, 1)), 1) == 0).where(1.0, -1.0).reshape((-1, 16, 2, 8))
      qh = q_to_uint8(blocks[:, 66:74].reshape((-1, 8, 1)), 2).reshape((-1, 32)).cast(dtypes.uint16)
      q = blocks[:, 2:34].cast(dtypes.uint16) + qh.lshift(8)
      return (db * _ggml_iq_grid(t.device, _ggml.iq2s_grid, (1024, 8))[q].reshape((-1, 16, 2, 8)) * signs).flatten(-3)
    if ggml_type == 23:
      d = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1))
      scale_shifts = Tensor.const((0, 2, 4, 6, 8, 10, 12, 14), dtypes.uint16)
      iq4_xs_lut = Tensor.const(tuple(_ggml.kvalues_iq4nl), dtypes.float32).to(t.device, force=True)
      scales_l = Tensor.stack((sl:=blocks[:, 4:8]).bitwise_and(0xF), sl.rshift(4), dim=2).reshape((-1, 8))
      scales_h = blocks[:, 2:4].bitcast(dtypes.uint16).unsqueeze(-1).rshift(scale_shifts).bitwise_and(0x03).reshape((-1, 8)).cast(dtypes.uint8)
      scales = (scales_l.bitwise_or(scales_h.lshift(4)).bitcast(dtypes.int8) - 32).cast(dtypes.float32).reshape((-1, 8, 1))
      q = (qs:=blocks[:, 8:].reshape((-1, 8, 16))).bitwise_and(0xF).cat(qs.rshift(4), dim=2)
      return (d * scales * iq4_xs_lut[q]).flatten(-2)
    # IQ1_M: 256 elements per 56-byte block (qs:32, qh:16, scales:8). f16 scale packed in high nibbles.
    if ggml_type == 29:
      sc16 = blocks[:, 48:56].bitcast(dtypes.uint16)
      d = sc16.bitwise_and(0xF000).rshift(Tensor.const((12, 8, 4, 0), dtypes.uint16))
      d = d[:, 0:1].bitwise_or(d[:, 1:2]).bitwise_or(d[:, 2:3]).bitwise_or(d[:, 3:4])
      d = d.bitcast(dtypes.float16).cast(dtypes.float32).reshape((-1, 1, 1, 1, 1))
      scales = sc16.unsqueeze(-1).rshift(Tensor.const((0, 3, 6, 9), dtypes.uint16)).bitwise_and(7)
      dl = d * (scales.cast(dtypes.float32) * 2 + 1).reshape((-1, 8, 2, 1, 1))
      qh_n = Tensor.stack(blocks[:, 32:48].bitwise_and(0x0F), blocks[:, 32:48].rshift(4), dim=-1).reshape((-1, 32))
      q = blocks[:, :32].cast(dtypes.uint16) + qh_n.bitwise_and(7).cast(dtypes.uint16).lshift(8)
      delta = (qh_n.bitwise_and(0x08) == 0).where(0.125, -0.125).reshape((-1, 8, 2, 2, 1))
      grid = _ggml_iq_grid(t.device, _ggml.iq1s_grid, (2048, 8))[q].reshape((-1, 8, 2, 2, 8))
      grid = (grid > 127).where(grid - 256, grid)
      return (dl * (grid + delta)).flatten(-4)
    if ggml_type == 39:
      e = blocks[:, 0].cast(dtypes.uint32)
      small_bits = Tensor([0x00200000, 0x00400000], dtype=dtypes.uint32, device=t.device)[e.clip(0, 1).cast(dtypes.int32)] # e = 0 or e = 1 case
      d = (e < 2).where(small_bits, (e - 1) * 0x00800000).bitcast(dtypes.float32).unsqueeze(-1)
      codes = q_to_uint8(blocks[:, 1:17], 4)
      fp4_lut = Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                       -0.0,-1.0,-2.0,-3.0,-4.0,-6.0,-8.0,-12.0],
                      dtype=dtypes.float32, device=t.device)
      fp4_val = fp4_lut[codes]
      return (fp4_val * d).flatten(-2)[:n]
    if ggml_type == 41:
      d = blocks[:,:2].bitcast(dtypes.float16)
      bits = q_to_uint8(blocks[:,2:], 1).reshape(-1, 8, 16).transpose(-1, -2).flatten(-2).bitcast(dtypes.int8)
      return d * (bits * 2 - 1)
  raise ValueError(f"GGML type '{ggml_type}' is not supported!")

def _read_unpack(fmt: str, n: int, r:io.BufferedIOBase): return struct.unpack(fmt, r.read(n))[0]
def read_str(r:io.BufferedIOBase): return str(r.read(read_uint64(r)), "utf-8")
def read_arr(r:io.BufferedIOBase):
  item_reader, n = readers[read_int32(r)], read_uint64(r)
  return [item_reader(r) for _ in range(n)]

readers: dict[int, Callable[[io.BufferedIOBase], Any]] = { 8: read_str, 9: read_arr,
  **{ t: functools.partial(_read_unpack, "<"+f, nb) for t,f,nb in \
    [ (0,"c",1), (1,"b",1), (2,"H",2), (3,"h",2), (4,"I",4), (5,"i",4), (6,"f",4), (7,"?",1), (10,"Q",8), (11,"q",8), (12,"d",8) ] } }
read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

def gguf_parse(fn: Tensor|str|pathlib.Path) -> tuple[dict, dict[str, tuple[Tensor, tuple[int, ...], int]]]:
  """
  Parses a .gguf file, returning metadata and `(data, shape, ggml_type)` entries without decoding the weights.
  Each `data` tensor is a bounded byte view on the source device; paths yield lazy DISK views.
  Multi-part splits are auto-merged when loaded by path.
  """
  files, state_dict = [fn], {}
  for i, file in enumerate(files):
    tensor = file if isinstance(file, Tensor) else Tensor(pathlib.Path(file))
    r = io.BufferedReader(TensorIO(tensor), 1_000_000)
    magic, version, n_tensors, n_kv = r.read(4), read_int32(r), read_int64(r), read_int64(r)
    if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")

    kv = {}
    for _ in range(n_kv):
      k, typ = read_str(r), read_int32(r)
      kv[k] = readers[typ](r)
    if i == 0:
      kv_data = kv
      if (total := kv.get('split.count', 1)) > 1:
        if isinstance(fn, Tensor): raise ValueError("multi-part GGUF requires a path argument (got Tensor)")
        if kv.get('split.no', 0) != 0: raise ValueError(f"multi-part GGUF must be loaded from the first split, got split.no={kv['split.no']}")
        if not (m := re.match(r"^(.*)-00001-of-\d{5}\.gguf$", str(fn))): raise ValueError(f"first split must end with -00001-of-NNNNN.gguf: {fn}")
        files.extend(pathlib.Path(f"{m.group(1)}-{part:05d}-of-{total:05d}.gguf") for part in range(2, total+1))

    t_infos = [ (read_str(r), tuple(read_uint64(r) for _ in range(read_uint32(r))), read_int32(r), read_uint64(r)) for _ in range(n_tensors) ]
    data_start = round_up(r.tell(), kv.get("general.alignment", 32))
    for name, dims, typ, off in t_infos:
      if typ in _GGML_NATIVE: block, size = 1, _GGML_NATIVE[typ].itemsize
      elif typ in _GGML_QUANT: block, size = _GGML_QUANT[typ]
      else: raise ValueError(f"GGML type '{typ}' is not supported!")
      start, nbytes = data_start + off, prod(dims)//block*size
      state_dict[name] = (tensor[start:start+nbytes], tuple(reversed(dims)), typ)
  return kv_data, state_dict

def gguf_load(fn: Tensor|str|pathlib.Path) -> tuple[dict, dict[str, Tensor]]:
  """
  Loads a .gguf file, returning the `kv_data` and `state_dict`. Multi-part splits are auto-merged when loaded by path.

  ```python
  import pathlib
  from tinygrad import Device, Tensor
  from tinygrad.llm.gguf import gguf_load

  gguf_tensor = Tensor(pathlib.Path("Meta-Llama-3-8B-Instruct.Q4_0.gguf")).to(Device.DEFAULT)
  kv_data, state_dict = gguf_load(gguf_tensor)
  ```

  Packed weights are copied to the default device before constructing the lazy decoding expressions.
  """
  kv, entries = gguf_parse(fn)
  return kv, gguf_shard(entries, (Device.DEFAULT,))

def gguf_pad(entry:tuple[Tensor, tuple[int, ...], int], axis:int, size:int) -> tuple[Tensor, tuple[int, ...], int]:
  """
  Zero pads a parsed `(data, shape, ggml_type)` entry on `axis` to `size`, in its packed form. Zero bytes decode to zero weights.
  """
  data, shape, typ = entry
  if shape[axis] == size: return entry
  block, nbytes = (1, _GGML_NATIVE[typ].itemsize) if typ in _GGML_NATIVE else _GGML_QUANT[typ]
  if axis == len(shape)-1:
    if shape[axis] % block or size % block: raise ValueError(f"can't pad {shape} on axis {axis} to {size} in blocks of {block}")
    data = data.to("CPU").reshape(prod(shape[:-1]), shape[-1]//block*nbytes).pad(((0, 0), (0, (size-shape[-1])//block*nbytes)))
  else: data = data.to("CPU").reshape(prod(shape[:axis]), shape[axis], -1).pad(((0, 0), (0, size-shape[axis]), (0, 0)))
  return data.flatten(), shape[:axis] + (size,) + shape[axis+1:], typ

@functools.cache
def _file_mmap(path:str) -> memoryview:
  import mmap
  with open(path, "rb") as f: return memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))

def _disk_bytes(data:Tensor) -> memoryview|None:
  # the bytes of a parsed tensor in the mmapped file, None if it isn't a view of a gguf file
  from tinygrad.device import Buffer
  try: buf = data.uop.buffer
  except Exception: return None
  if not isinstance(buf, Buffer) or not buf.device.startswith("DISK:") or buf.device.startswith("DISK:shm:"): return None
  return _file_mmap(buf.device[5:])[buf.offset:buf.offset+buf.nbytes]

def _strided_copy(dst:memoryview, src:memoryview, offset:int, count:int, size:int, stride:int, dst_stride:int|None=None):
  # count pieces of `size` bytes of src, from `offset` every `stride` bytes, to dst every dst_stride (default size) bytes: strided copies of words
  dst_stride = size if dst_stride is None else dst_stride
  if size == 0 or count == 0: return
  item = next(i for i in (8, 4, 2, 1) if size % i == 0 and stride % i == 0 and offset % i == 0 and dst_stride % i == 0)
  # whichever is fewer python steps: a copy per piece, or a strided copy per word
  if count <= size // item:
    for c in range(count): dst[c*dst_stride:c*dst_stride+size] = src[offset+c*stride:offset+c*stride+size]
    return
  fmt = cast(Any, {8: "Q", 4: "I", 2: "H", 1: "B"}[item])
  s, d, words, dstep = src.cast(fmt), dst.cast(fmt), size // item, dst_stride // item
  base, step = offset // item, stride // item
  for w in range(words): d[w:w+(count-1)*dstep+1:dstep] = s[base+w:base+w+(count-1)*step+1:step]

def _direct_load(entries:dict[str, tuple[Tensor, tuple[int, ...], int]], devices:tuple[str, ...], shard_map:dict[str, int],
                 units:dict[str, tuple[int, list[list[int]]]], pads:dict[str, int], fuse:dict[str, str]) -> dict[str, Tensor]|None:
  """
  loads every tensor's bytes straight from the mmapped gguf: one buffer per device holds all its tensors (views), filled through a host
  buffer of the device in big copies. None if a tensor isn't in a file or the devices have no host memory
  """
  from tinygrad.device import Buffer, BufferSpec, MultiBuffer
  n = len(devices)
  if not getenv("GGUF_DIRECT", 1) or not all(d.split(":")[0] == "AMD" for d in devices): return None
  srcs = {name: _disk_bytes(data) for name, (data, _, _) in entries.items()}
  if any(v is None for v in srcs.values()): return None
  # every device's part of every tensor: (nbytes, writer(dst, src))
  plan:dict[str, list[tuple[int, Callable]]] = {}
  fused_up = set(fuse.values())
  for name, (data, shape, typ) in entries.items():
    src = cast(memoryview, srcs[name])
    nb = len(src)
    if name in fused_up: continue
    if name in fuse:
      # the gate and up units in one: their rows interleaved by 16 (fuse_gate_up)
      slices, per_device = units[name]
      E, rows, _ = shape
      unit, up = nb // (E*slices), cast(memoryview, srcs[fuse[name]])
      chunk = unit // (rows // slices) * 16
      def w_fused(dst, src, us, up=up, unit=unit, chunk=chunk):
        for i,u in enumerate(us):
          for k,s in enumerate((src, up)): _strided_copy(dst[2*i*unit+k*chunk:2*(i+1)*unit], s, u*unit, unit // chunk, chunk, chunk, 2*chunk)
      plan[name] = [(2*unit*len(us), functools.partial(w_fused, us=us)) for us in per_device]
      continue
    if name in units:
      slices, per_device = units[name]
      E, rows, _ = shape
      unit = nb // (E*slices)
      def w_units(dst, src, us, by_rows=not name.endswith("down_exps.weight"), unit=unit, rows=rows, slices=slices):
        for i,u in enumerate(us):
          if by_rows: dst[i*unit:(i+1)*unit] = src[u*unit:(u+1)*unit]
          else:
            e, j = divmod(u, slices)
            _strided_copy(dst[i*unit:(i+1)*unit], src, e*unit*slices + j*(unit//rows), rows, unit//rows, unit*slices//rows)
      plan[name] = [(unit*len(us), functools.partial(w_units, us=us)) for us in per_device]
    elif (axis:=shard_map.get(name)) is None: plan[name] = [(nb, lambda dst, src: dst.__setitem__(slice(None), src))]*n
    else:
      # the bytes of an outer row (the axis and the inner dims), padded with zeros to the padded size of the axis
      outer, size = prod(shape[:axis]), pads.get(name, shape[axis])
      if size % n or (axis == len(shape)-1 and size//n % _GGML_QUANT.get(typ, (1,))[0]):
        raise ValueError(f"{name}: can't split {shape} on axis {axis} over {n} devices")
      row = nb // outer
      part = row * size // shape[axis] // n
      def w_part(dst, src, i, outer=outer, part=part, row=row):
        real = max(0, min(part, row - i*part))
        if real < part: dst[:] = bytes(len(dst))
        _strided_copy(dst, src, i*part, outer, real, row, part)
      plan[name] = [(part*outer, functools.partial(w_part, i=i)) for i in range(n)]
  # the offsets of every tensor in every device's buffer
  align, offsets, totals = 256, cast(dict[str, list[int]], {}), [0]*n
  for name, parts in plan.items():
    offsets[name] = []
    for i,(nb, _) in enumerate(parts):
      offsets[name].append(totals[i])
      totals[i] += round_up(nb, align)
  with Context(ALLOW_DEVICE_USAGE=1):
    bufs = [Buffer(d, t).ensure_allocated() for d,t in zip(devices, totals)]
    stage_size = max(1 << 28, max(nb for parts in plan.values() for nb,_ in parts))
    stages = [Buffer(d, stage_size, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated() for d in devices]
    # fill a device's host buffer with its next tensors, then one copy to the device, all the devices in turn
    for i, (d, stage) in enumerate(zip(devices, stages)):
      mv, fill, start = stage.as_memoryview(allow_zero_copy=True), 0, 0
      # the tensors are in order in the device's buffer: the host buffer holds the next ones, copied when it's full
      for name, parts in plan.items():
        nb, writer = parts[i]
        if fill + round_up(nb, align) > stage_size:
          bufs[i].view(fill, start).ensure_allocated().copy_from(stage.view(fill, 0).ensure_allocated())
          Device[d].synchronize()
          start, fill = offsets[name][i], 0
        writer(mv[fill:fill+nb], srcs[name])
        fill += round_up(nb, align)
      if fill: bufs[i].view(fill, start).ensure_allocated().copy_from(stage.view(fill, 0).ensure_allocated())
      Device[d].synchronize()
    del stages
  packed:dict[str, Tensor] = {}
  for name, parts in plan.items():
    # a realized multi device buffer of the views: copies of a replicated tensor, or the parts of a sharded one
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [bufs[i].view(nb, offsets[name][i]).ensure_allocated() for i,(nb,_) in enumerate(parts)]
    buf = UOp.from_buffer(mb, dtypes.uint8, devices)
    packed[name] = Tensor(buf if name not in units and shard_map.get(name) is None else buf.reshape(1, -1).unshard(0))
  return packed

# **** the parallel loader: file reads (preadv, no gil) into host buffers of the devices, copies to them, strided layouts gathered on them ****

def _gather_word(*nbytes:int) -> Any:
  return next(dt for dt in (dtypes.uint32, dtypes.uint16, dtypes.uint8) if all(n % dt.itemsize == 0 for n in nbytes))

def _units_gather_kernel(dst:UOp, src:UOp, units:UOp, rows:int, slices:int) -> UOp:
  # dst (units, rows, L) words: unit i (expert e, slice j) is the column block j of the rows of expert e in src (experts, rows, slices, L)
  from tinygrad.uop.ops import AxisType, KernelInfo
  U, L = units.shape[0], dst.shape[0] // (units.shape[0] * rows)
  i, r, k = UOp.range(U, 0, AxisType.GLOBAL), UOp.range(rows, 1, AxisType.GLOBAL), UOp.range(L, 2, AxisType.LOOP)
  u = units[i].load().cast(dtypes.weakint)
  return dst[(i*rows + r)*L + k].store(src[((u // slices * rows + r) * slices + u % slices) * L + k].load()).end(i, r, k).sink(
    arg=KernelInfo(name="units_gather"))

def _part_gather_kernel(dst:UOp, src:UOp, outer:int, off:int) -> UOp:
  # dst (outer, P) words: the words off.. of every row of src (outer, R), zero past the row
  from tinygrad.uop.ops import AxisType, KernelInfo
  P, R = dst.shape[0] // outer, src.shape[0] // outer
  r, k = UOp.range(outer, 0, AxisType.GLOBAL), UOp.range(P, 1, AxisType.LOOP)
  c = off + k
  return dst[r*P + k].store((c < R).where(src[r*R + c.minimum(R-1)].load(), UOp.const(0, src.dtype))).end(r, k).sink(
    arg=KernelInfo(name="part_gather"))

def _run_gather(fxn:Callable, *bufs:UOp):
  # the kernel on the buffers, a call run as is (compiled once: the kernel is the same for buffers of the same sizes, no schedule)
  from tinygrad.engine.realize import run_linear
  from tinygrad.uop.ops import Ops
  run_linear(UOp(Ops.LINEAR, src=(fxn(*[UOp.placeholder_like(b, slot=i) for i,b in enumerate(bufs)]).call(*bufs),)), update_stats=False)

def _load_plan(entries:dict[str, tuple[Tensor, tuple[int, ...], int]], devices:tuple[str, ...], shard_map:dict[str, int],
               units:dict[str, tuple[int, list[list[int]]]], pads:dict[str, int], fuse:dict[str, str],
               locs:dict[str, tuple[str, int, int]]) -> dict[str, list[tuple[int, tuple]]]:
  """every device's part of every tensor: (nbytes, fill) (see _parallel_load)"""
  n = len(devices)
  # every device's part of every tensor: (nbytes, fill). a fill is ("segs", segments, zero ranges) or ("gather", kind, args). a segment
  # (src, length, dst, piece, stride): the bytes src.. split in pieces, piece p to dst + p*stride
  plan:dict[str, list[tuple[int, tuple]]] = {}
  fused_up = set(fuse.values())
  for name, (data, shape, typ) in entries.items():
    _, off, nb = locs[name]
    if name in fused_up: continue
    if name in fuse:
      slices, per_device = units[name]
      E, rows, _ = shape
      unit, up_off = nb // (E*slices), locs[fuse[name]][1]
      chunk = unit // (rows // slices) * 16
      # a unit's gate and up rows interleaved by 16 (fuse_gate_up): the chunks of each go every 2*chunk
      plan[name] = [(2*unit*len(us), ("segs", [(base + u*unit, unit, 2*i*unit + k*chunk, chunk, 2*chunk) for i,u in enumerate(us)
                                                for k, base in enumerate((off, up_off))], [])) for us in per_device]
      continue
    if name in units:
      slices, per_device = units[name]
      E, rows, cols = shape
      unit = nb // (E*slices)
      if name.endswith("down_exps.weight"):
        plan[name] = [(unit*len(us), ("gather", "units", (E, rows, slices, unit // rows, us))) for us in per_device]
      else: plan[name] = [(unit*len(us), ("segs", [(off + u*unit, unit, i*unit, unit, 0) for i,u in enumerate(us)], [])) for us in per_device]
    elif (axis:=shard_map.get(name)) is None: plan[name] = [(nb, ("segs", [(off, nb, 0, nb, 0)], []))]*n
    else:
      outer, size = prod(shape[:axis]), pads.get(name, shape[axis])
      if size % n or (axis == len(shape)-1 and size//n % _GGML_QUANT.get(typ, (1,))[0]):
        raise ValueError(f"{name}: can't split {shape} on axis {axis} over {n} devices")
      row = nb // outer
      part = row * size // shape[axis] // n
      if outer == 1:
        plan[name] = [(part, ("segs", [(off + i*part, real, 0, real, 0)] if (real:=max(0, min(part, row - i*part))) else [],
                              [(real, part)] if real < part else [])) for i in range(n)]
      else: plan[name] = [(part*outer, ("gather", "part", (outer, row, part, i))) for i in range(n)]
  return plan

def _parallel_load(entries:dict[str, tuple[Tensor, tuple[int, ...], int]], devices:tuple[str, ...], shard_map:dict[str, int],
                   units:dict[str, tuple[int, list[list[int]]]], pads:dict[str, int], fuse:dict[str, str],
                   defer:list[Callable[[], None]]|None=None) -> dict[str, Tensor]|None:
  """
  _direct_load in parallel: every device's tensors are file reads (os.preadv, threads) into host buffers of the device, then copies to its
  buffer, the devices in turn. tensors split by columns (an expert's slices of the down projection, the parts of a row parallel weight) are
  read whole into device memory and gathered there (a kernel). the same bytes as _direct_load
  defer: the device memory is allocated (the tensors are made), the bytes are read by the function added to it
  """
  import os, time, concurrent.futures
  from tinygrad.device import Buffer, BufferSpec, MultiBuffer
  n = len(devices)
  if not getenv("GGUF_DIRECT", 1) or not all(d.split(":")[0] == "AMD" for d in devices): return None
  locs:dict[str, tuple[str, int, int]] = {}  # the file, offset and size of every tensor's bytes
  for name, (data, _, _) in entries.items():
    try: buf = data.uop.buffer
    except Exception: return None
    if not isinstance(buf, Buffer) or not buf.device.startswith("DISK:") or buf.device.startswith("DISK:shm:"): return None
    locs[name] = (buf.device[5:], buf.offset, buf.nbytes)
  plan = _load_plan(entries, devices, shard_map, units, pads, fuse, locs)
  # the offsets of every tensor in every device's buffer
  align, offsets, totals = 256, cast(dict[str, list[int]], {}), [0]*n
  for name, parts in plan.items():
    offsets[name] = []
    for i,(nb, _) in enumerate(parts):
      offsets[name].append(totals[i])
      totals[i] += round_up(nb, align)
  fds = {path: os.open(path, os.O_RDONLY) for path in dedup(path for path, _, _ in locs.values())}
  STAGE, K = getenv("LOAD_STAGE", 256 << 20), getenv("LOAD_STAGES", 3)
  # every device's jobs in order: ("copy", reads, zeros, size, target, offset) a host buffer filled by reads (fd, src, [(at, len)]) and zero
  # ranges then copied to the buffer or the temp buffer, ("gather", ...) a kernel from the temp buffer
  def device_jobs(i:int) -> list[tuple]:
    jobs:list[tuple] = []
    start, end = 0, 0
    reads:list[tuple] = []
    zeros:list[tuple[int, int]] = []
    def flush():
      nonlocal reads, zeros
      if (reads or zeros) and end > start: jobs.append(("copy", reads, zeros, end - start, "buf", start))
      reads, zeros = [], []
    def place(lo:int, hi:int):
      # a window of the buffer covering lo..hi (a new one starts at lo)
      nonlocal start, end
      if (not reads and not zeros) or lo < start or hi > start + STAGE:
        flush()
        start = end = lo
      end = max(end, hi)
    for name, parts in plan.items():
      nb, fill = parts[i]
      fd, base = fds[locs[name][0]], offsets[name][i]
      if fill[0] == "segs":
        for src, length, dst, piece, stride in fill[1]:
          if piece == length:  # contiguous: split in windows
            for o in range(0, length, STAGE):
              place(base + dst + o, base + dst + (e:=min(length, o + STAGE)))
              reads.append((fd, src + o, [(base + dst + o - start, e - o)]))
          else:
            place(base + dst, base + dst + (length // piece - 1) * stride + piece)
            reads.append((fd, src, [(base + dst + p*stride - start, piece) for p in range(length // piece)]))
        for z0, z1 in fill[2]:
          place(base + z0, base + z1)
          zeros.append((base + z0 - start, z1 - z0))
      else:
        flush()
        # the tensor (or the experts used) into the temp buffer, a host buffer at a time, then the gather to the device's part
        _, src_off, src_nb = locs[name]
        args = fill[2]
        if fill[1] == "units" and (ebytes:=src_nb // args[0]) <= STAGE:
          # only the experts of the device's units (contiguous in the file): compacted in the temp buffer, the units renumbered in them
          ex = sorted(set(u // args[2] for u in args[4]))
          pos = {e: j for j, e in enumerate(ex)}
          args = (*args[:4], [pos[u // args[2]]*args[2] + u % args[2] for u in args[4]])
          for e0 in range(0, len(ex), per:=max(1, STAGE // ebytes)):
            ereads:list[tuple] = []
            at = 0
            for j in range(e0, min(len(ex), e0 + per)):  # runs of consecutive experts in one read
              if ereads and ex[j] == ex[j-1] + 1: ereads[-1][2][0] = (ereads[-1][2][0][0], ereads[-1][2][0][1] + ebytes)
              else: ereads.append((fd, src_off + ex[j]*ebytes, [(at, ebytes)]))
              at += ebytes
            jobs.append(("copy", ereads, [], at, "tmp", e0*ebytes))
          src_nb = len(ex) * ebytes
        else:
          for o in range(0, src_nb, STAGE): jobs.append(("copy", [(fd, src_off + o, [(0, sz:=min(STAGE, src_nb - o))])], [], sz, "tmp", o))
        jobs.append(("gather", name, fill[1], args, base, nb, src_nb))
    flush()
    return jobs
  jobs = [device_jobs(i) for i in range(n)]
  tmp_size = max([j[6] for js in jobs for j in js if j[0] == "gather"], default=0)
  def fill(mv:memoryview, reads:list, zeros:list):
    for fd, src, iov in reads:
      views, total, got = [mv[at:at+ln] for at, ln in iov], sum(ln for _, ln in iov), 0
      while got < total:
        got += os.preadv(fd, views, src + got)
        if got < total:  # a short read: the rest of the views
          skip, rest = got, []
          for v in views:
            if skip >= len(v):
              skip -= len(v)
              continue
            rest.append(v[skip:])
            skip = 0
          views = rest
    for at, ln in zeros: mv[at:at+ln] = bytes(ln)
  with Context(ALLOW_DEVICE_USAGE=1): bufs = [Buffer(d, t).ensure_allocated() for d,t in zip(devices, totals)]
  def fill_all():
    with Context(ALLOW_DEVICE_USAGE=1):
      tmps = [Buffer(d, round_up(tmp_size, align)).ensure_allocated() for d in devices] if tmp_size else []
      stages = [[Buffer(d, STAGE, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated() for _ in range(K)] for d in devices]
      mvs = [[s.as_memoryview(allow_zero_copy=True) for s in ss] for ss in stages]
      pool = concurrent.futures.ThreadPoolExecutor(getenv("LOAD_THREADS", 16))
      # per device: the next job to fill, the next job to issue, the timeline (the device signals it: the value of the last work issued),
      # per host buffer: its job and fill, or the timeline value of its copy (free once reached). the temp buffer is written once the gather
      # before read it, a gather runs once the copies before it are done. nothing waits holding the gil (fills run meanwhile)
      nxt, done = [0]*n, [0]*n
      tls = [Device[d].timeline.host.view(fmt='Q') for d in devices]
      fills:list[list[Any]] = [[None]*K for _ in range(n)]
      busy:list[list[int]] = [[0]*K for _ in range(n)]
      last_copy, last_gather = [0]*n, [0]*n
      unit_tables:dict[tuple, Buffer] = {}
      def gather(i:int, job:tuple):
        _, name, kind, args, dst0, nb, src_nb = job
        word = _gather_word(nb, src_nb, *((args[3],) if kind == "units" else (args[1], args[2])))
        dst = UOp.from_buffer(bufs[i].view(nb, dst0).ensure_allocated(), word, devices[i])
        src = UOp.from_buffer(tmps[i].view(src_nb, 0).ensure_allocated(), word, devices[i])
        if kind == "units":
          E, rows, slices, blk, us = args
          if (ut:=unit_tables.get((i, tuple(us)))) is None:
            ut = unit_tables[(i, tuple(us))] = Buffer(devices[i], 4*len(us), initial_value=struct.pack(f"<{len(us)}i", *us)).ensure_allocated()
          _run_gather(functools.partial(_units_gather_kernel, rows=rows, slices=slices), dst, src, UOp.from_buffer(ut, dtypes.int32, devices[i]))
        else:
          outer, row, part, di = args
          _run_gather(functools.partial(_part_gather_kernel, outer=outer, off=di*part//word.itemsize), dst, src)
      while any(done[i] < len(jobs[i]) for i in range(n)):
        progressed = False
        for i in range(n):
          cur = tls[i][0]
          # fill the free host buffers with the next copy jobs
          for k in range(K):
            if fills[i][k] is None and cur >= busy[i][k] and nxt[i] < len(jobs[i]) and (job:=jobs[i][nxt[i]])[0] == "copy":
              fills[i][k], nxt[i], progressed = (job, pool.submit(fill, mvs[i][k], job[1], job[2])), nxt[i] + 1, True
          if done[i] >= len(jobs[i]): continue
          job = jobs[i][done[i]]
          if job[0] == "gather":
            if cur < last_copy[i]: continue
            gather(i, job)
            last_gather[i] = tls[i][1]
            nxt[i] = max(nxt[i], done[i] + 1)
          else:
            kf = next((k for k in range(K) if fills[i][k] is not None and fills[i][k][0] is job), None)
            if kf is None or not fills[i][kf][1].done() or (job[4] == "tmp" and cur < last_gather[i]): continue
            k = kf
            fills[i][k][1].result()
            target = bufs[i] if job[4] == "buf" else tmps[i]
            # (a copy at a time: a linear of the same structure is compiled once)
            target.view(job[3], job[5]).ensure_allocated().copy_from(stages[i][k].view(job[3], 0).ensure_allocated())
            fills[i][k], busy[i][k] = None, tls[i][1]
            if job[4] == "tmp": last_copy[i] = tls[i][1]
          done[i], progressed = done[i] + 1, True
        if not progressed:
          pending = [x[1] for row in fills for x in row if x is not None and not x[1].done()]
          if pending: concurrent.futures.wait(pending, timeout=0.0005, return_when=concurrent.futures.FIRST_COMPLETED)
          else: time.sleep(0.0002)
      for d in devices: Device[d].synchronize()
      pool.shutdown()
      mvs.clear()
      stages.clear()
      tmps.clear()
    for fd in fds.values(): os.close(fd)
  if defer is not None: defer.append(fill_all)
  else: fill_all()
  packed:dict[str, Tensor] = {}
  for name, parts in plan.items():
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [bufs[i].view(nb, offsets[name][i]).ensure_allocated() for i,(nb,_) in enumerate(parts)]
    buf = UOp.from_buffer(mb, dtypes.uint8, devices)
    packed[name] = Tensor(buf if name not in units and shard_map.get(name) is None else buf.reshape(1, -1).unshard(0))
  return packed


# **** the streaming loader: the files read once (O_DIRECT, threads) into windows of host memory every device maps, the devices take their parts ****

@functools.cache
def _strided_gather_kernel(dst:UOp, src:UOp, table:UOp, count:int, piece:int, real:int, ss:int, ds:int) -> UOp:
  # for every table entry (src base, dst base): count pieces of `piece` words, src base + c*ss -> dst base + c*ds, the words past `real` zero.
  # the lanes of a workgroup the words of pieces (the reads of host memory coalesced)
  from tinygrad.uop.ops import AxisType, KernelInfo
  E = table.shape[0] // 2
  # (a workgroup: up to 256 lanes over the words of a piece, a loop over the rest)
  L = min(256, 1 << (piece - 1).bit_length())
  e, c = UOp.range(E, 0, AxisType.GLOBAL), UOp.range(count, 1, AxisType.GLOBAL)
  lane, it = UOp.range(L, 2, AxisType.LOCAL), UOp.range((piece + L - 1) // L, 3, AxisType.LOOP)
  k = it * L + lane
  if piece % L: k = k.minimum(piece - 1)  # (lanes past the piece write its last word again)
  sb, db = table[2*e].load().cast(dtypes.weakint), table[2*e+1].load().cast(dtypes.weakint)
  val = src[sb + c*ss + (k if real == piece else k.minimum(max(real, 1)-1))].load()
  if real < piece: val = (k < real).where(val, UOp.const(0, src.dtype))
  return dst[db + c*ds + k].store(val).end(e, c, lane, it).sink(arg=KernelInfo(name="strided_gather", opts_to_apply=()))

@functools.cache
def _runs_kernel(dst:UOp, src:UOp, table:UOp, run:int) -> UOp:
  # for every table entry (src, dst, count): count (at most run) words src.. -> dst.. (a workgroup of 256 lanes a chunk of 256 words)
  from tinygrad.uop.ops import AxisType, KernelInfo
  e, g, lane = UOp.range(table.shape[0] // 3, 0, AxisType.GLOBAL), UOp.range(run // 256, 1, AxisType.GLOBAL), UOp.range(256, 2, AxisType.LOCAL)
  w = g * 256 + lane
  sb, db, cnt = (table[3*e + i].load().cast(dtypes.weakint) for i in range(3))
  return dst[(db + w).valid(w < cnt)].store(src[sb + w.minimum(cnt - 1).maximum(0)].load()).end(e, g, lane).sink(
    arg=KernelInfo(name="load_runs", opts_to_apply=()))

# a reader process: reads (O_DIRECT, threads) the jobs of its stdin ("id file offset length at") into the shared memory, writes the ids done
_READER_SRC = r"""
import os, sys, mmap, threading, concurrent.futures
fd, size, threads, paths = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
mv = memoryview(mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE))
fds = [os.open(p, os.O_RDONLY | os.O_DIRECT) for p in paths]
lock, out = threading.Lock(), sys.stdout.buffer
def job(line):
  jid, pi, off, ln, at = map(int, line.split())
  got = 0
  while got < ln:
    if (r:=os.preadv(fds[pi], [mv[at+got:at+ln]], off+got)) <= 0: break
    got += r
  with lock:
    out.write(b"%d\n" % jid)
    out.flush()
pool = concurrent.futures.ThreadPoolExecutor(threads)
for line in sys.stdin.buffer: pool.submit(job, line)
pool.shutdown()
"""

def _stream_load(entries:dict[str, tuple[Tensor, tuple[int, ...], int]], devices:tuple[str, ...], shard_map:dict[str, int],
                 units:dict[str, tuple[int, list[list[int]]]], pads:dict[str, int], fuse:dict[str, str],
                 defer:list[Callable[[], None]]|None=None) -> dict[str, Tensor]|None:
  """
  _parallel_load reading every byte of the files once: runs of tensors (windows, in file order) are read (O_DIRECT, reader processes) into
  shared host memory that every device maps, a device copies its contiguous parts out of it and gathers its strided ones (kernels reading the
  host memory, from tables of the parts in the host memory too): a window is a linear of a few kernels a device
  """
  import os, sys, time, subprocess, threading, mmap
  from tinygrad.device import Buffer, BufferSpec, BufferStorage, MultiBuffer
  from tinygrad.engine.realize import compile_linear, link_linear, run_linear
  from tinygrad.runtime.support.system import FileIOInterface, MAP_FIXED, MAP_POPULATE
  n = len(devices)
  if not all(d.split(":")[0] == "AMD" for d in devices): return None
  locs:dict[str, tuple[str, int, int]] = {}
  for name, (data, _, _) in entries.items():
    try: buf = data.uop.buffer
    except Exception: return None
    if not isinstance(buf, Buffer) or not buf.device.startswith("DISK:") or buf.device.startswith("DISK:shm:"): return None
    locs[name] = (buf.device[5:], buf.offset, buf.nbytes)
  plan = _load_plan(entries, devices, shard_map, units, pads, fuse, locs)
  align, offsets, totals = 256, cast(dict[str, list[int]], {}), [0]*n
  for name, parts in plan.items():
    offsets[name] = []
    for i,(nb, _) in enumerate(parts):
      offsets[name].append(totals[i])
      totals[i] += round_up(nb, align)
  # the tensors by their bytes in the files (a fused up tensor's are its gate's plan's segments from it), in windows of up to WIN bytes
  PAGE, WIN = 4096, getenv("LOAD_WINDOW", 2 << 30)
  owner = {**{nm: nm for nm in plan}, **{u: g for g, u in fuse.items()}}
  windows:list[tuple[str, int, int, list[str]]] = []  # (path, lo, hi, tensors): lo, hi page aligned
  for nm in sorted(owner, key=lambda nm: (locs[nm][0], locs[nm][1])):
    path, lo, hi = locs[nm][0], locs[nm][1] // PAGE * PAGE, round_up(locs[nm][1] + locs[nm][2], PAGE)
    if windows and windows[-1][0] == path and lo >= windows[-1][1] and max(hi, windows[-1][2]) - windows[-1][1] <= WIN:
      w = windows[-1]
      windows[-1] = (path, w[1], max(w[2], hi), w[3] + [nm])
    else: windows.append((path, lo, hi, [nm]))
  wsize = round_up(max(hi - lo for _, lo, hi, _ in windows), 2 << 20)
  K = min(getenv("LOAD_WINDOWS", 3), len(windows))
  RUN = 1 << 14  # (the words of a run entry at most)
  ZB = round_up(max([z1 - z0 for parts in plan.values() for _, f in parts if f[0] == "segs" for z0, z1 in f[2]] + [PAGE]), PAGE)
  TB = 4 << 20  # (the table bytes of a device for a window)
  # the device memory, and views of it of a fixed size (the words of a view index in int32): the kernels' destinations (one kernel for all)
  with Context(ALLOW_DEVICE_USAGE=1): bufs = [Buffer(d, t).ensure_allocated() for d,t in zip(devices, totals)]
  def dstw(wb:int) -> int: return min(min(totals), ((1 << 31) - 1) * wb) // PAGE * PAGE
  def fill_all():
    st = time.perf_counter()
    with Context(ALLOW_DEVICE_USAGE=1):
      # shared memory (the reader processes write it) at an address of the devices' address space, mapped on every device: the windows, the
      # zeros of the padding, the tables of every device for every window
      SLOT = ZB + wsize  # (a slot: zeros, then the window: its source indices from the slot's start)
      total = K*SLOT + K*n*TB
      memfd = os.memfd_create("tinygrad_load")
      os.ftruncate(memfd, total)
      vaddr = cast(Any, Device[devices[0]]).iface.dev_impl.mm.alloc_vaddr(total, align=2 << 20)
      FileIOInterface._mmap(vaddr, total, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED | MAP_POPULATE, memfd, 0)
      cpu = Buffer("CPU", total, options=BufferSpec(external_ptr=vaddr)).ensure_allocated()
      host = cpu.as_memoryview(allow_zero_copy=True)
      region = [Buffer(d, total, opaque=BufferStorage(a:=cpu.get_buf(d)), options=BufferSpec(external_ptr=a)) for d in devices]
      t_setup = time.perf_counter() - st
      paths = dedup(w[0] for w in windows)
      NP, NT, CHUNK = getenv("LOAD_PROCS", 8), getenv("LOAD_THREADS", 8), getenv("LOAD_CHUNK", 16 << 20)
      readers = [subprocess.Popen([sys.executable, "-c", _READER_SRC, str(memfd), str(total), str(NT), *paths], pass_fds=(memfd,),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE) for _ in range(NP)]
      left:dict[int, int] = {}
      job_window:dict[int, int] = {}
      left_lock = threading.Lock()
      def collect(r:subprocess.Popen):
        for line in cast(Any, r.stdout):
          with left_lock: left[job_window.pop(int(line))] -= 1
      for r in readers: threading.Thread(target=collect, args=(r,), daemon=True).start()
      tls = [Device[d].timeline.host.view(fmt='Q') for d in devices]
      busy:list[list[int]] = [[0]*n for _ in range(K)]  # the timeline value of every device's last use of a slot
      # a window's kernels (every device: runs, gathers) read tables in the shared memory: a linear of the same kernels and table sizes (the
      # windows of the layers of a type) is compiled and linked once, the next ones write its tables and run it
      linears:dict[tuple, UOp] = {}
      tab_views:dict[tuple[int, int, int, int], Buffer] = {}
      typed:dict[tuple[int, DType, bool], UOp] = {}
      def as_word(i:int, word:DType, dst:bool) -> UOp:
        if (u:=typed.get(key:=(i, word, dst))) is None: u = typed[key] = UOp.from_buffer((bufs if dst else region)[i], word, devices[i])
        return u
      def issue(w:int, k:int):
        path, lo, hi, names = windows[w]
        win0 = k*SLOT + ZB  # (the window in the shared memory, the zeros before it)
        sig:list[tuple] = []
        tables:list[tuple[int, int, bytes]] = []  # (device, offset in its table area, int64 values)
        for i in range(n):
          # the device's parts in the window: runs (src, dst, nbytes) from the window (or the zeros), strided gathers
          runs:list[tuple[int, int, int]] = []
          gathers:list[tuple[DType, tuple[int, ...], tuple[int, int, int, int, int]]] = []
          for tname in names:
            name, (_, t_off, t_nb) = owner[tname], locs[tname]
            nb, fill = plan[name][i]
            base = offsets[name][i]
            if fill[0] == "segs":
              strided:list[tuple[int, int, int, int, int]] = []
              for src, length, dst, piece, stride in fill[1]:
                if not t_off <= src < t_off + t_nb: continue  # (the segments of the other tensor of a fused pair)
                if piece == length: runs.append((win0 + src - lo, base + dst, length))
                else: strided.append((win0 + src - lo, length, base + dst, piece, stride))
              for z0, z1 in (fill[2] if tname == name else []): runs.append((k*SLOT, base + z0, z1 - z0))
              def shape_of(x:tuple[int, int, int, int, int]) -> tuple[int, int, int]: return (x[1], x[3], x[4])
              for (length, piece, stride), sg in itertools.groupby(sorted(strided, key=shape_of), key=shape_of):
                sgl = list(sg)
                word = _gather_word(piece, stride, *(x for g in sgl for x in (g[0], g[2])))
                gathers.append((word, tuple(x for g in sgl for x in (g[0], g[2])), (length // piece, piece, piece, piece, stride)))
            else:
              t0 = win0 + locs[name][1] - lo
              if fill[1] == "units":
                E, rows, slices, blk, us = fill[2]
                word = _gather_word(blk, t0, base)
                gathers.append((word, tuple(x for j, u in enumerate(us) for x in (t0 + (u // slices * rows * slices + u % slices) * blk,
                                                                                   base + j * rows * blk)), (rows, blk, blk, slices * blk, blk)))
              else:
                outer, row, part, di = fill[2]
                real = max(0, min(part, row - di*part))
                word = _gather_word(row, part, real or part, t0, base)
                gathers.append((word, (t0 + di*part, base), (outer, part, real, row, part)))
          at = 0  # (the next table of the device)
          # the runs in words of 4 bytes (or 2, 1 where they don't align), entries of RUN words, padded to a power of two
          for wb in (4, 2, 1):
            ent:list[int] = []
            for src, dst, nbytes in runs:
              if next(x for x in (4, 2, 1) if src % x == 0 and dst % x == 0 and nbytes % x == 0) != wb: continue
              for o in range(0, nbytes // wb, RUN): ent += [src // wb + o, dst // wb + o, min(RUN, nbytes // wb - o)]
            if not ent: continue
            ent += [0, 0, 0] * ((1 << (len(ent) // 3 - 1).bit_length()) - len(ent) // 3)
            sig.append((i, "runs", wb, len(ent), at))
            tables.append((i, at, struct.pack(f"<{len(ent)}q", *ent)))
            at += round_up(8*len(ent), 256)
          for word, pairs, prm in gathers:
            wb = word.itemsize
            # (the count of pieces as it is, the sizes and strides in words)
            sig.append((i, "gather", wb, len(pairs), at, (prm[0],) + tuple(x // wb for x in prm[1:])))
            tables.append((i, at, struct.pack(f"<{len(pairs)}q", *(x // wb for x in pairs))))
            at += round_up(8*len(pairs), 256)
          assert at <= TB, "the load tables of a window don't fit"
        # the tables to the shared memory, the window's linear (made the first time its kernels are)
        for i, at, data in tables: host[(t:=K*SLOT + (k*n + i)*TB + at):t+len(data)] = data
        if (lin:=linears.get(key:=(k, tuple(sig)))) is None:
          calls = []
          for i, kind, wb, cnt, at, *gp in sig:
            word = {4: dtypes.uint32, 2: dtypes.uint16, 1: dtypes.uint8}[wb]
            if (tv:=tab_views.get(tk:=(k, i, at, cnt))) is None:
              tv = tab_views[tk] = region[i].view(8*cnt, K*SLOT + (k*n + i)*TB + at).ensure_allocated()
            fxn = functools.partial(_runs_kernel, run=RUN) if kind == "runs" else \
                  functools.partial(_strided_gather_kernel, count=gp[0][0], piece=gp[0][1], real=gp[0][2], ss=gp[0][3], ds=gp[0][4])
            bs = (as_word(i, word, True), as_word(i, word, False), UOp.from_buffer(tv, dtypes.int64, devices[i]))
            calls.append(fxn(*[UOp.placeholder_like(b, slot=j) for j,b in enumerate(bs)]).call(*bs))
          lin = linears[key] = link_linear(compile_linear(UOp(Ops.LINEAR, src=tuple(calls)), cache=True))
        run_linear(lin, update_stats=False, jit=True)
        for i in range(n): busy[k][i] = tls[i][1]
      # the pipeline: a window's reads start once its slot is free (every device done with it), its copies once they're done
      reading:dict[int, int] = {}  # window -> slot
      nxt_read, nxt_issue, read_bytes, jid = 0, 0, 0, 0
      t_issue, t_wait_read, t_wait_slot = 0.0, 0.0, 0.0
      while nxt_issue < len(windows):
        while nxt_read < len(windows) and nxt_read - nxt_issue < K:
          k = nxt_read % K
          if not all(tls[i][0] >= busy[k][i] for i in range(n)): break
          path, lo, hi, _ = windows[nxt_read]
          jobs = [(o, min(o+CHUNK, hi) - o) for o in range(lo, hi, CHUNK)]
          with left_lock: left[nxt_read] = len(jobs)
          for o, ln in jobs:
            with left_lock: job_window[jid] = nxt_read
            cast(Any, readers[jid % NP].stdin).write(f"{jid} {paths.index(path)} {o} {ln} {k*SLOT + ZB + o - lo}\n".encode())
            jid += 1
          for r in readers: cast(Any, r.stdin).flush()
          reading[nxt_read] = k
          read_bytes += hi - lo
          nxt_read += 1
        t0 = time.perf_counter()
        if nxt_issue not in reading or left[nxt_issue]:
          time.sleep(0.0002)
          if nxt_issue not in reading: t_wait_slot += time.perf_counter() - t0
          else: t_wait_read += time.perf_counter() - t0
          continue
        issue(nxt_issue, reading.pop(nxt_issue))
        t_issue += time.perf_counter() - t0
        nxt_issue += 1
      t0 = time.perf_counter()
      for d in devices: Device[d].synchronize()
      t_sync = time.perf_counter() - t0
      for r in readers: cast(Any, r.stdin).close()
      for r in readers: r.wait()
      linears.clear()
      tab_views.clear()
      typed.clear()
      os.close(memfd)
      dt = time.perf_counter() - st
      if DEBUG_LOAD:
        print(f"stream load: {len(windows)} windows of up to {wsize/1e9:.2f} GB, {K} at once: setup {t_setup:.2f}s, issue {t_issue:.2f}s, "
              f"waiting for reads {t_wait_read:.2f}s, for slots {t_wait_slot:.2f}s, sync {t_sync:.2f}s")
        print(f"stream load: {read_bytes/1e9:.1f} GB read in {dt:.2f}s, {read_bytes/1e9/dt:.1f} GB/s")
  if defer is not None: defer.append(fill_all)
  else: fill_all()
  packed:dict[str, Tensor] = {}
  for name, parts in plan.items():
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [bufs[i].view(nb, offsets[name][i]).ensure_allocated() for i,(nb,_) in enumerate(parts)]
    buf = UOp.from_buffer(mb, dtypes.uint8, devices)
    packed[name] = Tensor(buf if name not in units and shard_map.get(name) is None else buf.reshape(1, -1).unshard(0))
  return packed

_staging:dict[str, Any] = {}
def _gather_units(data:Tensor, E:int, rows:int, slices:int, units:list[int], by_rows:bool, device:str) -> Tensor:
  """the bytes of `units` of the (E, rows, cols) packed tensor `data` on `device`, (1, bytes). a unit is the rows (or the column blocks) of a slice"""
  from tinygrad.device import Buffer, BufferSpec
  buf = data.uop.buffer
  assert isinstance(buf, Buffer) and buf.device.startswith("DISK:"), "units are gathered from the gguf file"
  src = _file_mmap(buf.device[5:])[buf.offset:buf.offset+buf.nbytes]
  unit = buf.nbytes // (E*slices)
  nbytes = unit * len(units)
  # a host buffer of the device: written here, then one copy to its memory
  if (stage:=_staging.get(device)) is None or stage.nbytes < nbytes:
    stage = _staging[device] = Buffer(device, nbytes, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated()
  dst = stage.as_memoryview(allow_zero_copy=True)[:nbytes]
  if by_rows:
    for i,u in enumerate(units): dst[i*unit:(i+1)*unit] = src[u*unit:(u+1)*unit]
  else:
    # unit (e, j) is the j-th column block of every row of expert e: strided copies of words
    block = unit // rows
    item = next(i for i in (8, 4, 2, 1) if block % i == 0 and buf.offset % i == 0)
    fmt = {8: "Q", 4: "I", 2: "H", 1: "B"}[item]
    s, d, words, row_words = src.cast(cast(Any, fmt)), dst.cast(cast(Any, fmt)), block // item, block * slices // item
    for i,u in enumerate(units):
      e, j = divmod(u, slices)
      base, out = e*rows*row_words + j*words, d[i*rows*words:(i+1)*rows*words]
      for t in range(words): out[t::words] = s[base+t:base+rows*row_words:row_words]
  out_buf = Buffer(device, nbytes).ensure_allocated().copy_from(stage.view(nbytes, 0).ensure_allocated() if stage.nbytes != nbytes else stage)
  Device[device].synchronize()
  return Tensor(UOp.from_buffer(out_buf, dtypes.uint8, device)).reshape(1, -1)

def gguf_shard(entries:dict[str, tuple[Tensor, tuple[int, ...], int]], devices:tuple[str, ...],
               shard_map:dict[str, int]|None=None, units:dict[str, tuple[int, list[list[int]]]]|None=None,
               pads:dict[str, int]|None=None, fuse:dict[str, str]|None=None, defer:list[Callable[[], None]]|None=None) -> dict[str, Tensor]:
  """
  Loads the parsed `entries` on `devices`: a tensor in `shard_map` is sharded on the given axis, the others are copied to every device.
  a tensor (E, rows, cols) in `units` (slices, per device unit lists) is cut in E*slices units: unit e*slices+j is expert e's rows j*rows/slices..
  (its columns for the down projection). every device gets its units in order, a tensor (devices*units, unit rows, unit cols)
  pads: a sharded tensor is zero padded on its shard axis to this size (gguf_pad)
  fuse: gate -> up tensors in units to load as one gate tensor, their unit rows interleaved by 16 (if loaded straight from the file)
  defer: the bytes are read later, by the function added to it (if loaded straight from the file)
  """
  pads, fuse = pads or {}, fuse or {}
  n, packed, shard_map, units, shapes = len(devices), {}, shard_map or {}, units or {}, {}
  for name, (_, shape, _) in entries.items():
    if name in units: shapes[name] = (n*len(units[name][1][0]), shape[1] // units[name][0], shape[2]) if not name.endswith("down_exps.weight") else \
      (n*len(units[name][1][0]), shape[1], shape[2] // units[name][0])
  direct = _stream_load(entries, devices, shard_map, units, pads, fuse, defer) if getenv("GGUF_STREAM", 1) else \
           _parallel_load(entries, devices, shard_map, units, pads, fuse, defer) if getenv("GGUF_LOAD2", 1) else \
           _direct_load(entries, devices, shard_map, units, pads, fuse)
  if direct is not None:
    packed, entries_left = direct, {}
    for g, u in fuse.items():
      shapes[g] = (shapes[g][0], 2*shapes[g][1], shapes[g][2])
      del entries[u]
  else: entries_left = {k:gguf_pad(v, unwrap(shard_map.get(k)), pads[k]) if k in pads else v for k,v in entries.items()}
  entries = {k:(d, s[:shard_map[k]] + (pads[k],) + s[shard_map[k]+1:] if k in pads else s, t) for k,(d,s,t) in entries.items()}
  for name, (data, shape, typ) in entries_left.items():
    if name in units:
      slices, per_device = units[name]
      E, rows, cols = shape
      by_rows = not name.endswith("down_exps.weight")
      parts = [_gather_units(data, E, rows, slices, us, by_rows, d) for us,d in zip(per_device, devices)]
      packed[name] = Tensor(UOp.mstack(*[p.uop.base for p in parts]).reshape(parts[0].shape).unshard(0))
      shapes[name] = (n*len(per_device[0]), rows // slices, cols) if by_rows else (n*len(per_device[0]), rows, cols // slices)
      continue
    if (axis:=shard_map.get(name)) is None: packed[name] = data.shard(devices)
    else:
      if shape[axis] % n or (axis == len(shape)-1 and shape[axis]//n % _GGML_QUANT.get(typ, (1,))[0]):
        raise ValueError(f"{name}: can't split {shape} on axis {axis} over {n} devices")
      # every device copies its own part: (devices, part bytes) sharded on the devices
      outer = prod(shape[:axis])
      if outer == 1: parts = [data[i*(sz:=data.shape[0]//n):(i+1)*sz] for i in range(n)]
      else: parts = [x[:, i].contiguous() for x in [data.to("CPU").reshape(outer, n, -1)] for i in range(n)]
      Tensor.realize(*(parts:=[p.reshape(1, -1).to(d).contiguous() for p,d in zip(parts, devices)]))
      packed[name] = Tensor(UOp.mstack(*[p.uop.base for p in parts]).reshape(parts[0].shape).unshard(0))
  if entries_left:
    for t in packed.values(): t.realize()
  for d in devices: Device[d].allocator.free_cache()
  def decode(name:str, shape:tuple[int, ...], typ:int) -> Tensor:
    if name in shapes: shape = shapes[name]
    decoded = ggml_data_to_tensor(packed[name].flatten(), prod(shape), typ)
    _remember_decoded(decoded, typ, packed[name].uop)
    if (axis:=0 if name in shapes else shard_map.get(name)) is None: return decoded.reshape(shape)
    local = decoded.reshape(n, *shape[:axis], shape[axis]//n, *shape[axis+1:])
    return local.permute(*range(1, axis+1), 0, *range(axis+1, local.ndim)).reshape(shape)
  return {name: decode(name, shape, typ) for name, (_, shape, typ) in entries.items()}
