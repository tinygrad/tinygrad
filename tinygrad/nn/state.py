import json, pathlib, zipfile, pickle, tarfile, struct, functools, io, zlib
from collections import OrderedDict
from typing import Any, Callable, BinaryIO, Iterable, cast
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, argsort, DEBUG, Timing, CI, unwrap, GlobalCounters, tqdm, round_up, T, strides_for_shape

class TensorIO(io.RawIOBase, BinaryIO):
  def __init__(self, t: Tensor):
    if t.ndim != 1 or t.dtype != dtypes.uint8: raise ValueError("Tensor must be 1d and of dtype uint8!")
    self._position, self._tensor = 0, t

  def readable(self) -> bool: return True
  def read(self, size: int = -1) -> bytes:
    if (buf:=super().read(size)) is None: raise ValueError("io.RawIOBase.read returned None") # only happens if readinto returns None (never)
    return buf
  def readinto(self, buffer: Any) -> int:
    data = self._tensor[self._position:self._position+len(buffer)].data()
    buffer[:len(data)] = data
    self._position += len(data)
    return len(data)

  def seekable(self) -> bool: return True
  def seek(self, offset: int, whence: int = 0) -> int:
    self._position = min(len(self._tensor), max(0, [offset, self._position+offset, len(self._tensor)+offset][whence]))
    return self._position

  # required to correctly implement BinaryIO
  def __enter__(self): return self
  def write(self, s: Any): raise io.UnsupportedOperation("TensorIO.write not supported")
  def writelines(self, lines: Iterable[Any]): raise io.UnsupportedOperation("TensorIO.writelines not supported")

safe_dtypes = {"BOOL":dtypes.bool, "I8":dtypes.int8, "U8":dtypes.uint8, "I16":dtypes.int16, "U16":dtypes.uint16, "I32":dtypes.int, "U32":dtypes.uint,
               "I64":dtypes.int64, "U64":dtypes.uint64, "F16":dtypes.float16, "BF16":dtypes.bfloat16, "F32":dtypes.float32, "F64":dtypes.float64}
inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}

def accept_filename(func: Callable[[Tensor], T]) -> Callable[[Tensor|str|pathlib.Path], T]:
  @functools.wraps(func)
  def wrapper(fn: Tensor|str|pathlib.Path, *args, **kwargs) -> T:
    return func(Tensor(pathlib.Path(fn)) if not isinstance(fn, Tensor) else fn, *args, **kwargs)
  return wrapper

@accept_filename
def safe_load_metadata(t:Tensor) -> tuple[Tensor, int, dict[str, Any]]:
  """
  Loads a .safetensor file, returning the source tensor, data start position, and metadata.
  """
  data_start = int.from_bytes(t[0:8].data(), "little") + 8
  return t, data_start, json.loads(t[8:data_start].data().tobytes())

def safe_load(fn:Tensor|str|pathlib.Path) -> dict[str, Tensor]:
  """
  Loads a .safetensor file, returning the `state_dict`.

  ```python
  state_dict = nn.state.safe_load("test.safetensor")
  ```
  """
  t, data_start, metadata = safe_load_metadata(fn)
  data = t[data_start:]
  return { k: data[v['data_offsets'][0]:v['data_offsets'][1]].bitcast(safe_dtypes[v['dtype']]).reshape(v['shape'])
          for k, v in metadata.items() if k != "__metadata__" }

def safe_save(tensors:dict[str, Tensor], fn:str, metadata:dict[str, Any]|None=None):
  """
  Saves a `state_dict` to disk in a .safetensor file with optional metadata.

  ```python
  t = Tensor([1, 2, 3])
  nn.state.safe_save({'t':t}, "test.safetensor")
  ```
  """
  headers, offset = {}, 0
  if metadata: headers['__metadata__'] = metadata
  for k,v in tensors.items():
    headers[k] = {'dtype': inverse_safe_dtypes[v.dtype], 'shape': list(v.shape), 'data_offsets':[offset, offset+v.nbytes()]}
    offset += v.nbytes()
  j = json.dumps(headers, separators=(',', ':'))
  j += "\x20"*(round_up(len(j),8)-len(j))
  pathlib.Path(fn).unlink(missing_ok=True)
  t = Tensor.empty(8+len(j)+offset, dtype=dtypes.uint8, device=f"disk:{fn}")
  t[0:8].assign(Tensor([len(j)], dtype=dtypes.int64, device="CPU").bitcast(dtypes.uint8))
  t[8:8+len(j)].assign(list(j.encode('utf-8')))
  for k,v in safe_load(t).items(): v.assign(tensors[k])

# state dict

def get_state_dict(obj, prefix:str='', tensor_type=Tensor) -> dict[str, Tensor]:
  """
  Returns a `state_dict` of the object, with optional prefix.

  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(nn.state.get_state_dict(net).keys())
  ```
  """
  if isinstance(obj, tensor_type): return {prefix.strip('.'):obj}
  if hasattr(obj, '_asdict'): return get_state_dict(obj._asdict(), prefix, tensor_type)  # namedtuple
  if isinstance(obj, OrderedDict): return get_state_dict(dict(obj), prefix, tensor_type)
  if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__, prefix, tensor_type)
  state_dict = {}
  if isinstance(obj, (list, tuple)):
    for i,x in enumerate(obj): state_dict.update(get_state_dict(x, f"{prefix}{str(i)}.", tensor_type))
  elif isinstance(obj, dict):
    for k,v in obj.items(): state_dict.update(get_state_dict(v, f"{prefix}{str(k)}.", tensor_type))
  return state_dict

def get_parameters(obj) -> list[Tensor]:
  """
  ```python exec="true" source="above" session="tensor" result="python"
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  print(len(nn.state.get_parameters(net)))
  ```
  """
  return list(get_state_dict(obj).values())

def load_state_dict(model, state_dict:dict[str, Tensor], strict=True, verbose=True, consume=False, realize=True) -> list[Tensor]:
  """
  Loads a `state_dict` into a model. Return the loaded Tensors.

  ```python
  class Net:
    def __init__(self):
      self.l1 = nn.Linear(4, 5)
      self.l2 = nn.Linear(5, 6)

  net = Net()
  state_dict = nn.state.get_state_dict(net)
  nn.state.load_state_dict(net, state_dict)
  ```
  """
  start_mem_used = GlobalCounters.mem_used
  ret = []
  with Timing("loaded weights in ",
              lambda et_ns: f", {(B:=(GlobalCounters.mem_used-start_mem_used))/1e9:.2f} GB loaded at {B/et_ns:.2f} GB/s", enabled=verbose):
    model_state_dict = get_state_dict(model)
    if DEBUG >= 1 and len(state_dict) > len(model_state_dict):
      print("WARNING: unused weights in state_dict", sorted(list(state_dict.keys() - model_state_dict.keys())))
    for k,v in (t := tqdm(model_state_dict.items(), disable=CI or not verbose)):
      t.desc = f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, {k:50s}: "
      if k not in state_dict and not strict:
        if DEBUG >= 1: print(f"WARNING: not loading {k}")
        continue
      if v.shape != state_dict[k].shape:
        if {(), (1,)} == {state_dict[k].shape, v.shape}: state_dict[k] = state_dict[k].reshape(v.shape)
        else: raise ValueError(f'Shape mismatch in layer `{k}`: Expected shape {v.shape}, but found {state_dict[k].shape} in state dict.')
      if isinstance(v.device, tuple):
        if isinstance(state_dict[k].device, tuple): v.replace(state_dict[k])
        else: v.replace(state_dict[k].shard(v.device, v.uop.axis))
      else: v.replace(state_dict[k].to(v.device))
      if realize: v.realize()
      if consume: del state_dict[k]
      ret.append(v)
  return ret

@accept_filename
def zip_extract(t: Tensor) -> dict[str, Tensor]:
  files: dict[str, Tensor] = {}
  file_offsets: dict[str, tuple[Tensor, int, int]] = {}
  with zipfile.ZipFile(TensorIO(t), "r") as myzip:
    for zi in myzip.filelist:
      file_offset = zi.header_offset+30+t[zi.header_offset+26:zi.header_offset+30].bitcast(dtypes.uint16).to("CPU").sum()
      file_offsets[zi.filename] = (file_offset, zi.compress_size, zi.compress_type)
  # sadly, the extra length needs to be read from the local header of each file. this is a limitation of the zip file format
  Tensor.realize(*[x[0] for x in file_offsets.values()])
  for filename, (file_offset, compress_size, compress_type) in file_offsets.items():
    # possible to remove this realize/item? it's slow
    file_offset_int = int(file_offset.item())
    files[filename] = t[file_offset_int:file_offset_int+compress_size]
    match compress_type:
      case zipfile.ZIP_STORED: pass
      # TODO: we need a zlib UOp so this can be lazy
      case zipfile.ZIP_DEFLATED: files[filename] = Tensor(zlib.decompress(files[filename].data(), -15))
      case _: raise NotImplementedError(f"compression {compress_type} not supported")
  return files

@accept_filename
def tar_extract(t: Tensor) -> dict[str, Tensor]:
  """
  ```python
  tar_extract(fn: Tensor | str | Path) -> dict[str, Tensor]
  ```

  Extracts files from a tar archive and returns them as a dictionary of names (keys) and tensors (values).

  ```python
  tensors = nn.state.tar_extract(Tensor(pathlib.Path("archive.tar")))
  ```
  """
  with tarfile.open(fileobj=TensorIO(t), mode="r") as tar:
    return {member.name:t[member.offset_data:member.offset_data+member.size] for member in tar if member.type == tarfile.REGTYPE}

# torch support!

# TODO: this should use tar_extract and zip_extract
@accept_filename
def torch_load(t:Tensor) -> dict[str, Tensor]:
  """
  ```python
  torch_load(fn: Tensor | str | Path) -> dict[str, Tensor]
  ```

  Loads a torch .pth file, returning the `state_dict`.

  ```python
  state_dict = nn.state.torch_load("test.pth")
  ```
  """
  offsets: dict[str|int, int] = {}
  lens: dict[str|int, int] = {}

  def _rebuild_tensor(storage, storage_offset, size, stride):
    return _rebuild_tensor_v2(storage, storage_offset, size, stride)

  def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=None, backward_hooks=None, metadata=None):
    #print(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
    lens[storage[2]] = storage[4] * storage[1].itemsize
    if storage[2] not in offsets: return None
    byte_offset = offsets[storage[2]]+storage_offset*storage[1].itemsize
    ret = t[byte_offset:byte_offset+prod(size)*storage[1].itemsize].bitcast(storage[1])

    # 7 lines to deal with permuted tensors. NOTE: this currently requires reading off the disk
    shape_strides = [(s, st) for s,st in zip(size, stride) if s != 1]
    permute_indexes = [len(shape_strides)-1-y for y in argsort([x[1] for x in shape_strides])]
    if tuple(permute_indexes) != tuple(range(len(permute_indexes))):
      intermediate_shape = tuple([shape_strides[x][0] for x in argsort(permute_indexes)])
      assert tuple([shape_strides[i][1] for i in argsort(permute_indexes)]) == strides_for_shape(intermediate_shape), "nonpermutable strides"
      if DEBUG >= 3: print(f"WARNING: this torch load is slow. to permute {intermediate_shape} with {permute_indexes}")
      assert storage[1] != dtypes.bfloat16, "can't permute BF16"
      # TODO: find a nice way to support all movement ops on disktensors
      ret = ret.to(None).reshape(intermediate_shape).permute(permute_indexes)

    return ret.reshape(size)

  class Parameter:
    def __setstate__(self, state): self.tensor = state[0]

  deserialized_objects: dict[str, Any] = {}
  intercept = {"HalfStorage": dtypes.float16, "FloatStorage": dtypes.float32, "BFloat16Storage": dtypes.bfloat16,
               "IntStorage": dtypes.int32, "BoolStorage": dtypes.bool,
               "LongStorage": dtypes.int64, "_rebuild_tensor": _rebuild_tensor, "_rebuild_tensor_v2": _rebuild_tensor_v2,
               "FloatTensor": None, "Parameter": Parameter}
  whitelist = {"torch", "collections", "numpy", "_codecs"}  # NOTE: this is not for security, only speed
  class Dummy: pass
  class TorchPickle(pickle.Unpickler):
    def find_class(self, module, name):
      module_root = module.split(".")[0]
      if module_root not in whitelist:
        if DEBUG >= 2: print(f"WARNING: returning Dummy for {module} {name}")
        return Dummy
      return intercept[name] if module_root == "torch" else super().find_class(module, name)
    def persistent_load(self, pid): return deserialized_objects.get(pid, pid)

  fobj = io.BufferedReader(TensorIO(t))
  def passthrough_reset(v: bool): return fobj.seek(0, 0) or v

  if passthrough_reset(zipfile.is_zipfile(fobj)): # NOTE: passthrough_reset required to support python < 3.14
    myzip = zipfile.ZipFile(fobj, 'r')
    base_name = None
    header_offsets = {}
    for zi in myzip.filelist:
      if base_name is None: base_name = zi.filename.split('/', 1)[0]
      if zi.filename.startswith(f'{base_name}/data/'): header_offsets[zi.filename.split("/")[-1]] = zi.header_offset
    # sadly there's no way to get the start of the file in the zip without reading the header
    # at least here we read them in parallel
    header_contents = [t[v+26:v+30].bitcast(dtypes.uint16).to('CPU') for v in header_offsets.values()]
    Tensor.realize(*header_contents)
    for (n,o),c in zip(header_offsets.items(), header_contents):
      # header_offset + sizeFileHeader + File name length + Extra field length : https://en.wikipedia.org/wiki/ZIP_(file_format)
      offsets[n] = o+30+sum(cast(list[int], c.tolist()))
    with myzip.open(f'{base_name}/data.pkl') as myfile:
      return TorchPickle(myfile).load()
  elif passthrough_reset(tarfile.is_tarfile(fobj)): # NOTE: passthrough_reset required to support python < 3.11
    with tarfile.open(fileobj=fobj, mode="r") as tar:
      storages_offset = tar.getmember('storages').offset_data
      f = unwrap(tar.extractfile('storages'))
      for i in range(TorchPickle(f).load()):  # num_storages
        (key, _, storage_type), sz = TorchPickle(f).load(), struct.unpack('<q', f.read(8))[0]
        offsets[key] = storages_offset + f.tell()
        f.seek(sz*storage_type.itemsize, 1)
      f = unwrap(tar.extractfile('tensors'))
      for _ in range(TorchPickle(f).load()):  # num_tensors
        (key, storage_id, _), ndim, _ = TorchPickle(f).load(), struct.unpack('<i', f.read(4))[0], f.read(4)
        size, stride = struct.unpack(f'<{ndim}q', f.read(8 * ndim)), struct.unpack(f'<{ndim}q', f.read(8 * ndim))
        storage_offset = struct.unpack('<q', f.read(8))[0]
        deserialized_objects[str(key)] = _rebuild_tensor_v2((None, storage_type, storage_id, None, -1), storage_offset, size, stride)
      return {k:v.tensor if isinstance(v, Parameter) else v for k,v in TorchPickle(unwrap(tar.extractfile('pickle'))).load().items()}
  else:
    pkl = TorchPickle(fobj)
    _, _, _, rwd, _, ids, base_offset = pkl.load(), pkl.load(), pkl.load(), fobj.tell(), pkl.load(), pkl.load(), fobj.tell()
    for i in ids:
      offsets[i] = base_offset + 8
      base_offset += 8 + lens[i]
    fobj.seek(rwd)
    return TorchPickle(fobj).load()

def _q_to_uint8(t: Tensor, b: int) -> Tensor:
  # TODO: rewrite with arange?
  shift_tensor, bitmask = Tensor.stack(*[ Tensor(2**(i*b), device=t.device, dtype=t.dtype) for i in range(8//b) ]), 0xff >> (8 - b)
  return t.unsqueeze(-1).expand((*t.shape,8//b)).idiv(shift_tensor).bitwise_and(bitmask).transpose(-1, -2).flatten(-2)

def _dequantize_q4_0(blocks: Tensor) -> Tensor:
  d = blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
  lo, hi = blocks[:,2:].bitwise_and(0xF), blocks[:,2:].rshift(4)
  return (Tensor.cat(lo, hi, dim=-1).cast(dtypes.float32) - 8) * d

def _dequantize_q4_1(blocks: Tensor) -> Tensor:
  d, m = (blocks[:,s:s+2].bitcast(dtypes.float16).cast(dtypes.float32) for s in [0, 2])
  return _q_to_uint8(blocks[:,4:], 4).bitcast(dtypes.int8) * d + m

def _dequantize_q5(blocks: Tensor, signed: bool) -> Tensor:
  d = blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32)
  qh_start = 4 if not signed else 2
  qh = blocks[:,qh_start:qh_start+4].flatten(-2).bitcast(dtypes.uint32)
  qs = blocks[:,qh_start+4:qh_start+20]
  ql_lo, ql_hi = qs.bitwise_and(0x0F), qs.rshift(4)
  shifts_lo = Tensor([1 << i for i in range(16)], dtype=dtypes.uint32, device=blocks.device)
  shifts_hi = Tensor([1 << (i + 16) for i in range(16)], dtype=dtypes.uint32, device=blocks.device)
  qh_exp = qh.unsqueeze(-1)
  qh_lo = qh_exp.bitwise_and(shifts_lo).ne(0).cast(dtypes.uint8).lshift(4)
  qh_hi = qh_exp.bitwise_and(shifts_hi).ne(0).cast(dtypes.uint8).lshift(4)
  if not signed:
    q = ql_lo.bitwise_or(qh_lo).cat(ql_hi.bitwise_or(qh_hi), dim=-1).cast(dtypes.float32)
    return d * q + blocks[:,2:4].bitcast(dtypes.float16).cast(dtypes.float32)
  return d * (ql_lo.bitwise_or(qh_lo).cat(ql_hi.bitwise_or(qh_hi), dim=-1).bitcast(dtypes.int8).cast(dtypes.float32) - 16.0)

def _dequantize_q8_0(blocks: Tensor) -> Tensor:
  return blocks[:,:2].bitcast(dtypes.float16).cast(dtypes.float32) * blocks[:,2:].bitcast(dtypes.int8)

def _dequantize_q4k(blocks: Tensor) -> Tensor:
  d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [0, 2])
  s = blocks[:,4:16]
  sc = s[:,0:4].bitwise_and(63).cat(s[:,8:12].bitwise_and(0xF).bitwise_or(s[:,0:4].rshift(6).lshift(4)), dim=-1)
  mn = s[:,4:8].bitwise_and(63).cat(s[:,8:12].rshift(4).bitwise_or(s[:,4:8].rshift(6).lshift(4)), dim=-1)
  q = Tensor.stack((qs:=blocks[:,16:144].reshape(-1,4,32)).bitwise_and(0xF), qs.rshift(4), dim=2).reshape(-1,8,32).cast(dtypes.float32)
  return (d * sc.cast(dtypes.float32).unsqueeze(-1) * q - dmin * mn.cast(dtypes.float32).unsqueeze(-1)).flatten(-2)

def _dequantize_q5k(blocks: Tensor) -> Tensor:
  d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [0, 2])
  s = blocks[:,4:16]
  sc_low, mn_low = s[:,0:4].bitwise_and(63), s[:,4:8].bitwise_and(63)
  sc_high = s[:,8:12].bitwise_and(0xF).bitwise_or(s[:,0:4].rshift(6).lshift(4))
  mn_high = s[:,8:12].rshift(4).bitwise_or(s[:,4:8].rshift(6).lshift(4))
  sc, mn = sc_low.cat(sc_high, dim=-1), mn_low.cat(mn_high, dim=-1)
  qh, qs = blocks[:,16:48], blocks[:,48:176]
  ql = Tensor.stack(qs.reshape(-1,4,32).bitwise_and(0xF), qs.reshape(-1,4,32).rshift(4), dim=2).reshape(-1,8,32)
  qh_bits = _q_to_uint8(qh, 1).reshape(-1, 8, 32)
  q = ql.bitwise_or(qh_bits.lshift(4)).cast(dtypes.float32)
  return (d * sc.cast(dtypes.float32).unsqueeze(-1) * q - dmin * mn.cast(dtypes.float32).unsqueeze(-1)).flatten(-2)

def _dequantize_q6k(blocks: Tensor) -> Tensor:
  xl, xh = _q_to_uint8(blocks[:,:128].reshape((-1, 2, 64)), 4), _q_to_uint8(blocks[:,128:192].reshape((-1, 2, 32)), 2).lshift(4)
  scales = blocks[:,192:208].bitcast(dtypes.int8).cast(dtypes.float32).unsqueeze(-1).expand((-1, 16, 16)).reshape((-1, 256))
  d = blocks[:,-2:].bitcast(dtypes.float16).cast(dtypes.float32).expand((-1, 256))
  return d * (xl.bitwise_or(xh).bitcast(dtypes.int8).cast(dtypes.float32) - 32).flatten(-2) * scales

def _dequantize_mxfp4(blocks: Tensor) -> Tensor:
  e_int = blocks[:, 0].cast(dtypes.int32)
  d = ((e_int >= 2).cast(dtypes.float32) * (e_int.cast(dtypes.float32) - 128).exp2() +
       (e_int == 1).cast(dtypes.float32) * 2.0**(-127) +
       (e_int == 0).cast(dtypes.float32) * 2.0**(-128)).unsqueeze(-1)
  codes = _q_to_uint8(blocks[:, 1:17], 4)
  sign = 1.0 - codes.rshift(3).cast(dtypes.float32) * 2.0
  exp, mant = codes.rshift(1).bitwise_and(0x3).cast(dtypes.float32), codes.bitwise_and(0x1).cast(dtypes.float32)
  fp4_val = sign * ((exp != 0).cast(dtypes.float32) * (1.0 + 0.5 * mant) * (exp - 1.0).exp2() +
                    (exp == 0).cast(dtypes.float32) * 0.5 * mant)
  return (fp4_val * d).flatten(-2)

# GGML quantization info: (elements_per_block, bytes_per_block, dequantize_fn)
GGML_QUANT_INFO: dict[int, tuple[int, int, Callable[[Tensor], Tensor]]] = {
  2: (32, 18, _dequantize_q4_0),                       # Q4_0
  3: (32, 20, _dequantize_q4_1),                       # Q4_1
  6: (32, 22, lambda b: _dequantize_q5(b, signed=True)),   # Q5_0
  7: (32, 24, lambda b: _dequantize_q5(b, signed=False)),  # Q5_1
  8: (32, 34, _dequantize_q8_0),                       # Q8_0
  12: (256, 144, _dequantize_q4k),                     # Q4_K
  13: (256, 176, _dequantize_q5k),                     # Q5_K
  14: (256, 210, _dequantize_q6k),                     # Q6_K
  39: (32, 17, _dequantize_mxfp4),                     # MXFP4
}

def ggml_data_to_tensor(t: Tensor, n: int, ggml_type: int) -> Tensor:
  """
  Converts ggml tensor data to a tinygrad tensor.

  Supported native types: float32 (id: 0), float16 (id: 1), int8 (id: 16), int16 (id: 17), int32 (id: 18)
  Supported quantized types:
    Q4_0 (id: 2), Q4_1 (id: 3), Q5_0 (id: 6), Q5_1 (id: 7), Q8_0 (id: 8),
    Q4_K (id: 12), Q5_K (id: 13), Q6_K (id: 14), MXFP4 (id: 39)
  """
  # https://github.com/ggerganov/ggml/blob/323951f1bdcdfbd5b5ff3a9a7c3770e63b1a560e/include/ggml.h#L356

  # native types
  if (dtype := { 0: dtypes.float32, 1: dtypes.float16, 16: dtypes.int8, 17: dtypes.int16, 18: dtypes.int32 }.get(ggml_type)) is not None:
    return t[:dtype.itemsize * n].bitcast(dtype)

  # quantized types
  if (info := GGML_QUANT_INFO.get(ggml_type)) is not None:
    el_per_block, bytes_per_block, dequant_fn = info
    blocks = t[:(n//el_per_block)*bytes_per_block].reshape((-1, bytes_per_block))
    return dequant_fn(blocks)[:n] if ggml_type == 39 else dequant_fn(blocks)

  raise ValueError(f"GGML type '{ggml_type}' is not supported!")

def extract_quantized_blocks(tensor: Tensor, t_infos: list, data_start: int) -> dict[str, tuple[Tensor, tuple, int, bool]]:
  """Extract raw quantized blocks from GGUF tensor data. Returns dict mapping tensor name to (blocks, shape, ggml_type, expert_first_in_memory)."""
  quantized_tensors = {}
  for name, dims, typ, off in t_infos:
    if typ not in GGML_QUANT_INFO: continue
    n = prod(dims)
    el_per_block, bytes_per_block, _ = GGML_QUANT_INFO[typ]
    nbytes = (n // el_per_block) * bytes_per_block
    raw_blocks = tensor[data_start + off:(data_start + off) + nbytes].reshape(-1, bytes_per_block)
    # detect expert memory layout: expert dim is smallest of 3 GGUF dims, last = expert-first in memory
    expert_first_in_memory = True
    if len(dims) == 3 and any(x in name for x in ['_exps', '_shexp']):
      expert_first_in_memory = (min(range(3), key=lambda i: dims[i]) == 2)
    quantized_tensors[name] = (raw_blocks, tuple(reversed(dims)), typ, expert_first_in_memory)
  return quantized_tensors

@accept_filename
def gguf_load(tensor: Tensor, quantized: bool = False) -> tuple[dict, dict[str, Tensor], dict[str, tuple[Tensor, tuple, int, bool]]|None]:
  """
  Loads a .gguf file, returning the `kv_data` and `state_dict`.

  ```python
  gguf_tensor = Tensor(pathlib.Path("Meta-Llama-3-8B-Instruct.Q4_0.gguf")).to(Device.DEFAULT)
  kv_data, state_dict, quantized_tensors = nn.state.gguf_load(gguf_tensor)
  ```
  If quantized=True, returns a third dict mapping tensor names to (raw_blocks, shape, ggml_type, expert_first_in_memory) for Q4_K/Q5_K/Q6_K tensors.
  These tensors are NOT included in state_dict, allowing streamed dequantization during inference.
  NOTE: The provided tensor must be on a device that supports execution.
  """
  reader, kv_data, state_dict = io.BufferedReader(TensorIO(tensor), 1_000_000), {}, {}
  quantized_tensors: dict[str, tuple[Tensor, tuple, int, bool]]|None = {} if quantized else None
  def read_unpack(fmt: str, n: int): return struct.unpack(fmt, reader.read(n))[0]
  def read_str(): return str(reader.read(read_uint64()), "utf-8")
  def read_arr():
    reader, n = readers[read_int32()], read_uint64()
    return [ reader() for _ in range(n) ]

  readers: dict[int, Callable[[], Any]] = { 8: read_str, 9: read_arr, **{ t: functools.partial(read_unpack, "<"+f, nb) for t,f,nb in \
    [ (0,"c",1), (1,"b",1), (2,"H",2), (3,"h",2), (4,"I",4), (5,"i",4), (6,"f",4), (7,"?",1), (10,"Q",8), (11,"q",8), (12,"d",8) ] } }
  read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

  magic, version, n_tensors, n_kv = reader.read(4), read_int32(), read_int64(), read_int64()
  if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")
  for _ in range(n_kv):
    k, typ = read_str(), read_int32()
    kv_data[k] = readers[typ]()

  t_infos = [ (read_str(), tuple(read_uint64() for _ in range(read_uint32())), read_int32(), read_uint64()) for _ in range(n_tensors) ]
  alignment, pos = kv_data.get("general.alignment", 32), reader.tell()
  data_start = round_up(pos, alignment)

  quantized_tensors = extract_quantized_blocks(tensor, t_infos, data_start) if quantized else None
  for name, dims, typ, off in t_infos:
    if quantized and quantized_tensors is not None and name in quantized_tensors: continue
    n = prod(dims)
    state_dict[name] = ggml_data_to_tensor(tensor[data_start + off:], n, typ).reshape(*reversed(dims))

  return kv_data, state_dict, quantized_tensors
