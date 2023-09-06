from collections import namedtuple
from enum import IntEnum
from io import IOBase
import pprint
from struct import unpack, calcsize
from typing import Union
import numpy as np

GGUF_MAGIC             = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32

class GGMLQuantizationType(IntEnum):
    F32  = 0
    F16  = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

# for checks that aren't strictly necessary
def paranoid_check(passed, error):
  if not passed:
    raise ValueError(error)

QK4_0 = 32

def dequantize_q4_0(src: IOBase, dest: np.ndarray, k):
  qk = QK4_0
  paranoid_check(k % qk == 0, f"k={k} not divisible by qk={qk}")
  nb = k // qk
  for i in range(nb):
    b = src.read(2)
    print(b)
    d = unpack('<e', b)[0]
    qs = src.read(QK4_0 // 2)
    for j in range(qk//2):
      x0 = (qs[j] & 0xF) - 8
      x1 = (qs[j] >> 4) - 8
      dest[i * qk + j + 0   ] = x0*d
      dest[i * qk + j + qk//2] = x1*d
  

class GGUFReader:
  Tensor = namedtuple('Tensor', 'name shape type offset')

  def __init__(self, stream: IOBase):
    if not stream.seekable():
      raise ValueError("Stream is not seekable")
    self.stream = stream

  def read_versioned_int(self):
    if self.version == 1:
      return unpack('<I', self.stream.read(4))[0]
    return unpack('<Q', self.stream.read(8))[0]


  def read_str(self):
    len = self.read_versioned_int()
    return self.stream.read(len).decode("utf-8")

  def read_array(self):
    vt = GGUFValueType(unpack('<I', self.stream.read(4))[0])
    len = self.read_versioned_int()
    if vt == GGUFValueType.ARRAY:
      raise ValueError('Nested arrays not allowed')
    result = []
    for i in range(len):
      result.append(self.read_vt(vt))
    return result

  _simple_value_packing = {
      GGUFValueType.UINT8:   "<B",
      GGUFValueType.INT8:    "<b",
      GGUFValueType.UINT16:  "<H",
      GGUFValueType.INT16:   "<h",
      GGUFValueType.UINT32:  "<I",
      GGUFValueType.INT32:   "<i",
      GGUFValueType.FLOAT32: "<f",
      GGUFValueType.UINT64:  "<Q",
      GGUFValueType.INT64:   "<q",
      GGUFValueType.FLOAT64: "<d",
      GGUFValueType.BOOL:    "?" ,
  }

  def read_vt(self, vt):
    if vt == GGUFValueType.STRING:
      return self.read_str()
    if vt == GGUFValueType.ARRAY:
      return self.read_array()
    if vt in GGUFReader._simple_value_packing:
      fmt = GGUFReader._simple_value_packing[vt]
      return unpack(fmt, self.stream.read(calcsize(fmt)))[0]
    raise ValueError(f'Value type not supported {vt}')

  def parse_header(self):
    magic, self.version = unpack('<II', self.stream.read(8))
    if magic != GGUF_MAGIC:
       raise ValueError(f'Wrong magic {magic:X}')
    if self.version != 1 and self.version != 2:
       raise ValueError(f'Unsupported version {self.version}')
    if self.version == 1:
      tensor_cnt, meta_cnt = unpack('<II', self.stream.read(8))
    else:
      tensor_cnt, meta_cnt = unpack('<QQ', self.stream.read(16))
    self.meta = dict()
    alignment = GGUF_DEFAULT_ALIGNMENT
    for i in range(meta_cnt):
       key = self.read_str()
       vt = GGUFValueType(unpack('<I', self.stream.read(4))[0])
       self.meta[key] = self.read_vt(vt)
       if key == "general.aligment":
         alignment = self.meta[key]
    self.tensors = []
    for i in range(tensor_cnt):
      name = self.read_str()
      dim_cnt = unpack('<I', self.stream.read(4))[0]
      paranoid_check(dim_cnt <= 4, f"Too many dimensions {dim_cnt} for tensor {name} expected max 4")
      dims = [self.read_versioned_int() for _ in range(dim_cnt)]
      vt = GGMLQuantizationType(unpack('<I', self.stream.read(4))[0])
      offset = unpack('<Q', self.stream.read(8))[0]
      self.tensors.append(GGUFReader.Tensor(name, dims, vt, offset))
    tensor_base = self.stream.tell()
    pad = tensor_base % alignment
    if pad:
      tensor_base += alignment - pad
    self.tensors = [t._replace(offset=t.offset + tensor_base) for t in self.tensors]
    return self.meta
  
  def read_tensor_as_np(self, id: Union[str, int]):
    if type(id) is str:
      tensor = next(t for t in self.tensors if t.name == id)
    else:
      tensor = self.tensors[id]
    self.stream.seek(tensor.offset)
    if tensor.type == GGMLQuantizationType.F32:
      values = 1
      for d in tensor.shape:
        values *= d
      return np.frombuffer(self.stream.read(values*4), dtype=np.float32).reshape(tensor.shape)
    if tensor.type != GGMLQuantizationType.Q4_0:
      raise ValueError(f'Only q4_0 and f32 types supported, {tensor.name} type is {tensor.type}')
    result = np.ndarray(tensor.shape, dtype=np.float32)
    for row in range(tensor.shape[0]):
      dequantize_q4_0(self.stream, result[row], tensor.shape[0])
    return result

if __name__ == '__main__':
  f = open(r'C:\Users\jfhs\Downloads\llama-2-7b.Q4_0.gguf', 'rb')
  reader = GGUFReader(f)
  reader.parse_header()
  pprint.pprint(reader.meta.keys())
  print([t.name for t in reader.tensors])
  print(reader.read_tensor_as_np('token_embd.weight')[:10, :64])