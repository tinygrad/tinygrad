"""Directional fences within explicitly opted-in, local-kernel AQL dispatch runs.

System acquire/release fences remain at run boundaries; waits and barriers are unchanged.
"""
import struct
from tinygrad.runtime.autogen import hsa


def narrow_aql_fences(blob:bytes|bytearray, patches:list) -> bytes:
  original = bytes(blob)
  assert len(original)%64 == 0, 'partial AQL packet'
  acquire,release = hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
  system = (hsa.HSA_FENCE_SCOPE_SYSTEM << acquire)|(hsa.HSA_FENCE_SCOPE_SYSTEM << release)
  barrier = 1 << hsa.HSA_PACKET_HEADER_BARRIER
  dispatch,vendor = hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH,hsa.HSA_PACKET_TYPE_VENDOR_SPECIFIC
  kinds = []
  for offset in range(0,len(original),64):
    header = struct.unpack_from('<H',original,offset)[0]
    kind = header & 255
    assert kind in (dispatch,vendor) and header == kind|barrier|system, 'unreviewed AQL header'
    if kind == dispatch:
      assert struct.unpack_from('<H',original,offset+2)[0] == 3, 'unreviewed dispatch setup'
      assert struct.unpack_from('<Q',original,offset+56)[0] == 0, 'dispatch completion signal must be null'
    else: assert struct.unpack_from('<I',original,offset)[0] == header|(1 << 16), 'unreviewed vendor subtype'
    kinds.append(kind)
  for offset,word in patches:
    width = word.dtype.itemsize
    assert type(offset) is int and type(width) is int and width > 0
    assert 0 <= offset and offset+width <= len(original) and offset//64 == (offset+width-1)//64, 'invalid relocation extent'
    assert offset%64 >= 4, 'relocation overlaps header/setup'
    assert kinds[offset//64] != dispatch or offset%64+width <= 56, 'completion signal relocation not allowed'
  changed = bytearray(original)
  for i,kind in enumerate(kinds):
    if kind != dispatch: continue
    header = struct.unpack_from('<H',original,64*i)[0]
    if i > 0 and kinds[i-1] == dispatch: header = (header & ~(3 << acquire))|(hsa.HSA_FENCE_SCOPE_AGENT << acquire)
    if i+1 < len(kinds) and kinds[i+1] == dispatch: header = (header & ~(3 << release))|(hsa.HSA_FENCE_SCOPE_AGENT << release)
    struct.pack_into('<H',changed,64*i,header)
  return bytes(changed)
