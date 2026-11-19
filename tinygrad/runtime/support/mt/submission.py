from __future__ import annotations
import struct
from tinygrad.runtime.autogen import mtfw

CDM_EXECUTION_ADDRESS = 0x81e0e00000

def build_submission(launch_id:int=1, execution_address:int=CDM_EXECUTION_ADDRESS) -> bytes:
  """Build one compute submission with its CDM kick and error region."""
  head = mtfw.MTFW_SUBMISSION_HEAD()
  head.timestampAddr = 0x8000438200
  head.fwCswBufer = 0x6800000000
  head.retryCount = 0xffffffff
  head.subRegionDescStartOffset = 0x58
  head.subRegionDescNums = 2
  head.frameId = 0
  head.cbId = launch_id
  head.processId = 0

  cdm = mtfw.MTFW_SUBMISSION_REGION_DESC()
  cdm.regionType = 2  # MTFW_SUB_REGION_CDM
  cdm.regionStart.offset = 0xa0
  cdm.regionSize = 0x198
  error = mtfw.MTFW_SUBMISSION_REGION_DESC()
  error.regionType = 10  # MTFW_SUB_REGION_ERROR_BUFFER
  error.regionStart.gpuAddr = execution_address
  error.regionSize = 8

  kick = bytearray(0x1a8)
  struct.pack_into('<QQ',kick,0x18,0x1800000000,0x2000000001)
  struct.pack_into('<Q',kick,0x20,0)
  struct.pack_into('<Q',kick,0x28,0x2000000000)
  struct.pack_into('<I',kick,0x30,1)
  struct.pack_into('<I',kick,0x38,0)
  struct.pack_into('<Q',kick,0x40,0x30d4000000178)
  struct.pack_into('<I',kick,0x48,0x14)
  struct.pack_into('<Q',kick,0xf8,0xff00000000)
  struct.pack_into('<Q',kick,0x118,0x10000001f0)
  struct.pack_into('<I',kick,0x120,2)
  struct.pack_into('<I',kick,0x128,0)
  struct.pack_into('<Q',kick,0x170,0x81e0840000)
  struct.pack_into('<Q',kick,0x178,0x8000420300)
  struct.pack_into('<Q',kick,0x180,0x8000420380)
  struct.pack_into('<I',kick,0x188,0x2c)
  struct.pack_into('<Q',kick,0x190,0x8000428300)
  struct.pack_into('<Q',kick,0x198,0x207820c000)
  struct.pack_into('<I',kick,0x1a0,0x200)

  raw = bytearray(bytes(head).ljust(0x58,b'\0'))
  struct.pack_into('<Q',raw,0x8,0)
  struct.pack_into('<Q',raw,0x10,0x8000438200)
  struct.pack_into('<Q',raw,0x18,0x6800000000)
  packet = bytearray((bytes(raw)+bytes(cdm)+bytes(error)+kick+bytes(8)).ljust(0x19250,b'\0'))
  struct.pack_into('<QQQQQQ',packet,0x300,0xe140000040010081,0x3606c1400002,0x3601,0,0x2000000,0x80000000)
  for off in range(0x380,0x1000,0x80): struct.pack_into('<Q',packet,off,0xb0)
  return bytes(packet)
