from __future__ import annotations
import dataclasses, struct
from pathlib import Path

MTFW_VERSION_SIZE = 0x1000
MTFW_VERSION_ITEM_SIZE = 16
MTFW_ABI_TYPE_GEN = 1


@dataclasses.dataclass(frozen=True)
class MTFirmwareVersion:
  abi_type: int
  min_version: int
  max_version: int
  abi_enum: int
  name: str

  @classmethod
  def from_bytes(cls, data:bytes) -> MTFirmwareVersion:
    if len(data) != MTFW_VERSION_ITEM_SIZE: raise ValueError("Invalid MTFW version item size")
    version, abi_enum = struct.unpack_from('<II', data)
    return cls(version & 0xff, (version >> 8) & 0xfff, (version >> 20) & 0xfff, abi_enum,
               data[8:].split(b'\0', 1)[0].decode('ascii'))


@dataclasses.dataclass(frozen=True)
class MTFirmwareImage:
  code: bytes
  versions: tuple[MTFirmwareVersion, ...]

  @classmethod
  def from_bytes(cls, data:bytes) -> MTFirmwareImage:
    if len(data) <= MTFW_VERSION_SIZE: raise ValueError("MTFW image is too small")
    code, trailer = data[:-MTFW_VERSION_SIZE], data[-MTFW_VERSION_SIZE:]
    versions = []
    for off in range(0, len(trailer), MTFW_VERSION_ITEM_SIZE):
      if (raw:=trailer[off:off + MTFW_VERSION_ITEM_SIZE]) == bytes(MTFW_VERSION_ITEM_SIZE): break
      versions.append(MTFirmwareVersion.from_bytes(raw))
    if not any(v.abi_type == MTFW_ABI_TYPE_GEN and v.min_version <= 1 <= v.max_version for v in versions):
      raise ValueError("MTFW generation 1 is not supported by this image")
    return cls(code, tuple(versions))

  @classmethod
  def load(cls, path:str|Path) -> MTFirmwareImage: return cls.from_bytes(Path(path).read_bytes())
