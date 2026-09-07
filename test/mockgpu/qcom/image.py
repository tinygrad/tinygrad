"""Immutable configuration for the QCOM producer's linear RGBA float images.

Descriptors choose storage, not its lifetime. PM4 reads them through the current
transaction; both planning and IR3 validate the registered pixel allocations.
"""
from dataclasses import dataclass
import struct

from tinygrad.runtime.autogen import mesa


@dataclass(frozen=True)
class Image2D:
  base: int
  width: int
  height: int
  pitch: int
  component_bytes: int

  def validate(self, memory, write=False):
    if self.component_bytes not in (2, 4) or not 0 < self.width <= 16384 or not 0 < self.height <= 16384:
      raise RuntimeError("unsupported image dimensions or component width")
    if self.base <= 0 or self.base % 32 or self.pitch < self.width * 4 * self.component_bytes or self.pitch % 64:
      raise RuntimeError("invalid image base or row pitch")
    size = (self.height - 1) * self.pitch + self.width * 4 * self.component_bytes
    memory.validate(self.base, size, write=write)


@dataclass(frozen=True)
class ImageBindings:
  textures: tuple[Image2D, ...] = ()
  outputs: tuple[Image2D, ...] = ()
  sampler_count: int = 0  # Admitted leading samplers; reserved zero slots are inaccessible.

  def validate(self, memory):
    for texture in self.textures:
      texture.validate(memory)
    for output in self.outputs:
      output.validate(memory, write=True)


def decode_image_descriptor(data: bytes, texture: bool) -> Image2D:
  if len(data) != 64:
    raise RuntimeError("an image descriptor must contain sixteen words")
  words = struct.unpack("<16I", data)
  pixel_format = (words[0] & mesa.A6XX_TEX_CONST_0_FMT__MASK) >> mesa.A6XX_TEX_CONST_0_FMT__SHIFT
  formats = {mesa.FMT6_16_16_16_16_FLOAT: 2, mesa.FMT6_32_32_32_32_FLOAT: 4}
  if pixel_format not in formats:
    raise RuntimeError("unsupported image pixel format")

  # ops_qcom emits linear, unmipped 2D images. Texture fetches use the identity
  # swizzle; typed UAV stores do not use those swizzle bits. Keep the producer's
  # fixed words explicit rather than silently accepting tiling or compression.
  swizzle = (1 << 7) | (2 << 10) | (3 << 13) | 8 if texture else 0
  if words[0] != (pixel_format << 22 | swizzle) or words[1] & ~0x3fffffff:
    raise RuntimeError("unsupported image layout or swizzle")
  width, height = words[1] & 0x7fff, words[1] >> 15
  pitch = (words[2] & mesa.A6XX_TEX_CONST_2_PITCH__MASK) >> mesa.A6XX_TEX_CONST_2_PITCH__SHIFT
  pitch_alignment = (pitch & -pitch).bit_length() - 7
  expected_pitch = mesa.A6XX_TEX_2D << 29 | pitch << 7 | pitch_alignment
  if pitch_alignment < 0 or words[2] != expected_pitch:
    raise RuntimeError("unsupported image pitch or texture type")
  if words[3] or words[4] & 31 or words[5] & ~0x1ffff:
    raise RuntimeError("unsupported image layers or base address")
  if words[6:] != (0x40000000, 13, 0, 0, 0, 0, 0, 0, 0, 0):
    raise RuntimeError("unsupported image planes or compression")
  return Image2D(words[4] | words[5] << 32, width, height, pitch, formats[pixel_format])


def validate_sampler(data: bytes):
  # The current producer supplies nearest, unnormalized coordinates and a black
  # border. The associated zero border allocation is checked by PM4 as well.
  if len(data) != 16 or struct.unpack("<4I", data) != (3 << 5 | 3 << 8 | 3 << 11, 0x30, 0, 0):
    raise RuntimeError("unsupported image sampler")
