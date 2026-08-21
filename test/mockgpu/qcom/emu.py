from __future__ import annotations
import base64, binascii, ctypes, pickle, struct
from dataclasses import dataclass
from typing import Callable, cast

from tinygrad.device import TinyELF
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import is_image_shape, prod, round_up, to_mv
from tinygrad.runtime.ops_python import PythonProgram
from tinygrad.uop.ops import Ops, UOp


BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2

def _read_u32(data:bytes, offset:int) -> int:
  if offset < 0 or offset + 4 > len(data): raise ValueError(f"invalid QCOM binary offset {offset:#x}")
  return struct.unpack_from("<I", data, offset)[0]


@dataclass(frozen=True)
class BufferPointer:
  offset: int
  descriptor: bool = False


@dataclass
class CapturedProgram:
  image: bytes
  runtime: PythonProgram
  signature: tuple[tuple[str|None, int, DType, tuple], ...]
  buffer_pointers: tuple[BufferPointer, ...]
  value_offsets: tuple[int, ...]

  def execute(self, args_addr:int, global_size:tuple[int, int, int], local_size:tuple[int, int, int], mapped_size:Callable[[int], int]):
    args_size = mapped_size(args_addr)
    buffers:list[memoryview] = []
    for (_, _, dtype, shape), pointer in zip(self.signature, self.buffer_pointers):
      pointer_addr = args_addr + pointer.offset + (16 if pointer.descriptor else 0)
      if pointer_addr + 8 > args_addr + args_size: raise RuntimeError(f"QCOM kernel argument offset {pointer.offset:#x} is out of bounds")
      address = ctypes.c_uint64.from_address(pointer_addr).value
      if address == 0: raise RuntimeError(f"QCOM kernel argument at {pointer_addr:#x} is a null pointer")
      size = prod(shape) * dtype.itemsize
      try: available = mapped_size(address)
      except RuntimeError as exc: raise RuntimeError(f"invalid QCOM buffer {shape=} {dtype=}") from exc
      if size > available and not is_image_shape(shape):
        raise RuntimeError(f"QCOM buffer {shape=} {dtype=} needs {size} bytes but its mapping has {available}")
      buffers.append(to_mv(address, available))

    value_signature = self.signature[len(self.buffer_pointers):]
    values = []
    for offset, (_, _, dtype, _) in zip(self.value_offsets, value_signature):
      if offset + dtype.itemsize > args_size: raise RuntimeError(f"QCOM scalar argument offset {offset:#x} is out of bounds")
      if dtype.fmt is None: raise RuntimeError(f"QCOM scalar argument has unsupported {dtype=}")
      values.append(cast(int, struct.unpack(dtype.fmt, ctypes.string_at(args_addr + offset, dtype.itemsize))[0]))
    self.runtime(*buffers, global_size=global_size, local_size=local_size, vals=tuple(values))


_bound_programs:dict[int, CapturedProgram] = {}


def bind_program(shader_addr:int, program:CapturedProgram):
  if ctypes.string_at(shader_addr, len(program.image)) != program.image:
    raise RuntimeError(f"captured QCOM program does not match shader memory at {shader_addr:#x}")
  _bound_programs[shader_addr] = program


def unbind_program(shader_addr:int): _bound_programs.pop(shader_addr, None)


def _qcomcl_layout(obj:TinyELF, buffer_count:int) -> tuple[bytes, tuple[BufferPointer, ...], tuple[int, ...]]:
  image_offset, image_size = _read_u32(obj.lib, 0xc0), _read_u32(obj.lib, 0x100)
  image_desc_off = _read_u32(obj.lib, 0x110)
  sampler_count = _read_u32(obj.lib, image_desc_off + 0xdc)
  descriptor_off = round_up(image_desc_off + 0x158 + len(obj.name), 4) + 8 * sampler_count
  descriptors:list[tuple[int, int]] = []
  while descriptor_off + 32 <= len(obj.lib):
    length, _, _, offset_words, _, _, _, typ = struct.unpack_from("<8I", obj.lib, descriptor_off)
    if length == 0: break
    if length < 32 or descriptor_off + length > len(obj.lib):
      raise ValueError(f"invalid QCOM argument descriptor length {length}")
    descriptors.append((offset_words * 4, typ))
    descriptor_off += length

  regular_offsets = [offset for offset, typ in descriptors if typ not in {BUFTYPE_TEX, BUFTYPE_IBO}]
  ibo_count = sum(typ == BUFTYPE_IBO for _, typ in descriptors)
  image_index = regular_index = 0
  pointers:list[BufferPointer] = []
  for _, _, _, shape in obj.signature[:buffer_count]:
    if is_image_shape(shape):
      descriptor_index = image_index if image_index < ibo_count else image_index
      pointers.append(BufferPointer(2048 + descriptor_index * 0x40, descriptor=True))
      image_index += 1
    else:
      if regular_index >= len(regular_offsets): raise ValueError("QCOM binary has too few buffer argument descriptors")
      pointers.append(BufferPointer(regular_offsets[regular_index]))
      regular_index += 1

  value_count = len(obj.signature) - buffer_count
  if regular_index + value_count > len(regular_offsets): raise ValueError("QCOM binary has too few scalar argument descriptors")
  return obj.lib[image_offset:image_offset + image_size], tuple(pointers), tuple(regular_offsets[regular_index:regular_index + value_count])


def _ir3_layout(obj:TinyELF, buffer_count:int) -> tuple[bytes, tuple[BufferPointer, ...], tuple[int, ...]]:
  from tinygrad.runtime.support.compiler_mesa import IR3Compiler
  variant, const_state, _, image = IR3Compiler.unpack_lib(obj.lib)
  args_base = const_state.ubo_state.range[0].offset
  ibo_count, texture_count = variant.num_uavs - variant.image_mapping.num_tex, variant.image_mapping.num_tex
  tex_to_image = tuple(variant.image_mapping.tex_to_image[:texture_count])

  pointers:list[BufferPointer] = []
  regular_index = image_index = 0
  for _, _, _, shape in obj.signature[:buffer_count]:
    if not is_image_shape(shape):
      pointers.append(BufferPointer(args_base + regular_index * 8))
      regular_index += 1
      continue
    if image_index < ibo_count:
      pointers.append(BufferPointer(2048 + (texture_count + image_index) * 0x40, descriptor=True))
    else:
      texture_index = image_index - ibo_count
      try: descriptor_index = tex_to_image.index(texture_index)
      except ValueError as exc: raise ValueError(f"IR3 texture {texture_index} has no descriptor mapping") from exc
      pointers.append(BufferPointer(2048 + descriptor_index * 0x40, descriptor=True))
    image_index += 1

  value_offsets = tuple(args_base + offset for offset, _ in TinyELF.iter_sig(obj.signature[buffer_count:], regular_index * 8))
  return image, tuple(pointers), value_offsets


def _uses_ir3_layout(obj:TinyELF, program:UOp) -> bool:
  if obj.target.renderer: return obj.target.renderer == "IR3"
  if len(program.src) < 3 or program.src[2].op is not Ops.SOURCE or not isinstance(source:=program.src[2].arg, str) or not source:
    raise ValueError("QCOM mock capture cannot identify the selected renderer")
  # Automatic renderer fallback leaves Target.renderer empty. IR3 source is serialized NIR in base64; QCOMCL source is OpenCL C.
  try: base64.b64decode(source, validate=True)
  except (binascii.Error, ValueError): return False
  return True


def _normalize_python_uops(uops:list[UOp]) -> list[UOp]:
  normalized:list[UOp] = []
  lowered:dict[UOp, UOp] = {}

  def lower_value(original:UOp) -> UOp:
    if (lowered_value:=lowered.get(original)) is not None: return lowered_value
    src = tuple(lower_value(value) if value.dtype is not dtypes.void else value for value in original.src)
    current = original.replace(src=src) if src != original.src else original
    # IR3 keeps native floating-point division in the final linear program while
    # PythonProgram expects it decomposed into reciprocal/multiply.
    if current.op is Ops.FDIV:
      reciprocal = UOp(Ops.RECIPROCAL, current.dtype, (current.src[1],))
      normalized.append(reciprocal)
      current = current.replace(op=Ops.MUL, src=(current.src[0], reciprocal))
    # Hardware shifts accept a scalar shift-count type different from the value
    # type. PythonProgram expects every non-comparison ALU input to share a dtype.
    if current.op in {Ops.SHL, Ops.SHR} and current.src[1].dtype != current.dtype:
      shift = current.src[1].cast(current.dtype)
      normalized.append(shift)
      current = current.replace(src=(current.src[0], shift, *current.src[2:]))
    normalized.append(current)
    lowered[original] = current
    return current

  for original in uops:
    if original.dtype is not dtypes.void:
      lower_value(original)
      continue
    src = tuple(lower_value(value) if value.dtype is not dtypes.void else value for value in original.src)
    current = original.replace(src=src) if src != original.src else original
    normalized.append(current)
    lowered[original] = current
  return normalized


def capture_program(obj:TinyELF, program:UOp):
  if program.op is not Ops.PROGRAM or len(program.src) < 2 or program.src[1].op is not Ops.LINEAR:
    raise ValueError("QCOM mock capture requires a linearized PROGRAM")
  uops = _normalize_python_uops(list(program.src[1].src))
  buffer_count = sum(u.op is Ops.PARAM and u.addrspace is not AddrSpace.ALU for u in uops)
  if _uses_ir3_layout(obj, program): image, pointers, value_offsets = _ir3_layout(obj, buffer_count)
  else: image, pointers, value_offsets = _qcomcl_layout(obj, buffer_count)
  if not image: raise ValueError("QCOM compiler produced an empty shader image")
  runtime_obj = TinyELF(pickle.dumps(uops), obj.name, obj.target, obj.signature)
  obj._qcom_mock_program = CapturedProgram(image, PythonProgram(None, runtime_obj), obj.signature, pointers, value_offsets)  # type: ignore[attr-defined,arg-type]


def find_program(shader_addr:int, available:int) -> CapturedProgram:
  if (program:=_bound_programs.get(shader_addr)) is None:
    raise RuntimeError(f"no captured QCOM program is bound at shader address {shader_addr:#x}")
  if len(program.image) > available or ctypes.string_at(shader_addr, len(program.image)) != program.image:
    raise RuntimeError(f"captured QCOM program no longer matches shader memory at {shader_addr:#x}")
  return program
