"""Compiler-produced image instructions, with independently populated pixels.

Only valid NIR enters the native compiler/decoder. Rejection cases manipulate
Python descriptor records and controlled byte storage, never host pointers.
"""
import base64
import ctypes
import functools
from dataclasses import dataclass, replace

import numpy as np
import pytest

from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.nir import IR3Renderer, nalu, nchannel, nimm, nlid, _nload_img, nstore_img
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.support.compiler_mesa import IR3Compiler
from test.mockgpu.qcom.emu import Workgroup, _image_instruction, decode, execute, validate_image_bindings
from test.mockgpu.qcom.image import Image2D, ImageBindings
from test.mockgpu.qcom.test_emu import Memory


@dataclass(frozen=True)
class ImageProgram:
  image: bytes
  constants: bytes
  local: tuple[int, int, int]
  wgid: int
  lid: int
  wgsize: int


@functools.lru_cache(maxsize=32)
def compile_image_program(local=(4, 3, 1), textures=1, component=None, read_offset=(0, 0), half_store=False, coherent=False):
  # Use the same NIR image helper calls as IR3Renderer. Giving each lane a
  # different coordinate exercises the actual compiled register allocation.
  renderer = IR3Renderer(Target(device="QCOM", renderer="IR3", arch="a630"))
  renderer.prerender([])
  builder = renderer.b
  builder.shader.contents.info.workgroup_size[:] = local
  builder.shader.contents.info.num_images = 1 if coherent else textures + 1
  coordinates = nlid(builder)
  x, y = (nchannel(builder, coordinates, axis) for axis in (0, 1))
  read_x = nalu(builder, "iadd", x, nimm(builder, read_offset[0], dtypes.int))
  read_y = nalu(builder, "iadd", y, nimm(builder, read_offset[1], dtypes.int))
  values = [
    _nload_img(builder, nimm(builder, index, dtypes.int), read_y, read_x, dtypes.float, readonly=not coherent)
    for index in range(textures)
  ]
  value = values[0]
  for additional in values[1:]:
    value = nalu(builder, "fadd", value, additional)
  if component is not None:
    selected = nchannel(builder, value, component)
    value = nalu(builder, "vec4", selected, selected, selected, selected)
  if half_store:
    value = nalu(builder, "f2f16_rtne", value)
  nstore_img(builder, nimm(builder, 0, dtypes.int), y, x, value, dtypes.half if half_store else dtypes.float)
  mesa.nir_validate_shader(builder.shader, b"image execution test")
  blob = mesa.struct_blob()
  mesa.nir_serialize(blob, builder.shader, False)
  try:
    source = base64.b64encode(ctypes.string_at(blob.data, blob.size)).decode()
    variant, state, immediates, image = IR3Compiler.unpack_lib(renderer.compiler.compile(source))
    offset = state.allocs.max_const_offset_vec4 * 16
    constants = bytes(offset) + immediates
    allocation = state.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]
    wgsize = allocation.offset_vec4 * 4 + 8 if allocation.size_vec4 else 0xfc
    if wgsize != 0xfc:
      constants = constants.ljust((wgsize + 3) * 4, b"\0")
    return ImageProgram(image, constants, local, variant.cs.work_group_id, variant.cs.local_invocation_id, wgsize)
  finally:
    mesa.ralloc_free(builder.shader)
    ctypes.CDLL(None).free(blob.data)


def image_storage(values, base, pitch=64):
  height, width, _ = values.shape
  data = bytearray(b"\x5a" * (height * pitch))
  for row in range(height):
    pixels = values[row].tobytes()
    data[row * pitch:row * pitch + len(pixels)] = pixels
  return Image2D(base, width, height, pitch, values.dtype.itemsize), data


def run_image(program, inputs, output_dtype=np.float32):
  regions, textures = {}, []
  for index, values in enumerate(inputs):
    texture, data = image_storage(values, 0x10000 * (index + 1))
    textures.append(texture)
    regions[texture.base] = data
  shape = (program.local[1], program.local[0], 4)
  output, data = image_storage(np.full(shape, -77, dtype=output_dtype), 0x400000)
  regions[output.base] = data
  memory = Memory(regions)
  bindings = ImageBindings(tuple(textures), (output,), len(textures))
  execute(program.image, program.constants, (1, 1, 1), program.local, program.wgid, program.lid, memory,
          wgsize=program.wgsize, image_bindings=bindings)
  rows = [np.frombuffer(memory.read(output.base + row * output.pitch, shape[1] * 4 * output.component_bytes), output_dtype)
          for row in range(shape[0])]
  actual = np.stack(rows).reshape(shape)
  # A pitch error or a store beyond the fourth component corrupts this padding.
  row_bytes = shape[1] * 4 * output.component_bytes
  for row in range(shape[0]):
    assert memory.read(output.base + row * output.pitch + row_bytes, output.pitch - row_bytes) == b"\x5a" * (output.pitch - row_bytes)
  return actual


@pytest.mark.parametrize("storage", [np.float32, np.float16])
def test_compiled_image_copy_respects_both_coordinates_and_storage_precision(storage):
  values = (np.arange(48).reshape(3, 4, 4) / 8 - 2).astype(storage)
  actual = run_image(compile_image_program(), [values], output_dtype=storage)
  np.testing.assert_array_equal(actual, values)


def test_partial_texture_mask_keeps_original_channel_positions():
  values = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
  program = compile_image_program(component=1)
  assert any(ins.op == "isam" and ins.fields["WRMASK"] == 2 for ins in decode(program.image))
  actual = run_image(program, [values])
  np.testing.assert_array_equal(actual, np.repeat(values[:, :, 1:2], 4, axis=2))


@pytest.mark.parametrize("count", [2, 17])
def test_distinct_textures_and_uniform_indirect_indexing(count):
  values = [np.arange(48, dtype=np.float32).reshape(3, 4, 4) / 16 + index for index in range(count)]
  program = compile_image_program(textures=count)
  if count == 17:
    assert any(ins.op == "isam" and "INDICES" in ins.operands for ins in decode(program.image))
  actual = run_image(program, values)
  np.testing.assert_array_equal(actual, np.sum(values, axis=0))


@pytest.mark.parametrize("offset", [(-1, 0), (1, 0), (0, -1), (0, 1)])
def test_outside_texture_coordinates_select_black_without_wrapping_rows(offset):
  values = (np.arange(48, dtype=np.float32).reshape(3, 4, 4) + 1)
  actual = run_image(compile_image_program(read_offset=offset), [values])
  expected = np.zeros_like(values)
  if offset == (-1, 0):
    expected[:, 1:] = values[:, :-1]
  elif offset == (1, 0):
    expected[:, :-1] = values[:, 1:]
  elif offset == (0, -1):
    expected[1:] = values[:-1]
  else:
    expected[:-1] = values[1:]
  np.testing.assert_array_equal(actual, expected)


def test_padded_rows_do_not_become_extra_pixels():
  values = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
  actual = run_image(compile_image_program(local=(3, 3, 1)), [values])
  np.testing.assert_array_equal(actual, values)


def test_half_store_uses_half_data_registers_and_nearest_even_conversion():
  values = np.resize(np.array([1.0006, -1.0006, 1.0015, -1.0015], np.float32), (3, 4, 4))
  actual = run_image(compile_image_program(half_store=True), [values], output_dtype=np.float16)
  np.testing.assert_array_equal(actual.view(np.uint16), values.astype(np.float16).view(np.uint16))


def test_full_float_store_converts_to_half_image_storage():
  values = np.resize(np.array([1.0006, -1.0006, 1.0015, -1.0015], np.float32), (3, 4, 4))
  actual = run_image(compile_image_program(), [values], output_dtype=np.float16)
  np.testing.assert_array_equal(actual.view(np.uint16), values.astype(np.float16).view(np.uint16))


@pytest.mark.parametrize("storage", [np.float16, np.float32])
def test_height_one_image_uses_the_declared_byte_pitch(storage):
  values = np.arange(16, dtype=storage).reshape(1, 4, 4) / 8
  actual = run_image(compile_image_program(local=(4, 1, 1)), [values], output_dtype=storage)
  np.testing.assert_array_equal(actual, values)


@pytest.mark.parametrize("field,value", [("TEX", 1), ("SAMP", 1)])
def test_static_sampling_indices_require_declared_resources(field, value):
  instruction = next(ins for ins in decode(compile_image_program().image) if ins.op == "isam")
  changed = replace(instruction, fields={**instruction.fields, field: value})
  bindings = ImageBindings((Image2D(0x10000, 4, 3, 64, 4),), (), 1)
  # This is pure admission of a Python instruction record; no altered encoding
  # enters the native disassembler.
  with pytest.raises(RuntimeError, match="index out of bounds"):
    validate_image_bindings((changed,), bindings)


@pytest.mark.parametrize("operation", ["stib.b", "ldib.b"])
def test_uav_index_requires_a_declared_writable_image(operation):
  program = compile_image_program(coherent=operation == "ldib.b")
  instruction = next(ins for ins in decode(program.image) if ins.op == operation)
  bindings = ImageBindings(textures=(Image2D(0x10000, 4, 3, 64, 4),), sampler_count=1)
  with pytest.raises(RuntimeError, match="output index out of bounds"):
    validate_image_bindings((instruction,), bindings)


@pytest.mark.parametrize("indices", [((17, 17), (0, 0)), ((0, 1), (0, 0)), ((0, 0), (0, 17))])
def test_indirect_sampling_validates_live_index_values_before_access(indices):
  instruction = next(ins for ins in decode(compile_image_program(textures=17).image) if "INDICES" in ins.operands)
  texture = Image2D(0x10000, 4, 3, 64, 4)
  # Empty storage makes an accidental pixel access fail independently. The
  # required error concerns the live resource indices, before any such access.
  group = Workgroup(b"", (2, 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, Memory({}), 0, 0,
                    image_bindings=ImageBindings((texture,) * 17, (), 17))
  register = instruction.operands["INDICES"].index
  group.registers[1][register] = indices[0]
  group.registers[1][register + 1] = indices[1]
  with pytest.raises(RuntimeError, match="indices differ|index out of bounds"):
    group.image_instruction(instruction, np.array([0, 1]))


@pytest.mark.parametrize("operation,field,value,shift,width", [
  ("isam", "DST", 255, 32, 8),
  ("isam", "SRC1", 255, 1, 8),
  ("stib.b", "SRC1", 254, 32, 8),
  ("stib.b", "SRC2", 255, 24, 8),
])
def test_image_register_spans_are_checked_without_native_decoding(operation, field, value, shift, width):
  instruction = next(ins for ins in decode(compile_image_program().image) if ins.op == operation)
  raw = instruction.raw & ~(((1 << width) - 1) << shift) | value << shift
  with pytest.raises(RuntimeError, match="register span out of bounds"):
    _image_instruction({**instruction.fields, field: value}, raw)


@pytest.mark.parametrize("operation,bit", [("isam", 0), ("isam", 48), ("isam", 52), ("stib.b", 8), ("stib.b", 23),
                                         ("ldib.b", 8), ("ldib.b", 23)])
def test_unimplemented_image_modes_are_rejected_in_python(operation, bit):
  instruction = next(ins for ins in decode(compile_image_program(coherent=operation == "ldib.b").image) if ins.op == operation)
  # A pure Python schema check covers unknown mode flags without feeding altered
  # bytes to the native ISA decoder or touching a host allocation.
  with pytest.raises(RuntimeError, match="unsupported image.*mode"):
    _image_instruction(instruction.fields, instruction.raw ^ (1 << bit))


@pytest.mark.parametrize("coordinates", [(0, 0), (-1, 0), (4, 0), (0, -1), (0, 3)])
def test_coherent_load_uses_the_writable_table_and_zero_for_invalid_texels(coordinates):
  instruction = next(ins for ins in decode(compile_image_program(coherent=True).image) if ins.op == "ldib.b")
  values = np.arange(48, dtype=np.float32).reshape(3, 4, 4) + 1
  image, storage = image_storage(values, 0x10000)
  memory = Memory({image.base:storage})
  group = Workgroup(b"", (1, 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, memory, 0, 0,
                    image_bindings=ImageBindings(outputs=(image,)))
  register = instruction.operands["COORD"].index
  group.registers[0][register:register + 2, 0] = np.array(coordinates, np.int32).view(np.uint32)
  group.image_instruction(instruction, np.array([0]))
  destination = instruction.operands["DST"].index
  actual = group.registers[0][destination:destination + 4, 0].view(np.float32)
  np.testing.assert_array_equal(actual, [1, 2, 3, 4] if coordinates == (0, 0) else [0, 0, 0, 0])
  assert memory.read(image.base, len(storage)) == storage


def test_invalid_store_coordinates_leave_every_registered_pixel_unchanged():
  # ARB_shader_image_load_store defines invalid-texel stores as having no
  # effect. The image is still a valid, writable registered resource.
  instruction = next(ins for ins in decode(compile_image_program().image) if ins.op == "stib.b")
  values = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
  image, storage = image_storage(values, 0x10000)
  memory = Memory({image.base:storage})
  group = Workgroup(b"", (2, 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, memory, 0, 0,
                    image_bindings=ImageBindings(outputs=(image,)))
  register = instruction.operands["COORD"].index
  group.registers[0][register, :] = np.array([-1, 4], np.int32).view(np.uint32)
  group.image_instruction(instruction, np.array([0, 1]))
  assert memory.read(image.base, len(storage)) == storage


def test_image_store_filters_invalid_lanes_and_preserves_valid_lane_values():
  instruction = next(ins for ins in decode(compile_image_program().image) if ins.op == "stib.b")
  values = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
  image, storage = image_storage(values, 0x10000)
  memory = Memory({image.base:storage})
  group = Workgroup(b"", (3, 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, memory, 0, 0,
                    image_bindings=ImageBindings(outputs=(image,)))
  data = instruction.operands["DATA"].index
  coordinate = instruction.operands["COORD"].index
  payload = np.array([[100, 200, 300], [101, 201, 301], [102, 202, 302], [103, 203, 303]], np.float32)
  group.registers[0][data:data + 4] = payload.view(np.uint32)
  group.registers[0][coordinate] = np.array([-1, 1, 4], np.int32).view(np.uint32)
  group.registers[0][coordinate + 1] = 0
  group.image_instruction(instruction, np.array([0, 1, 2]))
  expected = values.copy()
  expected[0, 1] = [200, 201, 202, 203]
  actual = np.frombuffer(memory.read(image.base, 192), np.float32).reshape(3, 4, 4)
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('readonly,access', [(None, mesa.ACCESS_CAN_REORDER), (True, mesa.ACCESS_CAN_REORDER), (False, 0)])
def test_image_load_fixture_keyword_preserves_the_existing_renderer_default(readonly, access):
  # The fixture may request a coherent load without changing any renderer call:
  # omitted and explicit-True permissions must remain identical to the base.
  renderer = IR3Renderer(Target(device='QCOM', renderer='IR3', arch='a630'))
  renderer.prerender([])
  builder = renderer.b
  try:
    zero = nimm(builder, 0, dtypes.int)
    value = _nload_img(builder, zero, zero, zero, dtypes.float, **({} if readonly is None else {'readonly': readonly}))
    instruction = ctypes.cast(value.parent_instr, ctypes.POINTER(mesa.nir_intrinsic_instr)).contents
    info = mesa.nir_intrinsic_infos[instruction.intrinsic]
    assert instruction.const_index[info.index_map[mesa.NIR_INTRINSIC_ACCESS] - 1] == access
    assert instruction.const_index[info.index_map[mesa.NIR_INTRINSIC_IMAGE_DIM] - 1] == mesa.GLSL_SAMPLER_DIM_2D
    assert instruction.const_index[info.index_map[mesa.NIR_INTRINSIC_DEST_TYPE] - 1] == mesa.nir_type_float32
    assert value.num_components == 4 and value.bit_size == 32
  finally:
    mesa.ralloc_free(builder.shader)
