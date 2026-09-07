"""PM4 image configuration and transaction boundaries over controlled bytes."""
import contextlib
import struct
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import pkt4_hdr, qreg
from test.mockgpu.qcom import qcomdriver, qcomgpu
from test.mockgpu.qcom.emu import decode, validate_image_bindings
from test.mockgpu.qcom.test_image_execution import compile_image_program
from test.mockgpu.qcom.test_pm4 import Memory, launch_words, packet, signal


def register(address, *values):
  return (pkt4_hdr(address, len(values)), *values)


def image_descriptor(address, texture=True, width=4, height=3, pitch=64, half=False):
  # Reproduce the producer's descriptor ABI, not the emulator's parser.
  fmt = mesa.FMT6_16_16_16_16_FLOAT if half else mesa.FMT6_32_32_32_32_FLOAT
  swizzle = dict(swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3) if texture else {}
  words = [
    qreg.a6xx_tex_const_0(8 if texture else 0, fmt=fmt, **swizzle),
    qreg.a6xx_tex_const_1(width=width, height=height),
    qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=pitch, pitchalign=(pitch & -pitch).bit_length() - 7),
    0, address & 0xffffffff, address >> 32, qreg.a6xx_tex_const_6(plane_pitch=0x400000), 13,
    0, 0, 0, 0, 0, 0, 0, 0,
  ]
  return struct.pack("<16I", *words)


def image_command(memory, count=1):
  program = compile_image_program(textures=count)
  base = memory.base
  textures, outputs, samplers, border = (base + offset for offset in (0x9000, 0xa000, 0xa100, 0xb000))
  words = launch_words(memory, local=program.local, image=program.image)[:-5]
  memory.write(base + 8192, program.constants.ljust(2048, b"\0"))
  units = len(program.image) // 128
  words += register(mesa.REG_A6XX_SP_CS_INSTR_SIZE, units)
  words += packet(mesa.CP_LOAD_STATE6_FRAG,
                  qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                       state_block=mesa.SB6_CS_SHADER, num_unit=units),
                  (base + 4096) & 0xffffffff, (base + 4096) >> 32)
  words += register(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
                    qreg.a6xx_sp_cs_const_config_0(wgidconstid=program.wgid, localidregid=program.lid,
                                                 wgsizeconstid=program.wgsize, wgoffsetconstid=0xfc))
  pixels = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
  for index in range(count):
    address = base + 0x10000 + index * 0x1000
    memory.write(address, (pixels + index).tobytes())
    memory.write(textures + index * 64, image_descriptor(address))
  memory.write(base + 0x30000, np.full((3, 4, 4), -77, np.float32).tobytes())
  memory.write(outputs, image_descriptor(base + 0x30000, texture=False))
  sampler = struct.pack("<4I", qreg.a6xx_tex_samp_0(wrap_s=3, wrap_t=3, wrap_r=3),
                        qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0)
  memory.write(samplers, sampler * count)
  words += register(mesa.REG_A6XX_SP_CS_CONFIG, qreg.a6xx_sp_cs_config(enabled=True, ntex=count, nsamp=count, nuav=1))
  for block, kind, address, load_count, target in (
    (mesa.SB6_CS_TEX, mesa.ST_CONSTANTS, textures, min(count, 16), mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE),
    (mesa.SB6_CS_TEX, mesa.ST_SHADER, samplers, count, mesa.REG_A6XX_SP_CS_SAMPLER_BASE),
    (mesa.SB6_CS_SHADER, mesa.ST6_UAV, outputs, 1, mesa.REG_A6XX_SP_CS_UAV_BASE),
  ):
    words += packet(mesa.CP_LOAD_STATE6_FRAG,
                    qreg.cp_load_state6_0(state_type=kind, state_src=mesa.SS6_INDIRECT, state_block=block, num_unit=load_count),
                    address & 0xffffffff, address >> 32)
    words += register(target, address & 0xffffffff, address >> 32)
  words += register(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, border & 0xffffffff, border >> 32)
  return words + packet(mesa.CP_EXEC_CS, 0, 1, 1, 1)


@contextlib.contextmanager
def driver_for(memory, command):
  # Replace only native byte transport. Mapping identity, transactions, PM4,
  # real shader decoding, image execution and the queue scheduler remain real.
  with patch.object(qcomdriver, "read_host", side_effect=memory.read), \
       patch.object(qcomdriver, "write_host", side_effect=memory.write), \
       patch.object(qcomdriver, "validate_host_mapping", side_effect=memory.validate):
    driver = qcomdriver.QCOMDriver()
    driver.memory.map(memory.base, len(memory.data), owner=memory)
    context = driver.contexts[1] = qcomdriver.Context(1, queued=1)
    context.pending.append((command, 1, 0))
    yield driver, context


def output_pixels(memory):
  return np.frombuffer(memory.read(memory.base + 0x30000, 192), np.float32).reshape(3, 4, 4)


@pytest.mark.parametrize("count", [1, 2, 17])
def test_pm4_uses_full_descriptor_tables_and_distinct_upload_kinds(count):
  memory = Memory(0x40000)
  words = image_command(memory, count)
  with driver_for(memory, words) as (driver, context):
    driver.drain()
    assert not context.pending and context.retired == 1
  pixels = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
  np.testing.assert_array_equal(output_pixels(memory), pixels * count + count * (count - 1) // 2)


@pytest.mark.parametrize("offset,value", [
  (0x9000, 0),                       # Unsupported texture format.
  (0x9004, 3 << 15),                 # Zero width.
  (0x9008, 1 << 29 | 32 << 7),       # Row pitch cannot hold this image.
  (0x900c, 1),                       # Unsupported layer pitch.
  (0xa100, 0),                       # Sampler changes from border to repeat.
  (0xb000, 1),                       # Nonblack border configuration.
  (0xa010, 0xfffffff0),              # Output span does not belong to memory.
])
def test_invalid_late_image_configuration_rejects_before_prefix(offset, value):
  memory = Memory(0x40000)
  words = image_command(memory)
  memory.write(memory.base + offset, struct.pack("<I", value))
  command = signal(memory.base, 123) + words
  before = bytes(memory.data)
  with driver_for(memory, command) as (driver, context):
    with pytest.raises(qcomgpu.PM4Fault):
      driver.drain()
    assert bytes(memory.data) == before
    assert context.pending[0][2] == 0 and context.consumed == context.retired == 0


def test_staged_descriptor_change_does_not_replace_prevalidated_configuration():
  memory = Memory(0x40000)
  words = image_command(memory)
  # The altered width is itself valid, but belongs to a different configuration
  # than the one admitted before the preceding store action ran.
  command = signal(memory.base + 0x9004, 3 << 15 | 3) + words
  before = bytes(memory.data)
  with driver_for(memory, command) as (driver, _):
    with pytest.raises(qcomgpu.PM4Fault, match="image configuration changed"):
      driver.drain()
  assert bytes(memory.data) == before


def test_short_image_backing_rejects_before_any_action():
  memory = Memory(0x40000)
  words = image_command(memory)
  memory.write(memory.base + 0xa000, image_descriptor(memory.base + len(memory.data) - 64, texture=False))
  before = bytes(memory.data)
  with pytest.raises(qcomgpu.PM4Fault):
    qcomgpu.execute_command(signal(memory.base, 123) + words, memory, 1)
  assert bytes(memory.data) == before


def test_read_only_image_output_rejects_before_any_action():
  class ReadOnlyOutputMemory(Memory):
    def validate(self, address, size, write=False):
      super().validate(address, size, write)
      output = self.base + 0x30000
      if write and address < output + 192 and output < address + size:
        raise RuntimeError("read-only image output")

  memory = ReadOnlyOutputMemory(0x40000)
  words = image_command(memory)
  before = bytes(memory.data)
  with pytest.raises(qcomgpu.PM4Fault, match="read-only image output"):
    qcomgpu.execute_command(signal(memory.base, 123) + words, memory, 1)
  assert bytes(memory.data) == before


@pytest.mark.parametrize("replacement", [
  lambda base: register(mesa.REG_A6XX_SP_CS_CONFIG, qreg.a6xx_sp_cs_config(enabled=True, ntex=2, nsamp=1, nuav=1)),
  lambda base: register(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, (base + 0x9040) & 0xffffffff, (base + 0x9040) >> 32),
])
def test_resource_count_or_base_mismatch_rejects_before_any_action(replacement):
  memory = Memory(0x40000)
  words = image_command(memory)
  words = words[:-5] + replacement(memory.base) + words[-5:]
  before = bytes(memory.data)
  with pytest.raises(qcomgpu.PM4Fault, match="resource load and configuration disagree"):
    qcomgpu.execute_command(signal(memory.base, 123) + words, memory, 1)
  assert bytes(memory.data) == before


def test_resource_after_the_texture_prefetch_limit_is_also_prevalidated():
  memory = Memory(0x40000)
  words = image_command(memory, count=17)
  memory.write(memory.base + 0x9000 + 16 * 64 + 4, struct.pack("<I", 0))
  before = bytes(memory.data)
  with pytest.raises(qcomgpu.PM4Fault):
    qcomgpu.execute_command(signal(memory.base, 123) + words, memory, 1)
  assert bytes(memory.data) == before


def command_with_reserved_sampler(memory):
  words = image_command(memory)
  address = memory.base + 0xa100
  extra = register(mesa.REG_A6XX_SP_CS_CONFIG, qreg.a6xx_sp_cs_config(enabled=True, ntex=1, nsamp=2, nuav=1))
  extra += packet(mesa.CP_LOAD_STATE6_FRAG,
                  qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                       state_block=mesa.SB6_CS_TEX, num_unit=2), address & 0xffffffff, address >> 32)
  return words[:-5] + extra + words[-5:]


def test_unused_zero_sampler_slot_matches_the_qcomcl_producer():
  # QCOMProgramData._parse_lib pads its one real sampler with one zero record.
  # This slot is configuration padding, not an executable repeat-mode sampler.
  memory = Memory(0x40000)
  words = command_with_reserved_sampler(memory)
  with driver_for(memory, words) as (driver, _):
    driver.drain()
  np.testing.assert_array_equal(output_pixels(memory), np.arange(48, dtype=np.float32).reshape(3, 4, 4))


def test_reserved_sampler_slot_cannot_be_used_by_an_instruction():
  memory = Memory(0x40000)
  launch, = qcomgpu.compile_plan(command_with_reserved_sampler(memory), memory)
  assert isinstance(launch, qcomgpu.Launch)
  instruction = next(ins for ins in decode(launch.image) if ins.op == "isam")
  changed = replace(instruction, fields={**instruction.fields, "SAMP": 1})
  with pytest.raises(RuntimeError, match="sampler index out of bounds"):
    validate_image_bindings((changed,), launch.image_bindings)


def test_staged_pixel_change_is_visible_to_image_execution():
  memory = Memory(0x40000)
  words = image_command(memory)
  command = signal(memory.base + 0x10000, 0x42280000) + words  # float32 42
  with driver_for(memory, command) as (driver, _):
    driver.drain()
  expected = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
  expected[0, 0, 0] = 42
  np.testing.assert_array_equal(output_pixels(memory), expected)


def test_wait_resume_rechecks_images_without_repeating_the_published_prefix():
  memory = Memory(0x40000)
  words = image_command(memory)
  address = memory.base + 4
  wait = packet(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
                address & 0xffffffff, address >> 32, 1, 0xffffffff, 32)
  command = signal(memory.base, 123) + wait + words
  with driver_for(memory, command) as (driver, context):
    driver.drain()
    assert context.pending[0][2] == 1 and context.retired == 0
    assert memory.read(memory.base, 4) == struct.pack("<I", 123)
    memory.write(memory.base, struct.pack("<2I", 456, 1))
    memory.write(memory.base + 0x10000, struct.pack("<f", 99))
    driver.drain()
    assert memory.read(memory.base, 4) == struct.pack("<I", 456)
    assert not context.pending and context.retired == 1
  expected = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
  expected[0, 0, 0] = 99
  np.testing.assert_array_equal(output_pixels(memory), expected)
