import ctypes, mmap, os, pathlib, struct, subprocess, sys, tempfile
import unittest
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import pkt4_hdr, pkt7_hdr, qreg


class Memory:
  def __init__(self, size=64):
    self.base = 0x123400001000
    self.data = bytearray(size)

  def read(self, address, size):
    offset = address - self.base
    if offset < 0 or offset + size > len(self.data):
      raise ValueError("outside test allocation")
    return bytes(self.data[offset:offset + size])

  def validate(self, address, size, write=False): self.read(address, size)

  def region(self, address, size):
    self.read(address, size)
    return self.base, self.data

  def write(self, address, data):
    self.read(address, len(data))
    offset = address - self.base
    self.data[offset:offset + len(data)] = data


def packet(opcode, *payload):
  return (pkt7_hdr(opcode, len(payload)), *payload)


def signal(address, value):
  return packet(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, address & 0xffffffff, address >> 32, value)


class TestPM4(unittest.TestCase):
  def test_relocated_signal_and_full_word_wait(self):
    from test.mockgpu.qcom.qcomgpu import execute_command
    memory = Memory()
    address = memory.base + 12
    words = signal(address, 0x12340007) + packet(mesa.CP_WAIT_REG_MEM,
      qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
      address & 0xffffffff, address >> 32, 5, 0xffffffff, 32)
    self.assertEqual(execute_command(words, memory, 1), (True, 2))
    self.assertEqual(memory.read(address, 4), struct.pack('<I', 0x12340007))

  def test_unsatisfied_wait_does_not_execute_tail(self):
    from test.mockgpu.qcom.qcomgpu import execute_command
    memory = Memory()
    words = packet(mesa.CP_WAIT_REG_MEM,
      qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
      memory.base & 0xffffffff, memory.base >> 32, 1, 0xffffffff, 32) + signal(memory.base + 8, 123)
    self.assertEqual(execute_command(words, memory, 1), (False, 0))
    self.assertEqual(memory.data, bytes(64))

  def test_validate_whole_stream_before_any_effect(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    memory = Memory()
    with self.assertRaisesRegex(PM4Fault, "opcode"):
      execute_command(signal(memory.base, 77) + packet(0x7f), memory, 1)
    self.assertEqual(memory.data, bytes(64))

  def test_parity_and_reserved_bits(self):
    from test.mockgpu.qcom.qcomgpu import decode_packets, PM4Fault
    valid = signal(0x100001000, 9)
    for bit in (14, 15, 23, 24, 27):
      with self.subTest(bit=bit), self.assertRaises(PM4Fault):
        decode_packets((valid[0] ^ (1 << bit), *valid[1:]))
    type4 = (pkt4_hdr(mesa.REG_A6XX_SP_CS_CONFIG, 1), mesa.A6XX_SP_CS_CONFIG_ENABLED)
    for bit in (7, 26, 27):
      with self.subTest(type4_bit=bit), self.assertRaises(PM4Fault):
        decode_packets((type4[0] ^ (1 << bit), type4[1]))

  def test_truncation_and_packet_extent(self):
    from test.mockgpu.qcom.qcomgpu import decode_packets, PM4Fault
    valid = signal(0x1000, 9)
    for count in range(1, len(valid)):
      with self.subTest(count=count), self.assertRaises(PM4Fault):
        decode_packets(valid[:count])
    with self.assertRaises(PM4Fault):
      decode_packets((pkt4_hdr(0x3ffff, 2), 0, 0))

  def test_unknown_register_and_dispatch_without_state(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    for words in ((pkt4_hdr(1, 1), 0), packet(mesa.CP_EXEC_CS, 0, 1, 1, 1)):
      with self.subTest(words=words), self.assertRaises(PM4Fault):
        execute_command(words, Memory(), 1)

  def test_signal_alignment_and_unsupported_wait(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    memory = Memory()
    with self.assertRaises(PM4Fault):
      execute_command(signal(memory.base + 1, 7), memory, 1)
    words = packet(mesa.CP_WAIT_REG_MEM, 0, memory.base & 0xffffffff, memory.base >> 32, 0, 0xffffffff, 32)
    with self.assertRaises(PM4Fault):
      execute_command(words, memory, 1)

  def test_partial_wait_mask_is_explicitly_unsupported(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    memory = Memory()
    words = packet(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
                   memory.base & 0xffffffff, memory.base >> 32, 5, 0xff, 32)
    with self.assertRaises(PM4Fault): execute_command(words, memory, 1)

  def test_signal_prefix_then_wait_resumes_without_repeating_prefix(self):
    from test.mockgpu.qcom.qcomgpu import execute_command
    memory = Memory()
    words = signal(memory.base, 7) + packet(mesa.CP_WAIT_REG_MEM,
      qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
      (memory.base + 4) & 0xffffffff, (memory.base + 4) >> 32, 1, 0xffffffff, 32) + signal(memory.base + 8, 9)
    self.assertEqual(execute_command(words, memory, 1), (False, 1))
    memory.write(memory.base, struct.pack('<I', 8))
    memory.write(memory.base + 4, struct.pack('<I', 1))
    self.assertEqual(execute_command(words, memory, 1, start=1), (True, 3))
    self.assertEqual(struct.unpack('<3I', memory.read(memory.base, 12)), (8, 1, 9))

  def test_late_unmapped_signal_does_not_publish_valid_prefix(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    memory = Memory()
    with self.assertRaises(PM4Fault): execute_command(signal(memory.base, 7) + signal(memory.base + 64, 9), memory, 1)
    self.assertEqual(memory.data, bytes(64))

  def test_each_timestamp_samples_its_action_time(self):
    from test.mockgpu.qcom.qcomgpu import execute_command
    from unittest.mock import patch
    memory = Memory()
    control = qreg.cp_reg_to_mem_0(reg=mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER, cnt=2, _64b=True)
    words = packet(mesa.CP_REG_TO_MEM, control, memory.base & 0xffffffff, memory.base >> 32)
    words += packet(mesa.CP_REG_TO_MEM, control, (memory.base + 8) & 0xffffffff, (memory.base + 8) >> 32)
    with patch('test.mockgpu.qcom.qcomgpu.time.perf_counter_ns', side_effect=(10000, 15000)):
      self.assertEqual(execute_command(words, memory, 999), (True, 2))
    self.assertEqual(struct.unpack('<2Q', memory.read(memory.base, 16)), (192, 288))


ADD = (0x202cc00000000002, 0x202cc00100000003, 0x202cc00300000004, 0x202cc00400000005,
       0x202cc00500000000, 0x202cc00600000001, 0xc006000201800001, 0xc00600070180c001,
       0x5218080200070002, 0xc0c60b0001800004, 0x0300000000000000)


def launch_words(memory, groups=(1, 1, 1), local=(1, 1, 1), image=None, base=None):
  # This fixture uses the current producer's named registers and a real IR3 add.
  if base is None: base = memory.base
  image_address, constants_address, private_address = base + 4096, base + 8192, base + 16384
  memory.write(image_address, image if image is not None else struct.pack('<11Q', *ADD).ljust(128, b'\0'))
  memory.write(constants_address, struct.pack('<3Q', base + 16, base + 20, base + 24))
  memory.write(base + 20, struct.pack('<2I', 7, 19))
  words = packet(mesa.CP_SET_MARKER, mesa.RM6_COMPUTE)
  def reg(address, *values):
    nonlocal words
    words += (pkt4_hdr(address, len(values)), *values)
  reg(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True))
  reg(mesa.REG_A6XX_SP_UPDATE_CNTL, 0)
  reg(mesa.REG_A6XX_SP_CS_TSIZE, 0x80)
  reg(mesa.REG_A6XX_SP_CS_USIZE, 0x40)
  reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL, constant_demotion_enable=True))
  reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, mesa.A6XX_SP_PERFCTR_SHADER_MASK_CS)
  reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL))
  reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
  reg(mesa.REG_A6XX_SP_CS_NDRANGE_0,
      qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local[0]-1, localsizey=local[1]-1, localsizez=local[2]-1),
      groups[0]*local[0], 0, groups[1]*local[1], 0, groups[2]*local[2], 0, 0xccc0cf,
      0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=mesa.THREAD64), *groups)
  reg(mesa.REG_A6XX_SP_CS_CNTL_0, qreg.a6xx_sp_cs_cntl_0(threadsize=mesa.THREAD64, fullregfootprint=2),
      qreg.a6xx_sp_cs_cntl_1(constantrammode=mesa.CONSTLEN_256, shared_size=1), 0, 0,
      image_address & 0xffffffff, image_address >> 32, 0, private_address & 0xffffffff, private_address >> 32, 0)
  words += packet(mesa.CP_LOAD_STATE6_FRAG,
                  qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                       state_block=mesa.SB6_CS_SHADER, num_unit=128), constants_address & 0xffffffff, constants_address >> 32)
  words += packet(mesa.CP_LOAD_STATE6_FRAG,
                  qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                       state_block=mesa.SB6_CS_SHADER, num_unit=1), image_address & 0xffffffff, image_address >> 32)
  reg(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc,
      qreg.a6xx_sp_cs_const_config(constlen=64, enabled=True)) # Four vec4 constants per encoded unit.
  reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, 2) # 4096 bytes per SP, encoded in 2 KiB units
  reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1)
  reg(mesa.REG_A6XX_SP_CS_CONFIG, mesa.A6XX_SP_CS_CONFIG_ENABLED)
  reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
      qreg.a6xx_sp_cs_const_config_0(wgidconstid=0xfc, wgsizeconstid=0xfc, wgoffsetconstid=0xfc, localidregid=0xfc),
      qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64))
  return words + packet(mesa.CP_EXEC_CS, 0, *groups)


class TestPM4Launch(unittest.TestCase):
  def test_shared_memory_encoding_covers_zero_and_every_nonzero_value(self):
    from test.mockgpu.qcom.qcomgpu import compile_plan, Launch
    expected = (32768, 2048, 3072, 4096, 5120, 6144, 7168, 8192,
                9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384,
                17408, 18432, 19456, 20480, 21504, 22528, 23552, 24576,
                25600, 26624, 27648, 28672, 29696, 30720, 31744, 32768)
    for encoded, byte_size in enumerate(expected):
      memory = Memory(65536)
      words = launch_words(memory)
      control = qreg.a6xx_sp_cs_cntl_1(constantrammode=mesa.CONSTLEN_256, shared_size=encoded)
      words = words[:-5] + (pkt4_hdr(mesa.REG_A6XX_SP_CS_CNTL_1, 1), control) + words[-5:]
      action = next(action for action in compile_plan(words, memory) if isinstance(action, Launch))
      self.assertEqual(action.shared_bytes, byte_size, f"shared encoding {encoded}")

  def test_constant_configuration_bounds_uploaded_bytes(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    for config in (0x100, 0x110, 0x40): # Zero capacity, 1024-byte capacity, disabled bank.
      memory = Memory(65536)
      words = launch_words(memory)
      words = words[:-5] + (pkt4_hdr(mesa.REG_A6XX_SP_CS_CONST_CONFIG, 1), config) + words[-5:]
      with self.assertRaises(PM4Fault): execute_command(signal(memory.base, 1) + words, memory, 1)
      self.assertEqual(memory.read(memory.base, 4), b'\0' * 4)

  def test_stack_offset_units_cover_all_four_sp_allocations(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    memory = Memory(24576) # Private base + 8192 bytes; four 4096-byte SP slices do not fit.
    with self.assertRaises(PM4Fault): execute_command(launch_words(memory), memory, 1)

  def test_staged_shader_modification_is_rejected_without_host_commit(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    from test.mockgpu.qcom.qcomdriver import AddressSpace
    with mmap.mmap(-1, 65536) as host:
      base = ctypes.addressof(ctypes.c_char.from_buffer(host))
      space = AddressSpace()
      space.map(base, 65536, host)
      setup = space.transaction()
      words = launch_words(setup, base=base)
      setup.commit()
      before = bytes(host)
      with self.assertRaisesRegex(PM4Fault, "shader.*changed"):
        execute_command(signal(base + 4096, 0xffffffff) + words, space.transaction(), 1)
      self.assertEqual(bytes(host), before)

  def test_real_shader_launch_uses_relocated_constants_and_dimensions(self):
    from test.mockgpu.qcom.qcomgpu import execute_command
    for groups, local in (((1, 1, 1), (1, 1, 1)), ((2, 1, 1), (2, 1, 1))):
      memory = Memory(65536)
      words = launch_words(memory, groups, local)
      self.assertEqual(execute_command(words, memory, 1), (True, 1))
      self.assertEqual(struct.unpack('<I', memory.read(memory.base + 16, 4)), (26,))

  def test_shader_encoding_is_validated_before_prefix_signal(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    memory = Memory(65536)
    words = launch_words(memory, image=b'\xff' * 128)
    with self.assertRaises(PM4Fault): execute_command(signal(memory.base, 77) + words, memory, 1)
    self.assertEqual(memory.read(memory.base, 4), b'\0' * 4)

  def test_unsupported_image_config_and_dispatch_mismatch_are_rejected(self):
    from test.mockgpu.qcom.qcomgpu import execute_command, PM4Fault
    memory = Memory(65536)
    words = launch_words(memory)
    for corrupt in (words[:-4] + (0, 2, 1, 1),
                    words[:-5] + (pkt4_hdr(mesa.REG_A6XX_SP_CS_CONFIG, 1),
                                  mesa.A6XX_SP_CS_CONFIG_ENABLED | (1 << mesa.A6XX_SP_CS_CONFIG_NTEX__SHIFT)) + words[-5:]):
      with self.assertRaises(PM4Fault): execute_command(corrupt, memory, 1)
      self.assertEqual(memory.read(memory.base + 16, 4), b'\0' * 4)

  def test_qcom_producer_scales_private_stack_register_and_keeps_byte_allocation(self):
    # Generate a real NIR kernel; vary only its compiler-private-memory metadata
    # to cover the allocation unit boundary without relying on a chosen spill.
    code = '''import struct
from tinygrad import Device, Tensor
from tinygrad.engine.realize import lower_and_compile
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import QCOMComputeQueue
from tinygrad.runtime.support.hcq2 import EncodeCtx, make_submit
from tinygrad.uop.ops import Ops
from test.mockgpu.qcom.qcomgpu import decode_packets
device = Device["QCOM"]
linear, _ = Tensor.linear_with_vars(Tensor.ones(1).contiguous())
call = lower_and_compile(linear).src[0]
program = call.src[0]
for private_bytes, expected_field, expected_allocation in ((0, 2, 16384), (1, 512, 4194304),
                                                          (512, 512, 4194304), (513, 1024, 8388608)):
  variant = mesa.struct_ir3_shader_variant.from_buffer_copy(program.src[3].arg)
  variant.pvtmem_size = private_bytes
  binary = bytes(variant) + program.src[3].arg[len(bytes(variant)):]
  changed = program.replace(src=program.src[:3] + (program.src[3].replace(arg=binary),))
  current = call.replace(src=(changed, *call.src[1:]))
  queue = QCOMComputeQueue(EncodeCtx(("QCOM",)), make_submit(current, devs="QCOM", queue="COMPUTE:0"))
  queue.exec(current, changed)
  for packet in decode_packets(struct.unpack(f"<{len(queue.blob)//4}I", queue.blob)):
    config_index = mesa.REG_A6XX_SP_CS_CONST_CONFIG - packet.target
    if packet.kind == 4 and 0 <= config_index < len(packet.words):
      assert packet.words[config_index] == 0x140, ("constant config", packet.words[config_index])
    index = mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET - packet.target
    if packet.kind == 4 and 0 <= index < len(packet.words):
      assert packet.words[index] == expected_field, (private_bytes, packet.words[index], expected_field)
      break
  else: raise AssertionError("stack register was not emitted")
  requested = {node.max_numel() for _, value in queue.patches for node in value.toposort() if node.op is Ops.PARAM and node.tag == "stack"}
  assert requested == {expected_allocation}, (private_bytes, requested, expected_allocation)
  buffer = device._ensure_stack_size(requested.pop())
  assert buffer.nbytes == expected_allocation and buffer._buf.size == expected_allocation
print("private stack producer passed")
'''
    with tempfile.TemporaryDirectory() as cache:
      result = subprocess.run([sys.executable, "-c", code], cwd=pathlib.Path(__file__).parents[3],
                              env={**os.environ, "DEV":"MOCK+QCOM:IR3", "PARALLEL":"0", "XDG_CACHE_HOME":cache},
                              capture_output=True, text=True, timeout=20)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("private stack producer passed", result.stdout)


if __name__ == '__main__':
  unittest.main()
