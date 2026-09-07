"""PM4 folding contracts: exact admission policy and a typed memory-effect carrier."""
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import struct
import unittest
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import qreg
from test.mockgpu.qcom import qcomgpu as pm4
from test.mockgpu.qcom.test_pm4 import Memory, packet, signal, launch_words


class TestPM4Folding(unittest.TestCase):
  def memory_packets(self, memory):
    address = memory.base
    wait = packet(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
                  address & 0xffffffff, address >> 32, 7, 0xffffffff, 32)
    timestamp = packet(mesa.CP_REG_TO_MEM, qreg.cp_reg_to_mem_0(reg=mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER, cnt=2, _64b=True),
                       (address + 8) & 0xffffffff, (address + 8) >> 32)
    return signal(address, 7), wait, timestamp

  def test_memory_actions_share_named_effect_contract(self):
    # One typed carrier makes the operation, address, operand, and footprint inspectable.
    memory = Memory()
    words = sum(self.memory_packets(memory), ())
    actions = pm4.compile_plan(words, memory)
    self.assertTrue(hasattr(pm4, 'MemoryAction'), 'scalar PM4 effects need a common typed carrier')
    memory_actions = [action for action in actions if isinstance(action, pm4.MemoryAction)]
    self.assertEqual(len(memory_actions), len(actions))
    self.assertEqual([(action.operation, action.address, action.value, action.size) for action in memory_actions],
                     [('store', memory.base, 7, 4), ('wait', memory.base, 7, 4), ('timestamp', memory.base + 8, 0, 8)])
    with self.assertRaises(FrozenInstanceError): setattr(actions[0], 'address', 0)
    self.assertEqual(memory.data, bytes(64))
    with patch('test.mockgpu.qcom.qcomgpu.time.perf_counter_ns', return_value=10000) as clock:
      self.assertEqual(pm4.execute_command(words, memory, 999), (True, 3))
    self.assertEqual(clock.call_count, 1)
    self.assertEqual(struct.unpack('<I4xQ', memory.read(memory.base, 16)), (7, 192))

  def test_each_memory_effect_is_prevalidated_with_its_full_access(self):
    # A valid prefix cannot publish when any later effect lacks its exact access footprint.
    for denied_operation, denied_write, denied_size in (('store', True, 4), ('wait', False, 4), ('timestamp', True, 8)):
      memory = Memory()
      store, wait, timestamp = self.memory_packets(memory)
      words = {'store':store, 'wait':wait, 'timestamp':timestamp}[denied_operation]
      original = memory.validate
      def reject_selected(address, size, write=False):
        if address != memory.base + 24 and size == denied_size and write == denied_write: raise ValueError('effect access denied')
        return original(address, size, write)
      with patch.object(memory, 'validate', side_effect=reject_selected):
        with self.subTest(operation=denied_operation), self.assertRaisesRegex(pm4.PM4Fault, 'effect access denied'):
          pm4.execute_command(signal(memory.base + 24, 123) + words, memory, 1)
      self.assertEqual(memory.data, bytes(64))

  def test_wait_resume_does_not_resample_a_completed_timestamp(self):
    memory = Memory()
    store, wait, timestamp = self.memory_packets(memory)
    with patch('test.mockgpu.qcom.qcomgpu.time.perf_counter_ns', return_value=10000) as clock:
      self.assertEqual(pm4.execute_command(timestamp + wait + store, memory, 1), (False, 1))
      memory.write(memory.base, struct.pack('<I', 7))
      self.assertEqual(pm4.execute_command(timestamp + wait + store, memory, 1, start=1), (True, 3))
    self.assertEqual(clock.call_count, 1)

  def test_exact_register_whitelist_and_masks(self):
    # Independent baseline values prevent grouped policy from admitting neighbors or extra bits.
    fixed = {
      0xa9b2:(0,), 0xa9ba:(0x80,), 0xaa00:(0x40,), 0xab00:(5,2), 0xae0f:(0x20,),
      0xb309:(2,1), 0xb600:(0,), 0xb983:(0xfcfcfcfc,), 0xb984:(0xfcfcfcfc,), 0xb985:(0xfcfcfcfc,), 0xb986:(0xfc,),
      0xb992:(0,), 0xb994:(0,), 0xb996:(0,), 0xbb08:(0, 0x60),
    }
    masks = {
      0xa9b0:0x1fdffe, 0xa9b1:0x7f, 0xa9b3:0xffffffff, 0xa9b4:0xffffffff, 0xa9b5:0xffffffff, 0xa9b6:0xff, 0xa9b7:0xffffffff,
      0xa9b8:0xffffffff, 0xa9b9:0x3ffff, 0xa9bc:0xffffffff, 0xa9bd:0x7ffff, 0xb987:0x1ff, 0xb990:0xffffffff,
      0xb991:0xffffffff, 0xb993:0xffffffff, 0xb995:0xffffffff, 0xb997:0xffffffff, 0xb998:0x2ff,
      0xb999:0xffffffff, 0xb99a:0xffffffff, 0xb99b:0xffffffff,
      # A6xx compute resource counts and their optional 64-bit table bases.
      # Bindless flags (bits 0-3 of SP_CS_CONFIG) remain outside this contract.
      0xa9bb:0x1fffff00, 0xa9e2:0xffffffff, 0xa9e3:0xffffffff, 0xa9e6:0xffffffff, 0xa9e7:0xffffffff,
      0xa9f2:0xffffffff, 0xa9f3:0xffffffff, 0xb180:0xffffffff, 0xb181:0xffffffff,
    }
    self.assertEqual(pm4.FIXED_REGISTERS, fixed)
    self.assertEqual(pm4.REGISTER_MASKS, masks)
    self.assertEqual(set(pm4.FIXED_REGISTERS) & set(pm4.REGISTER_MASKS), set())
    allowed = set(pm4.FIXED_REGISTERS) | set(pm4.REGISTER_MASKS)
    for address in range(0xa000, 0xbc00):
      if address in allowed: continue
      with self.subTest(address=hex(address)), self.assertRaises(pm4.PM4Fault): pm4.register_write({}, address, 0)
    for address, mask in pm4.REGISTER_MASKS.items():
      for bit in range(32):
        registers:dict[int, int] = {}
        if mask & (1 << bit):
          pm4.register_write(registers, address, 1 << bit)
          self.assertEqual(registers, {address:1 << bit})
        else:
          with self.assertRaises(pm4.PM4Fault): pm4.register_write(registers, address, 1 << bit)
          self.assertEqual(registers, {})

  def test_packet_shapes_remain_exact(self):
    # Structural matching must retain every prior payload-length boundary before effects.
    memory = Memory(65536)
    state = launch_words(memory)[:-5]
    store, wait, timestamp = self.memory_packets(memory)
    cases = [packet(mesa.CP_WAIT_FOR_IDLE), packet(mesa.CP_WAIT_MEM_WRITES), packet(mesa.CP_SET_MARKER, mesa.RM6_COMPUTE),
             packet(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE), store, wait, timestamp, packet(mesa.CP_EXEC_CS, 0, 1, 1, 1),
             packet(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                    state_block=mesa.SB6_CS_SHADER, num_unit=1), (memory.base + 4096) & 0xffffffff, (memory.base + 4096) >> 32)]
    for encoded in cases:
      parsed, = pm4.decode_packets(encoded)
      pm4.compile_plan(state + encoded, memory)
      malformed = [parsed.words[:size] for size in range(len(parsed.words))] + [parsed.words + (0,)]
      for payload in malformed:
        with self.subTest(opcode=parsed.target, payload=payload), self.assertRaises(pm4.PM4Fault):
          pm4.execute_command(signal(memory.base, 123) + state + packet(parsed.target, *payload), memory, 1)
        self.assertEqual(memory.read(memory.base, 4), bytes(4))

  def test_launch_keeps_its_explicit_observation_fields(self):
    memory = Memory(65536)
    action, = pm4.compile_plan(launch_words(memory), memory)
    assert isinstance(action, pm4.Launch)
    self.assertEqual((action.image_address, action.constants_address, action.constants_size),
                     (memory.base + 4096, memory.base + 8192, 2048))
    self.assertEqual((action.groups, action.local, action.wgid, action.lid, action.wgsize, action.shared_bytes, action.private_bytes),
                     ((1, 1, 1), (1, 1, 1), 0xfc, 0xfc, 0xfc, 2048, 0))

  def test_malformed_payload_diagnostics_are_preserved(self):
    cases = (
      (mesa.CP_WAIT_FOR_IDLE, (1,), 'invalid wait packet length'),
      (mesa.CP_WAIT_MEM_WRITES, (1,), 'invalid wait packet length'),
      (mesa.CP_SET_MARKER, (0,), 'unsupported PM4 marker mode'),
      (mesa.CP_EVENT_WRITE, (), 'unsupported PM4 event write'),
      (mesa.CP_WAIT_REG_MEM, (), 'unsupported PM4 memory wait'),
      (mesa.CP_REG_TO_MEM, (), 'unsupported PM4 register-to-memory transfer'),
      (mesa.CP_LOAD_STATE6_FRAG, (), 'unsupported PM4 state load length'),
      (mesa.CP_EXEC_CS, (), 'unsupported PM4 compute dispatch'),
      (0x7f, (), 'unsupported PM4 opcode 0x7f'),
    )
    for opcode, payload, expected in cases:
      memory = Memory()
      with self.subTest(opcode=opcode), self.assertRaises(pm4.PM4Fault) as raised:
        pm4.execute_command(signal(memory.base, 123) + packet(opcode, *payload), memory, 1)
      self.assertEqual(str(raised.exception), expected)
      self.assertEqual(memory.data, bytes(64))


if __name__ == '__main__': unittest.main()
