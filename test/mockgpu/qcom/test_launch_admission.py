"""A630-2: static launch admission must precede even a resumable prefix.

All shader decoding and host-memory boundaries are inert in these tests.
"""
import contextlib, unittest
from unittest.mock import patch
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import pkt4_hdr, qreg
from test.mockgpu.qcom import emu, qcomdriver, qcomgpu
from test.mockgpu.qcom.test_pm4 import Memory, launch_words, packet, signal


def invocation_launch(memory, role, index):
  words = launch_words(memory)
  config = dict(wgidconstid=0xfc, localidregid=0xfc, wgsizeconstid=0xfc, wgoffsetconstid=0xfc)
  config[role] = index
  # Override the named ABI register immediately before the dispatch packet.
  return words[:-5] + (pkt4_hdr(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 1), qreg.a6xx_sp_cs_const_config_0(**config)) + words[-5:]


def wait_for(memory):
  address = memory.base + 4
  return packet(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),
                address & 0xffffffff, address >> 32, 1, 0xffffffff, 32)


@contextlib.contextmanager
def inert_driver(memory, command):
  with patch.object(emu, 'decode', return_value=(emu.Instruction('end', {}, {}, 0),)), \
       patch.object(qcomdriver, 'read_host', side_effect=memory.read), \
       patch.object(qcomdriver, 'write_host', side_effect=memory.write) as publish, \
       patch.object(qcomdriver, 'validate_host_mapping', side_effect=memory.validate):
    driver = qcomdriver.QCOMDriver()
    driver.memory.map(memory.base, len(memory.data), owner=memory)
    context = driver.contexts[1] = qcomdriver.Context(1, queued=1)
    context.pending.append((command, 1, 0))
    yield driver, context, publish


class TestLaunchAdmission(unittest.TestCase):
  def test_planner_and_executor_agree_for_every_encoded_base(self):
    with patch.object(emu, 'decode', return_value=(emu.Instruction('end', {}, {}, 0),)):
      for role in ('wgidconstid', 'localidregid'):
        for index in range(256):
          with self.subTest(role=role, index=index):
            memory = Memory(65536)
            words = invocation_launch(memory, role, index)
            wgid, lid = (index, 0xfc) if role == 'wgidconstid' else (0xfc, index)
            accepted = index <= 241 or index == 0xfc
            # Expected admission is independently stated, not derived from the
            # shared validator: the executor must retain its defensive check.
            if accepted:
              emu.execute(b'', b'', (1, 1, 1), (1, 1, 1), wgid, lid, memory)
              self.assertIsInstance(qcomgpu.compile_plan(words, memory)[-1], qcomgpu.Launch)
            else:
              with self.assertRaisesRegex(RuntimeError, 'invalid invocation register'):
                emu.execute(b'', b'', (1, 1, 1), (1, 1, 1), wgid, lid, memory)
              with self.assertRaises(qcomgpu.PM4Fault):
                qcomgpu.compile_plan(words, memory)

  def test_executor_rejects_bases_outside_encoded_range(self):
    with patch.object(emu, 'decode', return_value=(emu.Instruction('end', {}, {}, 0),)):
      for wgid, lid in ((-1, 0xfc), (256, 0xfc), (0xfc, -1), (0xfc, 256)):
        with self.subTest(wgid=wgid, lid=lid), self.assertRaisesRegex(RuntimeError, 'invalid invocation register'):
          emu.execute(b'', b'', (1, 1, 1), (1, 1, 1), wgid, lid, Memory())

  def test_invalid_suffix_rejects_before_prefix_with_or_without_wait(self):
    for role in ('wgidconstid', 'localidregid'):
      for index in (*range(242, 252), 253, 254, 255):
        for blocked in (False, True):
          with self.subTest(role=role, index=index, blocked=blocked):
            memory = Memory(65536)
            suffix = invocation_launch(memory, role, index)
            command = signal(memory.base, 123) + (wait_for(memory) if blocked else ()) + suffix
            original = bytes(memory.data)
            with inert_driver(memory, command) as (driver, context, publish), \
                 patch.object(qcomgpu.MemoryAction, 'apply', side_effect=AssertionError('prefix reached application')):
              with self.assertRaises(qcomgpu.PM4Fault):
                driver.drain()
              publish.assert_not_called()
              self.assertEqual(bytes(memory.data), original)
              self.assertEqual(list(context.pending), [(command, 1, 0)])
              self.assertEqual((context.queued, context.retired), (1, 0))
              self.assertFalse(driver.executing)

  def test_valid_boundaries_keep_wait_progress_and_do_not_replay_prefix(self):
    for role in ('wgidconstid', 'localidregid'):
      for index in (0, 241, 252):
        with self.subTest(role=role, index=index):
          memory = Memory(65536)
          command = signal(memory.base, 123) + wait_for(memory) + invocation_launch(memory, role, index)
          with inert_driver(memory, command) as (driver, context, publish):
            driver.drain()
            self.assertEqual(memory.read(memory.base, 4), (123).to_bytes(4, 'little'))
            self.assertEqual(context.pending[0][2], 1)
            self.assertEqual(context.retired, 0)
            self.assertEqual(publish.call_count, 1)
            # Another context/caller may update the signal while this one waits.
            # Resumption starts at the wait and must preserve that newer value.
            memory.write(memory.base, (456).to_bytes(4, 'little'))
            memory.write(memory.base + 4, (1).to_bytes(4, 'little'))
            driver.drain()
            self.assertEqual(memory.read(memory.base, 4), (456).to_bytes(4, 'little'))
            self.assertFalse(context.pending)
            self.assertEqual(context.retired, 1)
            self.assertEqual(publish.call_count, 1)


if __name__ == '__main__':
  unittest.main()
