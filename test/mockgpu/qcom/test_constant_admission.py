"""A630-7: immutable constant spans are admitted before workgroup or queue effects.

Ordinary ADD bytecode is decoded once. All host boundaries use owned bytearrays;
synthetic decoded control graphs isolate reachability without a native decoder.
"""
import contextlib, struct, unittest
from unittest.mock import patch
import numpy as np
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import qreg
from test.mockgpu.qcom import emu, qcomdriver, qcomgpu
from test.mockgpu.qcom.test_pm4 import ADD, Memory, launch_words, packet, signal
from test.mockgpu.qcom.test_launch_admission import wait_for


END = emu.Instruction('end', {}, {}, 0)


def constant_move(index, *, repeat=0, advance=False, half=False):
  return emu.Instruction('mov', {'SRC_TYPE':2 if half else 3, 'DST_TYPE':2 if half else 3, 'REPEAT':repeat},
                         {'DST':emu.Operand('register', 8, half),
                          'SRC':emu.Operand('constant', index, half, advance)}, 1 << 61)


def control(op, offset=0, **fields):
  return emu.Instruction(op, {'IMMED':offset & 0xffffffff, 'INV1':0, 'COMP1':0, 'INV2':0, 'COMP2':1, **fields}, {}, 0)


def short_load(memory, units):
  words = launch_words(memory)
  address = memory.base + 8192
  load = qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                             state_block=mesa.SB6_CS_SHADER, num_unit=units)
  return words[:-5] + packet(mesa.CP_LOAD_STATE6_FRAG, load, address & 0xffffffff, address >> 32) + words[-5:]


@contextlib.contextmanager
def inert_queue(memory, words, instructions):
  with patch.object(emu, 'decode', return_value=instructions), \
       patch.object(qcomdriver, 'read_host', side_effect=memory.read), \
       patch.object(qcomdriver, 'write_host', side_effect=memory.write) as publish, \
       patch.object(qcomdriver, 'validate_host_mapping', side_effect=memory.validate):
    driver = qcomdriver.QCOMDriver()
    driver.memory.map(memory.base, len(memory.data), owner=memory)
    context = driver.contexts[1] = qcomdriver.Context(1, queued=1)
    context.pending.append((words, 1, 0))
    yield driver, context, publish


class TestConstantAdmission(unittest.TestCase):
  add: tuple[emu.Instruction, ...]

  @classmethod
  def setUpClass(cls):
    cls.add = emu.decode(struct.pack('<11Q', *ADD).ljust(128, b'\0'))

  def admit(self, instructions, slots, accepted):
    # The boundary is before any Workgroup allocation, not a late read failure.
    with patch.object(emu, 'decode', return_value=instructions), patch.object(emu, 'Workgroup') as group:
      if accepted:
        emu.execute(b'', bytes(slots * 4), (1, 1, 1), (1, 1, 1), 252, 252, Memory())
        group.assert_called_once()
      else:
        with self.assertRaisesRegex(RuntimeError, 'constant.*(span|footprint)'):
          emu.execute(b'', bytes(slots * 4), (1, 1, 1), (1, 1, 1), 252, 252, Memory())
        group.assert_not_called()

  def test_direct_rejects_missing_constant_before_workgroup_creation(self):
    for slots in (0, 1, 4, 5):
      with self.subTest(slots=slots): self.admit(self.add, slots, False)
    for slots in (6, 8, 512):
      with self.subTest(slots=slots): self.admit(self.add, slots, True)

  def test_scalar_slot_and_repeat_endpoints_for_both_source_widths(self):
    for half in (False, True):
      for repeat in range(4):
        for advance in (False, True):
          indices = [3 + i if advance else 3 for i in range(repeat + 1)]
          instructions = (constant_move(3, repeat=repeat, advance=advance, half=half), END)
          for slots in (3, 4, 5, 6, 7, 8):
            with self.subTest(half=half, repeat=repeat, advance=advance, slots=slots):
              self.admit(instructions, slots, all(index < slots for index in indices))

  def test_all_synthetic_arithmetic_source_roles_contribute(self):
    # Enumerate generic operand roles. Cat3 SRC2 is GPR-only in actual decoded
    # images; that extra synthetic case checks enumeration, not ISA admission.
    for op, category, count in (('add.u', 2, 2), ('mad.f32', 3, 3), ('rcp', 4, 1)):
      for selected in range(1, count + 1):
        operands = {'DST':emu.Operand('register', 8)}
        operands.update({f'SRC{i}':emu.Operand('constant' if i == selected else 'register', 4)
                         for i in range(1, count + 1)})
        if category == 4: operands['SRC'] = operands.pop('SRC1')
        instructions = (emu.Instruction(op, {}, operands, category << 61), END)
        with self.subTest(op=op, source=selected):
          self.admit(instructions, 4, False)
          self.admit(instructions, 5, True)

  def test_end_and_enabled_jump_exclude_structurally_unreachable_references(self):
    missing = constant_move(100)
    for instructions in ((END, missing), (control('jump', 2), missing, END),
                         (control('predt'), END, missing),
                         (control('predf'), control('prede'), control('jump', 2, JP=1), missing, END)):
      with self.subTest(instructions=instructions): self.admit(instructions, 0, True)

  def test_branch_targets_after_an_earlier_end_remain_reachable(self):
    for op in ('br', 'brao', 'braa', 'jump'):
      with self.subTest(op=op):
        self.admit((control(op, 2), END, constant_move(4), END), 4, False)

  def test_predicated_jump_includes_fallthrough_even_when_current_mask_is_false(self):
    for pred in ('predt', 'predf'):
      for jp in (0, 1):
        with self.subTest(pred=pred, jp=jp):
          self.admit((control(pred), control('jump', 2, JP=jp), constant_move(4), END), 4, False)

  def test_merging_modes_and_backward_edges_does_not_lose_a_path(self):
    # PC3 is first reachable unpredicated, then through predt. Only its latter
    # arrival permits falling through the jump into the missing reference.
    graph = (control('br', 3), control('predt'), control('jump', 1), control('jump', 2), constant_move(4), END)
    self.admit(graph, 4, False)
    self.admit((constant_move(0), control('br', -1), END), 1, True)
    self.admit((control('jump', 0), constant_move(100), END), 0, True)

  def test_conditional_paths_are_not_pruned_using_live_values(self):
    # p0 starts false, but admission deliberately does not evaluate shader data.
    self.admit((control('predt'), constant_move(4), END), 4, False)
    for op in ('br', 'brao', 'braa'):
      with self.subTest(op=op): self.admit((control(op, 2), constant_move(4), END), 4, False)

  def test_insufficient_upload_rejects_queue_prefix_with_or_without_wait(self):
    for blocked in (False, True):
      with self.subTest(blocked=blocked):
        memory = Memory(65536)
        words = signal(memory.base, 123) + (wait_for(memory) if blocked else ()) + short_load(memory, 1)
        before = bytes(memory.data)
        with inert_queue(memory, words, self.add) as (driver, context, publish), \
             patch.object(qcomgpu.MemoryAction, 'apply', side_effect=AssertionError('prefix applied')):
          with self.assertRaisesRegex(qcomgpu.PM4Fault, 'constant.*(span|footprint)'): driver.drain()
          publish.assert_not_called()
          self.assertEqual(bytes(memory.data), before)
          self.assertEqual(list(context.pending), [(words, 1, 0)])
          self.assertEqual((context.queued, context.consumed, context.retired), (1, 0, 0))

  def test_short_and_larger_uploads_execute_and_resume_with_live_contents(self):
    for units in (2, 128):
      for blocked in (False, True):
        with self.subTest(units=units, blocked=blocked):
          memory = Memory(65536)
          words = signal(memory.base, 123) + (wait_for(memory) if blocked else ()) + short_load(memory, units)
          with inert_queue(memory, words, self.add) as (driver, context, publish):
            driver.drain()
            if blocked:
              self.assertEqual(context.pending[0][2], 1)
              self.assertEqual((context.consumed, context.retired), (1, 0))
              # Change the output pointer's contents without changing upload size.
              memory.write(memory.base + 8192, struct.pack('<Q', memory.base + 28))
              memory.write(memory.base, struct.pack('<I', 456))
              memory.write(memory.base + 4, struct.pack('<I', 1))
              driver.drain()
            self.assertEqual(memory.read(memory.base + (28 if blocked else 16), 4), struct.pack('<I', 26))
            self.assertEqual(memory.read(memory.base, 4), struct.pack('<I', 456 if blocked else 123))
            self.assertEqual((context.queued, context.consumed, context.retired), (1, 1, 1))
            self.assertFalse(context.pending)
            self.assertGreater(publish.call_count, 0)

  def test_staged_constant_contents_are_visible_to_short_launch(self):
    memory = Memory(65536)
    launch = short_load(memory, 2)
    words = signal(memory.base + 8192, (memory.base + 28) & 0xffffffff) + launch
    with inert_queue(memory, words, self.add) as (driver, context, _):
      driver.drain()
      self.assertEqual(memory.read(memory.base + 28, 4), struct.pack('<I', 26))
      self.assertEqual(memory.read(memory.base + 16, 4), bytes(4))
      self.assertEqual(context.retired, 1)

  def test_planner_shares_repeat_and_reachability_admission(self):
    cases = (((constant_move(3, repeat=2, advance=True), END), False),
             ((constant_move(3, repeat=2), END), True),
             ((control('jump', 2), constant_move(4), END), True),
             ((control('predt'), control('jump', 2), constant_move(4), END), False))
    for instructions, accepted in cases:
      memory = Memory(65536)
      words = short_load(memory, 1)
      with self.subTest(instructions=instructions), patch.object(emu, 'decode', return_value=instructions):
        if accepted: self.assertIsInstance(qcomgpu.compile_plan(words, memory)[-1], qcomgpu.Launch)
        else:
          with self.assertRaisesRegex(qcomgpu.PM4Fault, 'constant.*(span|footprint)'): qcomgpu.compile_plan(words, memory)

  def test_runtime_constant_read_bounds_remain_a_backstop(self):
    group = emu.Workgroup(bytes(16), (1, 1, 1), (0, 0, 0), 252, 252, 252, Memory(), 0, 0)
    for index in (-1, 4):
      with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, 'constant register index out of bounds'):
        group.read(emu.Operand('constant', index), np.array([0]))


if __name__ == '__main__': unittest.main()
