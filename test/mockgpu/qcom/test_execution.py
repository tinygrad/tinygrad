import ctypes
import unittest

from test.mockgpu.qcom.corpus import (DIVERGENT_BRANCH, END, GLOBAL_ATOMIC_ADD, IMAGE_SAMPLE, IMAGE_STORE, INVALID_FLOAT_LUT,
  MESA_STD_HOT_LOOP, MESA_STD_HOT_PREHEADER, PARTIAL_WAVE_ADD, PREDICATION, PRIVATE_LANE_SPILL, PRIVATE_SIGNED_SPILL,
  RELATIVE_DESTINATION, RELATIVE_SOURCE_REPEAT, SHARED_BARRIER, UNSUPPORTED_UL_ADD, ir3_program)
from test.mockgpu.qcom.decoder import IR3Instruction, decode_ir3
from test.mockgpu.qcom.executor import _use_native_blocks, _workgroup_batch_size, execute_dispatch, execute_ir3


class TestA630IR3Execution(unittest.TestCase):
  def test_native_block_workload_threshold(self):
    self.assertFalse(_use_native_blocks((4095, 1, 1), (1, 1, 1)))
    self.assertTrue(_use_native_blocks((4096, 1, 1), (1, 1, 1)))
    self.assertTrue(_use_native_blocks((16, 8, 1), (8, 4, 1)))

  def test_workgroup_batch_policy_preserves_wave_visible_boundaries(self):
    safe = decode_ir3(ir3_program(PARTIAL_WAVE_ADD, END))
    atomic = decode_ir3(ir3_program(GLOBAL_ATOMIC_ADD, END))
    natural_loop = decode_ir3(ir3_program(MESA_STD_HOT_PREHEADER, MESA_STD_HOT_LOOP, END))
    vote = (IR3Instruction('bany', None, (('p', 62, 0),), False, 0),)
    self.assertEqual((_workgroup_batch_size(safe, 1), _workgroup_batch_size(safe, 32)), (256, 8))
    self.assertEqual((_workgroup_batch_size(atomic, 32), _workgroup_batch_size(vote, 32),
                      _workgroup_batch_size(natural_loop, 1), _workgroup_batch_size(safe, 65)), (1, 1, 1, 1))

  def test_relative_register_destination_uses_a0_component_addressing(self):
    # mov.s32s32 r<a0.x + 4>, r2.x
    program = ir3_program(RELATIVE_DESTINATION, END)
    move, _ = decode_ir3(program)
    self.assertEqual(move.dst, ('relr', 4, 0))
    regs = {('a', 61, 0): [0, 4], ('r', 2, 0): [0x1234, 0x5678]}
    execute_ir3(program, regs)
    self.assertEqual(regs[('r', 1, 0)], [0x1234, 0])
    self.assertEqual(regs[('r', 2, 0)], [0x1234, 0x5678])

  def test_private_spill_store_and_load_use_signed_offsets(self):
    # stp.u32 p[r11.y-96], r5.y, 1; ldp.u32 r2.x, p[r6.x], 1
    program = ir3_program(PRIVATE_SIGNED_SPILL, END)
    store, load, _ = decode_ir3(program)
    self.assertEqual((store.name, store.srcs[2], load.name), ('stp', -96, 'ldp'))
    regs = {('r', 11, 1): [100], ('r', 5, 1): [0xdeadbeef], ('r', 6, 0): [4]}
    execute_ir3(program, regs, private=bytearray(256))
    self.assertEqual(regs[('r', 2, 0)], [0xdeadbeef])

  def test_private_spills_are_isolated_between_wave_lanes(self):
    # stp.u32 p[r0.w+16], r0.x, 1; ldp.u32 r0.x, p[r0.w+16], 1
    program = ir3_program(PRIVATE_LANE_SPILL, END)
    regs = {('r', 0, 0): [0, 1, 2, 3], ('r', 0, 3): [0, 0, 0, 0]}
    execute_ir3(program, regs, private=[bytearray(32) for _ in range(4)])
    self.assertEqual(regs[('r', 0, 0)], [0, 1, 2, 3])

  def test_private_spill_validates_each_lane_backing(self):
    # stp.u32 p[r0.w+16], r0.x, 1
    program = ir3_program(PRIVATE_LANE_SPILL.words[0], END)
    regs = {('r', 0, 0): [1, 2], ('r', 0, 3): [0, 0]}
    with self.assertRaisesRegex(RuntimeError, r'IR3 stp out of bounds at 0x10'):
      execute_ir3(program, regs, private=[bytearray(16), bytearray(16)])

  def test_typed_image_store_and_sample_use_coherent_mock_backing(self):
    # stib.b.typed.2d.f16.4.imm hr0.x, r1.x, 0
    image = (ctypes.c_float * 4)()
    resource = {'address':ctypes.addressof(image), 'width':1, 'height':1, 'pitch':16, 'itemsize':4, 'encoded_itemsize':2}
    def check_range(address, size): self.assertTrue(resource['address'] <= address <= resource['address'] + 16 - size)
    store = ir3_program(IMAGE_STORE, END)
    regs = {('r', 1, 0): [0], ('r', 1, 1): [0], ('hr', 0, 0): [0x3c00], ('hr', 0, 1): [0x4000],
            ('hr', 0, 2): [0x4200], ('hr', 0, 3): [0x4400]}
    execute_ir3(store, regs, check_range=check_range, ibos=(resource,))
    self.assertEqual(list(image), [1.0, 2.0, 3.0, 4.0])

    # isam.1d (f32)(xyzw)r2.w, r1.w, s#0, t#0
    sample = ir3_program(IMAGE_SAMPLE, END)
    regs = {('r', 1, 3): [0]}
    execute_ir3(sample, regs, check_range=check_range, textures=(resource,))
    self.assertEqual([regs[('r', 2, 3)][0], regs[('r', 3, 0)][0], regs[('r', 3, 1)][0], regs[('r', 3, 2)][0]],
                     [0x3f800000, 0x40000000, 0x40400000, 0x40800000])

  def test_partial_wave_uses_its_own_initial_register_lanes(self):
    # add.u r1.x, r2.x, 1
    program = ir3_program(PARTIAL_WAVE_ADD, END)
    initial_regs = {("r", 2, 0): [1000 + lane for lane in range(65)]}

    # A 65-lane workgroup has one complete wave and a one-lane final wave.
    # The final wave must see initial_regs' lane 64, not lane 0.
    regs = execute_dispatch(program, (1, 1, 1), (65, 1, 1), 0xfc, initial_regs)
    self.assertEqual(regs[("r", 1, 0)], [1065])

  def test_independent_logical_waves_batch_with_distinct_group_ids(self):
    # add.u r1.x, r2.x, 1, with r2.x populated from workgroup_id.x.
    program = ir3_program(PARTIAL_WAVE_ADD, END)
    regs = execute_dispatch(program, (4, 1, 1), (32, 1, 1), 0xfc, workgroup_id_register=2)
    self.assertEqual(regs[("r", 1, 0)], [4] * 32)
    scalar_regs = execute_dispatch(program, (4, 1, 1), (1, 1, 1), 0xfc, workgroup_id_register=2)
    self.assertEqual(scalar_regs[("r", 1, 0)], [4])

  def test_batched_lanes_keep_workgroup_shared_memory_isolated(self):
    program = ir3_program(SHARED_BARRIER, END)
    group_a, group_b = bytearray(0x10000), bytearray(0x10000)
    regs = {
      ("r", 0, 0): [0] * 32 + [1] * 32,
      ("r", 4, 0): [0] * 32 + [4] * 32,
    }
    execute_ir3(program, regs, shared=[group_a] * 32 + [group_b] * 32)
    self.assertEqual(regs[("r", 5, 0)], [1] * 32 + [2] * 32)
    self.assertEqual((int.from_bytes(group_a[:4], 'little'), int.from_bytes(group_b[4:8], 'little')), (1, 2))

  def test_batched_workgroups_resume_divergent_paths_across_barriers(self):
    import test.mockgpu.qcom.dispatch as dispatch
    import test.mockgpu.qcom.executor as executor
    pred, wgid, out = ('p', 62, 0), ('r', 2, 0), ('r', 1, 0)
    program = (
      IR3Instruction('cmps.u', pred, (wgid, 0), False, 0, repeat_srcs=(False, False), src_mods=(0, 0), condition=5),
      IR3Instruction('br', None, (pred,), False, 0, branch_offset=4),
      IR3Instruction('add.u', out, (out, 10), False, 0, repeat_srcs=(False, False), src_mods=(0, 0)),
      IR3Instruction('bar', None, (), False, 0),
      IR3Instruction('jump', None, (), False, 0, branch_offset=4),
      IR3Instruction('add.u', out, (out, 20), False, 0, repeat_srcs=(False, False), src_mods=(0, 0)),
      IR3Instruction('bar', None, (), False, 0),
      IR3Instruction('jump', None, (), False, 0, branch_offset=1),
      IR3Instruction('end', None, (), False, 0),
    )
    real_executor_decode, real_dispatch_decode = executor.decode_ir3, dispatch.decode_ir3
    def decode(code, gpu_id=630, program=program, decode_real=real_executor_decode):
      return program if code == b'barbranch' else decode_real(code, gpu_id)
    setattr(executor, 'decode_ir3', decode)
    setattr(dispatch, 'decode_ir3', decode)
    try: regs = executor.execute_dispatch(b'barbranch', (2, 1, 1), (1, 1, 1), 0xfc, workgroup_id_register=2)
    finally:
      setattr(executor, 'decode_ir3', real_executor_decode)
      setattr(dispatch, 'decode_ir3', real_dispatch_decode)
    self.assertEqual(regs[out], [20])

  def test_predication_leaves_inactive_lane_writes_unchanged(self):
    # predt; add.u r0.x, r0.x, 1; predf; add.u r1.x, r1.x, 2;
    # prede; add.u r2.x, r2.x, 3
    program = ir3_program(PREDICATION, END)
    regs = {("p", 62, 0): [0, 1, 0, 1], ("r", 0, 0): [10] * 4,
            ("r", 1, 0): [20] * 4, ("r", 2, 0): [30] * 4}
    execute_ir3(program, regs)
    self.assertEqual(regs[("r", 0, 0)], [10, 11, 10, 11])
    self.assertEqual(regs[("r", 1, 0)], [22, 20, 22, 20])
    self.assertEqual(regs[("r", 2, 0)], [33, 33, 33, 33])

  def test_divergent_branch_reconverges_before_following_instruction(self):
    # br p0.x, #3; add.u r0.x, r0.x, 10; jump #3;
    # add.u r0.x, r0.x, 20; nop; add.u r1.x, r0.x, 1
    program = ir3_program(DIVERGENT_BRANCH, END)
    branch, _, jump, _, _, _, _ = decode_ir3(program)
    self.assertEqual((branch.name, jump.name), ("br", "jump"))

    regs = {("p", 62, 0): [0, 1], ("r", 0, 0): [0, 0], ("r", 1, 0): [0, 0]}
    execute_ir3(program, regs)
    self.assertEqual(regs[("r", 0, 0)], [10, 20])
    self.assertEqual(regs[("r", 1, 0)], [11, 21])

  def test_relative_register_source_repeat_uses_a0_component_addressing(self):
    # (rpt3) mov.u32u32 r4.x, (r)r<a0.x + 4>
    program = ir3_program(RELATIVE_SOURCE_REPEAT, END)
    move, _ = decode_ir3(program)
    self.assertEqual((move.srcs, move.repeat, move.repeat_srcs), ((("rel", 4, 0),), 3, (True,)))

    # a0.x is an IR3 component index.  With a0.x == 0, relative component
    # 4 is r1.x, and repeat walks r1.x through r1.w.
    regs = {("a", 61, 0): [0], **{("r", 1, component): [0x100 + component] for component in range(4)}}
    execute_ir3(program, regs)
    self.assertEqual([regs[("r", 4, component)][0] for component in range(4)], [0x100, 0x101, 0x102, 0x103])

  def test_shared_barrier_makes_first_wave_store_visible_to_second_wave(self):
    # shl.b r2.x, r0.x, 2; add.u r3.x, r0.x, 1; stl.u32 l[r2.x], r3.x, 1;
    # bar; ldl.u32 r5.x, l[r4.x], 1
    program = ir3_program(SHARED_BARRIER, END)
    instructions = decode_ir3(program)
    self.assertEqual([instruction.name for instruction in instructions], ["shl.b", "add.u", "stl", "bar", "ldl", "end"])

    # Wave 0 writes l[0] = 1. Wave 1 (the final, one-lane partial wave)
    # reads l[0] only after every wave has reached the barrier.
    regs = execute_dispatch(program, (1, 1, 1), (65, 1, 1), 0)
    self.assertEqual(regs[("r", 0, 0)], [64])
    self.assertEqual(regs[("r", 5, 0)], [1])

  def test_global_atomic_add_returns_old_value_and_updates_memory(self):
    # atomic.g.add.untyped.1d.u32.1.g r3.x, r1.x, r2.x
    program = ir3_program(GLOBAL_ATOMIC_ADD, END)
    cell = ctypes.c_uint32(7)
    address = ctypes.addressof(cell)
    regs = {("r", 1, 0): [address & 0xffffffff], ("r", 1, 1): [address >> 32], ("r", 2, 0): [5]}

    def check_range(actual_address, size): self.assertEqual((actual_address, size), (address, 4))

    execute_ir3(program, regs, check_range=check_range)
    self.assertEqual(regs[("r", 3, 0)], [7])
    self.assertEqual(cell.value, 12)

  def test_invalid_modifiers_and_encodings_include_a_useful_failure(self):
    # (ul)add.u r1.x, r0.x, 1 is real IR3 but intentionally unsupported.
    with self.assertRaisesRegex(ValueError, r"IR3 decode failed at PC 0 .*unsupported IR3 modifier UL"):
      decode_ir3(ir3_program(UNSUPPORTED_UL_ADD))

    # A syntactically valid floating LUT immediate with an invalid table index.
    with self.assertRaisesRegex(ValueError, r"IR3 decode failed at PC 0 .*invalid IR3 float immediate 12"):
      decode_ir3(ir3_program(INVALID_FLOAT_LUT))

    with self.assertRaisesRegex(ValueError, "IR3 code size must be a multiple of 8 bytes"):
      decode_ir3(b"\\0")
