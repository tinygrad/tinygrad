# ruff: noqa: F405
"""Tests for GPU crash scenarios using AMD assembly to trigger invalid operations.

These tests intentionally cause GPU faults to verify error handling.
Run with: AMD=1 python -m pytest test/external/external_test_gpu_crash.py -v
"""
import unittest, os, re
from extra.assembly.amd.autogen.rdna3.ins import *  # noqa: F403
from extra.assembly.amd.dsl import s, v, Inst, NULL

def assemble_kernel(insts:list[Inst], name:str="test") -> str:
  kd = {"next_free_vgpr": 8, "next_free_sgpr": 8, "wavefront_size32": 1, "user_sgpr_kernarg_segment_ptr": 1, "kernarg_size": 8}
  disasm = "\n".join(inst.disasm() for inst in insts)
  hsasrc = f".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:\n{disasm}\n"
  return hsasrc + f".rodata\n.p2align 6\n.amdhsa_kernel {name}\n" + "\n".join(f".amdhsa_{k} {v}" for k, v in kd.items()) + "\n.end_amdhsa_kernel"

def assemble_raw_bytes(raw_bytes:bytes, name:str="test") -> str:
  """Assemble raw bytes into a kernel."""
  kd = {"next_free_vgpr": 8, "next_free_sgpr": 8, "wavefront_size32": 1, "user_sgpr_kernarg_segment_ptr": 1, "kernarg_size": 8}
  byte_str = ",".join(f"0x{b:02x}" for b in raw_bytes)
  hsasrc = f".text\n.globl {name}\n.p2align 8\n.type {name},@function\n{name}:\n.byte {byte_str}\n"
  return hsasrc + f".rodata\n.p2align 6\n.amdhsa_kernel {name}\n" + "\n".join(f".amdhsa_{k} {v}" for k, v in kd.items()) + "\n.end_amdhsa_kernel"

@unittest.skipIf(os.environ.get("AMD") != "1" or os.environ.get("MOCKGPU") == "1", "AMD with AM driver required")
class TestGPUCrash(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from tinygrad.device import Device
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    cls.dev = Device["AMD"]
    cls.compiler = HIPCompiler(cls.dev.arch)

  def setUp(self):
    # Verify device works before each test
    from tinygrad import Tensor
    try:
      t = Tensor([1.0, 2.0], device="AMD").realize()
      assert (t + 1).numpy().tolist() == [2.0, 3.0]
    except Exception:
      self.fail("Device not working before test")

  def _run_kernel(self, insts: list[Inst]):
    from tinygrad.runtime.ops_amd import AMDProgram
    asm_src = assemble_kernel(insts)
    lib = self.compiler.compile(asm_src)
    prg = AMDProgram(self.dev, "test", lib)
    buf = self.dev.allocator.alloc(64)
    prg(buf, global_size=(1,1,1), local_size=(1,1,1), wait=True)

  def _run_raw(self, raw_bytes: bytes):
    from tinygrad.runtime.ops_amd import AMDProgram
    asm_src = assemble_raw_bytes(raw_bytes)
    lib = self.compiler.compile(asm_src)
    prg = AMDProgram(self.dev, "test", lib)
    buf = self.dev.allocator.alloc(64)
    prg(buf, global_size=(1,1,1), local_size=(1,1,1), wait=True)

  def _assert_gpu_fault(self, func):
    """Assert that func raises a RuntimeError indicating a GPU fault (not a setup error)."""
    with self.assertRaises(RuntimeError) as cm:
      func()
    err_msg = str(cm.exception).lower()
    # Verify it's a GPU fault, not a setup/device initialization error
    self.assertTrue(
      re.search(r'fault|hang|timeout|illegal|memviol', err_msg),
      f"Expected GPU fault error, got: {cm.exception}"
    )


class TestIllegalInstructions(TestGPUCrash):
  """Tests for illegal/undefined instruction encodings."""

  def test_all_ones_encoding(self):
    """All-ones encoding (0xFFFFFFFF) is undefined."""
    raw_bytes = bytes([0xff, 0xff, 0xff, 0xff]) + s_endpgm().to_bytes()
    self._assert_gpu_fault(lambda: self._run_raw(raw_bytes))

  def test_all_zeros_encoding(self):
    """All-zeros encoding (0x00000000) is undefined."""
    raw_bytes = bytes([0x00, 0x00, 0x00, 0x00]) + s_endpgm().to_bytes()
    self._assert_gpu_fault(lambda: self._run_raw(raw_bytes))

  def test_random_garbage_bytes(self):
    """Random garbage bytes that don't decode to valid instructions."""
    raw_bytes = bytes([0xDE, 0xAD, 0xBE, 0xEF]) + s_endpgm().to_bytes()
    self._assert_gpu_fault(lambda: self._run_raw(raw_bytes))

  def test_truncated_instruction(self):
    """Only half of a 64-bit instruction."""
    partial = s_load_b64(s[0:1], s[2:3], 0, soffset=NULL).to_bytes()[:4]
    raw_bytes = partial + s_endpgm().to_bytes()
    self._assert_gpu_fault(lambda: self._run_raw(raw_bytes))


class TestOutOfBoundsMemoryAccess(TestGPUCrash):
  """Tests for out-of-bounds memory accesses."""

  def test_global_load_null_ptr(self):
    """Global load from NULL pointer."""
    insts = [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 0),
             global_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_global_store_null_ptr(self):
    """Global store to NULL pointer."""
    insts = [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 0), v_mov_b32_e32(v[2], 0xDEADBEEF),
             global_store_b32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_global_load_unmapped_high_address(self):
    """Global load from high unmapped address (0xDEAD00000000)."""
    insts = [v_mov_b32_e32(v[0], 0x00000000), v_mov_b32_e32(v[1], 0xDEAD),
             global_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_global_store_unmapped_high_address(self):
    """Global store to high unmapped address."""
    insts = [v_mov_b32_e32(v[0], 0x00000000), v_mov_b32_e32(v[1], 0xDEAD), v_mov_b32_e32(v[2], 0x12345678),
             global_store_b32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_global_load_b128_misaligned(self):
    """128-bit load from misaligned address."""
    insts = [v_mov_b32_e32(v[0], 0xBEEF0001), v_mov_b32_e32(v[1], 0xDEAD),
             global_load_b128(v[2:5], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_global_atomic_unmapped(self):
    """Atomic operation on unmapped memory."""
    insts = [v_mov_b32_e32(v[0], 0xBEEF0000), v_mov_b32_e32(v[1], 0xDEAD), v_mov_b32_e32(v[2], 1),
             global_atomic_add_u32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))


class TestSMEMFaults(TestGPUCrash):
  """Tests for scalar memory (SMEM) faults."""

  def test_smem_load_null(self):
    """SMEM load from NULL base."""
    insts = [s_mov_b32(s[2], 0), s_mov_b32(s[3], 0),
             s_load_b32(s[4], s[2:3], 0, soffset=NULL), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_smem_load_unmapped(self):
    """SMEM load from unmapped address."""
    insts = [s_mov_b32(s[2], 0xBEEF0000), s_mov_b32(s[3], 0xDEAD),
             s_load_b32(s[4], s[2:3], 0, soffset=NULL), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_smem_load_b256_misaligned(self):
    """256-bit SMEM load from misaligned address."""
    insts = [s_mov_b32(s[2], 0xBEEF0004), s_mov_b32(s[3], 0xDEAD),
             s_load_b256(s[4:11], s[2:3], 0, soffset=NULL), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))


class TestFlatMemoryFaults(TestGPUCrash):
  """Tests for FLAT memory instruction faults."""

  def test_flat_load_null(self):
    """FLAT load from NULL address."""
    insts = [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 0),
             flat_load_b32(v[2], addr=v[0:1], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_flat_store_null(self):
    """FLAT store to NULL address."""
    insts = [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 0), v_mov_b32_e32(v[2], 0xDEADBEEF),
             flat_store_b32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))

  def test_flat_atomic_null(self):
    """FLAT atomic on NULL address."""
    insts = [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 0), v_mov_b32_e32(v[2], 1),
             flat_atomic_add_u32(addr=v[0:1], data=v[2], saddr=NULL, offset=0), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))


class TestScratchMemoryFaults(TestGPUCrash):
  """Tests for private/scratch memory faults."""

  def test_scratch_load_huge_offset(self):
    """Scratch load with huge offset beyond allocated scratch."""
    insts = [v_mov_b32_e32(v[0], 0),
             scratch_load_b32(v[1], addr=v[0], saddr=NULL, offset=0x1FFF), s_waitcnt(0), s_endpgm()]
    self._assert_gpu_fault(lambda: self._run_kernel(insts))


class TestControlFlowFaults(TestGPUCrash):
  """Tests for control flow faults."""

  def test_missing_endpgm(self):
    """Kernel that runs off the end without s_endpgm."""
    raw_bytes = s_nop(0).to_bytes() + s_nop(0).to_bytes()
    self._assert_gpu_fault(lambda: self._run_raw(raw_bytes))


if __name__ == "__main__":
  unittest.main()
