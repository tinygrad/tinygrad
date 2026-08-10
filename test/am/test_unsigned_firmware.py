import hashlib, struct, unittest, zlib

from tinygrad.runtime.support.am.ip import AM_GFX, parse_gfx_firmware_v2, strict_rlc_ready
from tinygrad.runtime.autogen.am import am, regs
from test.mockgpu.am.amgpu import MockGFX


def firmware(*, ip=(11, 0), code=b"\x11\x22\x33\x44", data=b"\x55\x66\x77\x88", start=0x10000,
             code_off=64, data_off=68, size=None, header_size=60, version=(2, 0), crc=0xDEADBEEF):
  actual_total = max(code_off + len(code), data_off + len(data), 60)
  total = actual_total if size is None else size
  blob = bytearray(actual_total)
  struct.pack_into("<IIHHHHIIII", blob, 0, total, header_size, *version, *ip, 7, len(code), code_off, crc)
  struct.pack_into("<IIIIIII", blob, 32, 9, len(code), code_off, len(data), data_off, start & 0xffffffff, start >> 32)
  if code_off + len(code) <= total: blob[code_off:code_off+len(code)] = code
  if data_off + len(data) <= total: blob[data_off:data_off+len(data)] = data
  return bytes(blob)


def firmware_with_valid_crc(**kwargs):
  blob = bytearray(firmware(crc=0, **kwargs))
  struct.pack_into("<I", blob, 28, zlib.crc32(blob[32:]) & 0xffffffff)
  return bytes(blob)


class TestGFXFirmwareV2Parser(unittest.TestCase):
  def test_plan_and_metadata(self):
    blob = firmware()
    plan = parse_gfx_firmware_v2(blob, (11, 0))
    self.assertEqual((plan.ip_version_major, plan.ip_version_minor), (11, 0))
    self.assertEqual((plan.code, plan.data, plan.start_pc), (b"\x11\x22\x33\x44", b"\x55\x66\x77\x88", 0x10000))
    self.assertEqual(plan.sha256, hashlib.sha256(blob).hexdigest())
    self.assertEqual(plan.declared_crc32, 0xDEADBEEF)
    self.assertFalse(plan.crc32_matches)  # diagnostic only

  def test_immutable_input_only(self):
    with self.assertRaises(TypeError): parse_gfx_firmware_v2(bytearray(firmware()), (11, 0))
    with self.assertRaises(TypeError): parse_gfx_firmware_v2(memoryview(firmware()), (11, 0))
    with self.assertRaises(ValueError): parse_gfx_firmware_v2(firmware(), (11,))  # type: ignore[arg-type]

  def test_crc_uses_bytes_after_common_header(self):
    self.assertTrue(parse_gfx_firmware_v2(firmware_with_valid_crc(), (11, 0)).crc32_matches)

  def test_malformed_vectors(self):
    outside = bytearray(firmware())
    struct.pack_into("<I", outside, 48, len(outside))
    vectors = [
      firmware(size=59), firmware(header_size=59), firmware(version=(1, 0)), firmware(ip=(12, 0)),
      firmware(code_off=60, data_off=60), firmware(code_off=62, data_off=68),
      firmware(code=b"", code_off=64, data_off=68), firmware(data=b"", code_off=64, data_off=68), firmware(start=3), bytes(outside),
    ]
    for blob in vectors:
      with self.subTest(blob=blob[:60]):
        with self.assertRaises(ValueError): parse_gfx_firmware_v2(blob, (11, 0))

  def test_declared_size_must_equal_file(self):
    blob = bytearray(firmware())
    struct.pack_into("<I", blob, 0, len(blob) - 1)
    with self.assertRaises(ValueError): parse_gfx_firmware_v2(bytes(blob), (11, 0))


class FakeReg:
  def __init__(self, owner, name, value=0, fields=None): self.owner, self.name, self.value, self.fields = owner, name, value, fields or {}
  def read(self, inst=0): return self.value
  def read_bitfields(self, inst=0): return dict(self.fields)
  def write(self, value=0, inst=0, **kwargs):
    self.owner.writes.append((self.name, value, kwargs, inst)); self.value = value; self.fields.update(kwargs)
    fail = self.owner.fail_on == self.name or (self.owner.fail_on == "SELECTOR_RESTORE" and self.name == "SELECTOR" and value == 0xA5A5 and not kwargs)
    if fail: self.owner.fail_on = None; raise OSError(f"injected {self.name} write failure")
    if self.name == "DC_OP" and kwargs.get("invalidate_dcache") and not self.owner.dc_timeout: self.fields["invalidate_dcache_complete"] = 1
    if self.name == "IC_OP" and kwargs.get("invalidate_cache") and not self.owner.ic_timeout: self.fields["invalidate_cache_complete"] = 1
  def update(self, inst=0, **kwargs): self.write(self.value, inst=inst, **kwargs)


class FakeADev:
  def __init__(self, cp_stat=0, boot=1, dc_timeout=False, ic_timeout=False, fail_on=None):
    self.writes, self.is_err_state = [], False
    self.fail_on = fail_on
    self.dc_timeout, self.ic_timeout = dc_timeout, ic_timeout
    self.ip_ver = {am.GC_HWIP: (11, 0, 0)}
    self.regCP_STAT = FakeReg(self, "CP_STAT", cp_stat)
    self.regRLC_RLCS_BOOTLOAD_STATUS = FakeReg(self, "BOOT", fields={"bootload_complete": boot})
    self.regGRBM_GFX_CNTL = FakeReg(self, "SELECTOR", 0xA5A5)
    self.regCP_MEC_RS64_CNTL = FakeReg(self, "MEC_CNTL")
    self.regCP_CPC_IC_BASE_CNTL = FakeReg(self, "IC_BASE_CNTL")
    self.regCP_CPC_IC_BASE_LO = FakeReg(self, "IC_BASE_LO")
    self.regCP_CPC_IC_BASE_HI = FakeReg(self, "IC_BASE_HI")
    self.regCP_MEC_DC_BASE_CNTL = FakeReg(self, "DC_BASE_CNTL")
    self.regCP_MEC_MDBASE_LO = FakeReg(self, "MDBASE_LO")
    self.regCP_MEC_MDBASE_HI = FakeReg(self, "MDBASE_HI")
    self.regCP_MEC_RS64_PRGRM_CNTR_START = FakeReg(self, "START_LO")
    self.regCP_MEC_RS64_PRGRM_CNTR_START_HI = FakeReg(self, "START_HI")
    self.regCP_MEC_DC_OP_CNTL = FakeReg(self, "DC_OP", fields={"invalidate_dcache_complete": 0})
    self.regCP_CPC_IC_OP_CNTL = FakeReg(self, "IC_OP", fields={"invalidate_cache_complete": 0})


class TestUnsignedPipe0Transaction(unittest.TestCase):
  def setUp(self): self.plan = parse_gfx_firmware_v2(firmware(), (11, 0))

  def gfx(self, adev):
    gfx = AM_GFX(adev); gfx.xccs = 1
    return gfx

  def test_strict_readiness_truth_table(self):
    for cp, boot, ready in [(0, 0, False), (0, 1, True), (1, 0, False), (1, 1, False)]:
      self.assertEqual(strict_rlc_ready(cp, boot), ready)

  def test_exact_pipe0_sequence_and_selector_restore(self):
    adev, gfx = FakeADev(), None
    gfx = self.gfx(adev)
    gfx._load_unsigned_mec_firmware_pipe0(self.plan, 0x20000, 0x30000, timeout_ms=5)
    self.assertEqual([x[0] for x in adev.writes], [
      "MEC_CNTL", "SELECTOR", "IC_BASE_CNTL", "IC_BASE_LO", "IC_BASE_HI", "DC_BASE_CNTL", "MDBASE_LO", "MDBASE_HI",
      "START_LO", "START_HI", "SELECTOR", "DC_OP", "DC_OP", "IC_OP", "IC_OP", "MEC_CNTL"])
    self.assertEqual(adev.writes[0][2], {"mec_halt": 1, "mec_pipe0_reset": 1, "mec_pipe0_active": 0})
    self.assertEqual(adev.writes[-1][2], {"mec_pipe0_reset": 0, "mec_pipe0_active": 1, "mec_halt": 0})
    self.assertEqual(adev.writes[10][1], 0xA5A5)
    self.assertFalse(adev.is_err_state)

  def test_allocation_overlap_and_topology_reject_before_commit(self):
    adev = FakeADev()
    with self.assertRaises(ValueError): self.gfx(adev)._load_unsigned_mec_firmware_pipe0(self.plan, 0x20000, 0x20000)
    self.assertEqual(adev.writes, [])
    gfx = self.gfx(adev); gfx.xccs = 2
    with self.assertRaises(ValueError): gfx._load_unsigned_mec_firmware_pipe0(self.plan, 0x20000, 0x30000)
    self.assertEqual(adev.writes, [])

  def test_no_precommit_writes(self):
    adev = FakeADev(cp_stat=1)
    with self.assertRaises(RuntimeError): self.gfx(adev)._load_unsigned_mec_firmware_pipe0(self.plan, 0x20000, 0x30000)
    self.assertEqual(adev.writes, [])

  def test_invalid_preflight_address_does_not_write(self):
    adev = FakeADev()
    with self.assertRaises(ValueError): self.gfx(adev)._load_unsigned_mec_firmware_pipe0(self.plan, 0x20004, 0x30000)
    self.assertEqual(adev.writes, [])

  def test_allocation_extent_does_not_wrap_address_space(self):
    adev = FakeADev()
    code = b"\0" * (0x10000 + 4)
    plan = parse_gfx_firmware_v2(firmware(code=code, data_off=64 + len(code)), (11, 0))
    with self.assertRaises(ValueError): self.gfx(adev)._load_unsigned_mec_firmware_pipe0(plan, (1 << 48) - 0x10000, 0x30000)
    self.assertEqual(adev.writes, [])

  def test_missing_register_does_not_write(self):
    adev = FakeADev()
    del adev.regCP_CPC_IC_OP_CNTL
    with self.assertRaises(RuntimeError): self.gfx(adev)._load_unsigned_mec_firmware_pipe0(self.plan, 0x20000, 0x30000)
    self.assertEqual(adev.writes, [])

  def test_cache_timeouts_fail_stopped(self):
    for fault in ("dc_timeout", "ic_timeout"):
      adev = FakeADev(**{fault: True})
      with self.subTest(fault=fault), self.assertRaises(TimeoutError):
        self.gfx(adev)._load_unsigned_mec_firmware_pipe0(self.plan, 0x20000, 0x30000, timeout_ms=1)
      self.assertTrue(adev.is_err_state)
      self.assertEqual(adev.regCP_MEC_RS64_CNTL.fields["mec_halt"], 1)
      self.assertEqual(adev.regCP_MEC_RS64_CNTL.fields["mec_pipe0_reset"], 1)
      self.assertEqual(adev.writes[-1][:2], ("SELECTOR", 0xA5A5))

  def test_commit_and_selector_restore_failures_fail_stopped(self):
    for fail_on in ("MEC_CNTL", "SELECTOR_RESTORE"):
      adev = FakeADev(fail_on=fail_on)
      with self.subTest(fail_on=fail_on), self.assertRaises(OSError):
        self.gfx(adev)._load_unsigned_mec_firmware_pipe0(self.plan, 0x20000, 0x30000)
      self.assertTrue(adev.is_err_state)
      self.assertEqual(adev.regCP_MEC_RS64_CNTL.fields["mec_halt"], 1)
      self.assertEqual(adev.regCP_MEC_RS64_CNTL.fields["mec_pipe0_reset"], 1)

  def test_default_init_has_no_unsigned_activation(self):
    self.assertNotIn("_load_unsigned_mec_firmware_pipe0", AM_GFX.init_hw.__code__.co_names)


class TestGC11MECMock(unittest.TestCase):
  def test_required_register_metadata(self):
    required = {
      "regCP_CPC_IC_OP_CNTL": {"invalidate_cache", "invalidate_cache_complete"},
      "regCP_CPC_IC_BASE_LO": {"ic_base_lo"}, "regCP_CPC_IC_BASE_HI": {"ic_base_hi"},
      "regCP_CPC_IC_BASE_CNTL": {"vmid", "address_clamp", "exe_disable", "cache_policy"},
    }
    for name, fields in required.items(): self.assertTrue(fields <= regs.gc_11_0_0[name][2].keys())
  def test_cache_completion_and_fault_injection(self):
    mmio = type("MMIO", (), {"regs": {}})()
    gfx = MockGFX(object(), mmio, (11, 0, 0))
    for name, field, complete, fault in (
      ("regCP_MEC_DC_OP_CNTL", "invalidate_dcache", "invalidate_dcache_complete", "dc_invalidate_timeout"),
      ("regCP_CPC_IC_OP_CNTL", "invalidate_cache", "invalidate_cache_complete", "ic_invalidate_timeout")):
      reg = gfx.reg(name)
      self.assertIsNotNone(reg)
      value = gfx._regs[name].encode(**{field: 1})
      gfx.write(reg, value)
      self.assertEqual(gfx._regs[name].decode(gfx.read(reg))[complete], 1)
      gfx.inject_fault(fault)
      gfx.write(reg, value)
      self.assertEqual(gfx._regs[name].decode(gfx.read(reg))[complete], 0)

  def test_mock_bootload_is_complete(self):
    mmio = type("MMIO", (), {"regs": {}})()
    gfx = MockGFX(object(), mmio, (12, 0, 0))
    self.assertEqual(gfx.read(gfx.reg("regCP_STAT")), 0)
    status = gfx.read(gfx.reg("regRLC_RLCS_BOOTLOAD_STATUS"))
    self.assertEqual(gfx._regs["regRLC_RLCS_BOOTLOAD_STATUS"].decode(status)["bootload_complete"], 1)


if __name__ == "__main__": unittest.main()
