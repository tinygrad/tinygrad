import ctypes, unittest
import tinygrad.runtime.autogen.amd.rdna4.ins as r4
import tinygrad.runtime.autogen.amd.cdna.ins as rc
from test.mockgpu.amd.emu import run_asm

class TestFlatRDNA4(unittest.TestCase):
  def test_flat_load_store(self):
    # Use real host pointers, as the emulator does for global memory. Check signed instruction offsets
    # and every dword of wide loads/stores without relying on LLVM to select FLAT instructions.
    for width in (1, 2, 3, 4):
      for offset in (-8, 0, 8):
        with self.subTest(width=width, offset=offset):
          values = [0x12345678, 0x90ABCDEF, 0xFEDCBA98, 0x76543210]
          mem = (ctypes.c_uint32 * 16)(*values, *([0xDEADBEEF] * 12))
          src = ctypes.addressof(mem) - offset
          dst = ctypes.addressof(mem) + 32 - offset
          regs = r4.v[4] if width == 1 else r4.v[4:4+width-1]
          instructions = [
            r4.v_mov_b32_e32(r4.v[0], src & 0xFFFFFFFF),
            r4.v_mov_b32_e32(r4.v[1], src >> 32),
            r4.v_mov_b32_e32(r4.v[2], dst & 0xFFFFFFFF),
            r4.v_mov_b32_e32(r4.v[3], dst >> 32),
            getattr(r4, f'flat_load_b{width*32}')(vdst=regs, vaddr=r4.v[0:1], saddr=r4.NULL, ioffset=offset),
            getattr(r4, f'flat_store_b{width*32}')(vaddr=r4.v[2:3], vsrc=regs, saddr=r4.NULL, ioffset=offset),
            r4.s_endpgm(),
          ]
          code = b''.join(inst.to_bytes() for inst in instructions)
          kernel = ctypes.create_string_buffer(code)
          self.assertEqual(run_asm(ctypes.addressof(kernel), len(code), 1, 1, 1, 1, 1, 1, 0, arch='rdna4'), 0)
          expected = values + [0xDEADBEEF] * 12
          expected[8:8+width] = values[:width]
          self.assertEqual(list(mem), expected)

class TestFlatCDNA(unittest.TestCase):
  def test_flat_ignores_saddr(self):
    mem = (ctypes.c_uint32 * 8)(*([0xDEADBEEF] * 8))
    addr = ctypes.addressof(mem)
    # LLVM encodes FLAT's unused saddr bits as zero. If treated as a GLOBAL saddr, this
    # SGPR pair redirects the access by 8 bytes within mem, so the regression fails without a segfault.
    saddr = (addr & ~0xFFFFFFFF) + 8
    instructions = [
      rc.s_mov_b32(rc.s[0], saddr & 0xFFFFFFFF), rc.s_mov_b32(rc.s[1], saddr >> 32),
      rc.v_mov_b32_e32(rc.v[0], addr & 0xFFFFFFFF), rc.v_mov_b32_e32(rc.v[1], addr >> 32),
      rc.v_mov_b32_e32(rc.v[2], 0x12345678),
      rc.flat_store_dword(addr=rc.v[0], data=rc.v[2], saddr=rc.s[0], offset=4),
      rc.v_mov_b32_e32(rc.v[2], 0),
      rc.flat_load_dword(vdst=rc.v[2], addr=rc.v[0], saddr=rc.s[0], offset=4),
      rc.global_store_dword(addr=rc.v[0:1], data=rc.v[2], saddr=rc.NULL, offset=16),
      rc.s_endpgm(),
    ]
    code = b''.join(inst.to_bytes() for inst in instructions)
    kernel = ctypes.create_string_buffer(code)
    self.assertEqual(run_asm(ctypes.addressof(kernel), len(code), 1, 1, 1, 1, 1, 1, 0, arch='cdna'), 0)
    expected = [0xDEADBEEF] * 8
    expected[1] = expected[4] = 0x12345678
    self.assertEqual(list(mem), expected)

class TestDSCDNA(unittest.TestCase):
  def test_ds_write_acc(self):
    for width, count in ((32, 1), (64, 1), (96, 1), (128, 1), (32, 2), (64, 2)):
      for acc in (0, 1):
        with self.subTest(width=width, count=count, acc=acc):
          values = [0x12345678, 0x90ABCDEF, 0xFEDCBA98, 0x76543210]
          mem = (ctypes.c_uint32 * 4)(*([0xDEADBEEF] * 4))
          addr = ctypes.addressof(mem)
          instructions = [rc.v_mov_b32_e32(rc.v[0], addr & 0xFFFFFFFF), rc.v_mov_b32_e32(rc.v[1], addr >> 32),
                          rc.v_mov_b32_e32(rc.v[2], 0)]
          # Keep different data in VGPRs and ACCVGPRs; the LDS address always comes from VGPRs.
          for i, value in enumerate(values):
            instructions += [rc.v_mov_b32_e32(rc.v[4+i], value), rc.v_accvgpr_write(rc.v[4+i], rc.v[4+i]),
                             rc.v_mov_b32_e32(rc.v[4+i], value ^ 0xFFFFFFFF)]
          data0 = rc.v[4:3+width//32]
          if count == 1:
            instructions.append(getattr(rc, f'ds_write_b{width}')(addr=rc.v[2], data0=data0, acc=acc))
          else:
            instructions.append(getattr(rc, f'ds_write2_b{width}')(addr=rc.v[2], data0=data0, data1=rc.v[4+width//32:3+width//16],
                                                                 offset1=1, acc=acc))
          for i in range(width*count//32):
            instructions += [rc.ds_read_b32(vdst=rc.v[8], addr=rc.v[2], offset0=i*4),
                             rc.global_store_dword(addr=rc.v[0:1], data=rc.v[8], saddr=rc.NULL, offset=i*4)]
          code = b''.join(inst.to_bytes() for inst in instructions + [rc.s_endpgm()])
          kernel = ctypes.create_string_buffer(code)
          self.assertEqual(run_asm(ctypes.addressof(kernel), len(code), 1, 1, 1, 1, 1, 1, 0, rsrc2=0x19c | (1 << 15), arch='cdna'), 0)
          expected = [v if acc else v ^ 0xFFFFFFFF for v in values[:width*count//32]]
          self.assertEqual(list(mem), expected + [0xDEADBEEF] * (4-len(expected)))

  def test_cdna_lds_above_64k(self):
    # CDNA4 supports more than RDNA's 64 KiB of LDS; this access is beyond
    # the old emulator PARAM bound even though the LDS allocation is large enough.
    out = (ctypes.c_uint32 * 1)(0)
    ptr = ctypes.addressof(out)
    instructions = [
      rc.v_mov_b32_e32(rc.v[0], ptr & 0xFFFFFFFF), rc.v_mov_b32_e32(rc.v[1], ptr >> 32),
      rc.v_mov_b32_e32(rc.v[2], 65536), rc.v_mov_b32_e32(rc.v[4], 0x12345678),
      rc.ds_write_b32(addr=rc.v[2], data0=rc.v[4]),
      rc.ds_read_b32(vdst=rc.v[8], addr=rc.v[2]),
      rc.global_store_dword(addr=rc.v[0:1], data=rc.v[8], saddr=rc.NULL),
      rc.s_endpgm(),
    ]
    code = b''.join(inst.to_bytes() for inst in instructions)
    kernel = ctypes.create_string_buffer(code)
    self.assertEqual(run_asm(ctypes.addressof(kernel), len(code), 1, 1, 1, 1, 1, 1, ptr,
                             rsrc2=0x19c | (129 << 15), arch='cdna'), 0)
    self.assertEqual(out[0], 0x12345678)

  def _check_ds_read_b64_tr_b16(self, offset: int, acc: int):
    # CDNA4 transposes 16 source lanes (4 fp16 words each) into 16 destination
    # lanes (4 values from distinct rows). Check all 64 lanes and both VGPRs.
    out = (ctypes.c_uint32 * 128)(*([0xDEADBEEF] * 128))
    instructions = [
      rc.v_lshlrev_b32_e32(rc.v[1], 3, rc.v[0]), # each lane owns 8 consecutive LDS bytes
      rc.v_lshlrev_b32_e32(rc.v[2], 2, rc.v[0]),
      rc.v_add_u32_e32(rc.v[3], 1, rc.v[2]),
      rc.v_add_u32_e32(rc.v[4], 2, rc.v[2]),
      rc.v_add_u32_e32(rc.v[5], 3, rc.v[2]),
      rc.v_lshlrev_b32_e32(rc.v[6], 16, rc.v[3]),
      rc.v_or_b32_e32(rc.v[6], rc.v[2], rc.v[6]),
      rc.v_lshlrev_b32_e32(rc.v[7], 16, rc.v[5]),
      rc.v_or_b32_e32(rc.v[7], rc.v[4], rc.v[7]),
      rc.ds_write_b64(addr=rc.v[1], data0=rc.v[6:7], offset0=offset),
      rc.ds_read_b64_tr_b16(vdst=rc.v[8:9], addr=rc.v[1], offset0=offset, acc=acc),
      *([rc.v_accvgpr_read(rc.v[8], rc.v[8]), rc.v_accvgpr_read(rc.v[9], rc.v[9])] if acc else []),
      rc.v_mov_b32_e32(rc.v[10], rc.s[0]), rc.v_mov_b32_e32(rc.v[11], rc.s[1]),
      rc.v_add_u32_e32(rc.v[10], rc.v[1], rc.v[10]),
      rc.global_store_dwordx2(addr=rc.v[10:11], data=rc.v[8:9], saddr=rc.NULL),
      rc.s_endpgm(),
    ]
    code = b''.join(inst.to_bytes() for inst in instructions)
    kernel = ctypes.create_string_buffer(code)
    self.assertEqual(run_asm(ctypes.addressof(kernel), len(code), 1, 1, 1, 64, 1, 1, ctypes.addressof(out),
                             rsrc2=0x19c | (1 << 15), arch='cdna'), 0)
    for lane in range(64):
      group, col = divmod(lane, 16)
      vals = [((group * 16 + row * 4 + col // 4) * 4 + col % 4) for row in range(4)]
      self.assertEqual(out[lane * 2], vals[0] | (vals[1] << 16), f"lane {lane} VGPR low")
      self.assertEqual(out[lane * 2 + 1], vals[2] | (vals[3] << 16), f"lane {lane} VGPR high")
  def test_ds_read_b64_tr_b16(self):
    for offset, acc in ((0, 0), (24, 0), (24, 1)):
      with self.subTest(offset=offset, acc=acc): self._check_ds_read_b64_tr_b16(offset, acc)

if __name__ == '__main__': unittest.main()
