import ctypes, unittest
from tinygrad.runtime.autogen import nv

class TestNVFwsecBootloaderStructs(unittest.TestCase):
  def test_falcon_ucode_desc_v2_layout(self):
    self.assertEqual(ctypes.sizeof(nv.FALCON_UCODE_DESC_V2), 60)
    d = nv.FALCON_UCODE_DESC_V2(Hdr=nv.FALCON_UCODE_DESC_HEADER(vDesc=0x0203ff02), StoredSize=1, UncompressedSize=2, VirtualEntry=3,
      InterfaceOffset=4, IMEMPhysBase=5, IMEMLoadSize=6, IMEMVirtBase=7, IMEMSecBase=8, IMEMSecSize=9, DMEMOffset=10, DMEMPhysBase=11,
      DMEMLoadSize=12, AltIMEMLoadSize=13, AltDMEMLoadSize=14)
    assert int.from_bytes(d, 'little') == int.from_bytes(
      b''.join(v.to_bytes(4, 'little') for v in [0x0203ff02,1,2,3,4,5,6,7,8,9,10,11,12,13,14]), 'little')
    assert d.Hdr.vDesc == 0x0203ff02 and d.IMEMSecBase == 8 and d.AltDMEMLoadSize == 14

  def test_rm_flcn_bl_desc_layout(self):
    self.assertEqual(ctypes.sizeof(nv.RM_FLCN_BL_DESC), 24)
    d = nv.RM_FLCN_BL_DESC(blStartTag=0x11, blDmemDescLoadOff=0x22, blCodeOffset=0x33, blCodeSize=0x44, blDataOffset=0x55, blDataSize=0x66)
    assert int.from_bytes(d, 'little') == int.from_bytes(
      b''.join(v.to_bytes(4, 'little') for v in [0x11,0x22,0x33,0x44,0x55,0x66]), 'little')

  def test_rm_flcn_bl_dmem_desc_layout(self):
    self.assertEqual(ctypes.sizeof(nv.RM_FLCN_BL_DMEM_DESC), 84)
    d = nv.RM_FLCN_BL_DMEM_DESC(ctxDma=4, codeDmaBaseLo=0xaabbccdd, codeDmaBaseHi=0x1, nonSecureCodeOff=0x100, nonSecureCodeSize=0x200,
      secureCodeOff=0x300, secureCodeSize=0x400, codeEntryPoint=0, dataDmaBaseLo=0xeeff0011, dataDmaBaseHi=0x2, dataSize=0x500,
      argc=0, argv=0)
    b = bytes(d)
    assert b[0:32] == b'\x00' * 32
    assert int.from_bytes(b[32:36], 'little') == 4                # ctxDma
    assert int.from_bytes(b[36:40], 'little') == 0xaabbccdd        # codeDmaBaseLo
    assert int.from_bytes(b[40:44], 'little') == 0x1               # codeDmaBaseHi
    assert int.from_bytes(b[44:48], 'little') == 0x100             # nonSecureCodeOff
    assert int.from_bytes(b[48:52], 'little') == 0x200             # nonSecureCodeSize
    assert int.from_bytes(b[52:56], 'little') == 0x300             # secureCodeOff
    assert int.from_bytes(b[56:60], 'little') == 0x400             # secureCodeSize
    assert int.from_bytes(b[64:68], 'little') == 0xeeff0011        # dataDmaBaseLo
    assert int.from_bytes(b[68:72], 'little') == 0x2               # dataDmaBaseHi
    assert int.from_bytes(b[72:76], 'little') == 0x500             # dataSize

if __name__ == '__main__':
  unittest.main()
