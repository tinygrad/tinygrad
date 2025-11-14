import ctypes, subprocess, tempfile, unittest
from tinygrad.helpers import WIN
from tinygrad.runtime.support.c import Struct

class TestAutogen(unittest.TestCase):
  def test_packed_struct_sizeof(self):
    layout = [('a', ctypes.c_char), ('b', ctypes.c_int, 5), ('c', ctypes.c_char)]
    class X(ctypes.Structure): _fields_, _layout_ = layout, 'gcc-sysv'
    class Y(ctypes.Structure): _fields_, _pack_, _layout_ = layout, 1, 'ms'
    class Z(Struct): _packed_, _fields_ = True, layout
    self.assertNotEqual(ctypes.sizeof(X), 4) # ctypes bug! gcc-13.3.0 says this should have size 4
    self.assertEqual(ctypes.sizeof(Y), 6)
    self.assertEqual(ctypes.sizeof(Z), 3)
    layout = [('a', ctypes.c_int, 31), ('b', ctypes.c_int, 31), ('c', ctypes.c_int, 1), ('d', ctypes.c_int, 1)]
    class Foo(ctypes.Structure): _fields_, _layout_ = layout, 'gcc-sysv'
    class Bar(ctypes.Structure): _fields_, _pack_, _layout_ = layout, 1, 'ms'
    class Baz(Struct): _fields_, _packed_ = layout, True
    self.assertEqual(ctypes.sizeof(Foo), 12)
    self.assertEqual(ctypes.sizeof(Bar), 12)
    self.assertEqual(ctypes.sizeof(Baz), 8)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_packed_struct_interop(self):
    class Baz(Struct): pass
    Baz._packed_ = True
    Baz._fields_ = [('a', ctypes.c_int, 30), ('b', ctypes.c_int, 30), ('c', ctypes.c_int, 2), ('d', ctypes.c_int, 2)]
    src = '''
      struct __attribute__((packed)) baz {
        int a:30;
        int b:30;
        int c:2;
        int d:2;
      };

      int test(struct baz x) {
        return x.a + x.b + x.c + x.d;
      }
    '''
    args = ('-x', 'c', '-fPIC', '-shared')
    with tempfile.NamedTemporaryFile(suffix=".so") as f:
      subprocess.check_output(('clang',) + args + ('-', '-o', f.name), input=src.encode('utf-8'))
      b = Baz(0xAA000, 0x00BB0, 0, 1)
      test = ctypes.CDLL(f.name).test
      test.argtypes = [Baz]
      self.assertEqual(test(b), b.a + b.b + b.c + b.d)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_packed_structs(self):
    NvU32 = ctypes.c_uint32
    NvU64 = ctypes.c_uint64
    class FWSECLIC_READ_VBIOS_DESC(Struct): pass
    FWSECLIC_READ_VBIOS_DESC._packed_ = True
    FWSECLIC_READ_VBIOS_DESC._fields_ = [
      ('version', NvU32),
      ('size', NvU32),
      ('gfwImageOffset', NvU64),
      ('gfwImageSize', NvU32),
      ('flags', NvU32),
    ]
    class FWSECLIC_FRTS_REGION_DESC(Struct): pass
    FWSECLIC_FRTS_REGION_DESC._packed_ = True
    FWSECLIC_FRTS_REGION_DESC._fields_ = [
      ('version', NvU32),
      ('size', NvU32),
      ('frtsRegionOffset4K', NvU32),
      ('frtsRegionSize', NvU32),
      ('frtsRegionMediaType', NvU32),
    ]
    class FWSECLIC_FRTS_CMD(Struct): pass
    FWSECLIC_FRTS_CMD._packed_ = True
    FWSECLIC_FRTS_CMD._fields_ = [
      ('readVbiosDesc', FWSECLIC_READ_VBIOS_DESC),
      ('frtsRegionDesc', FWSECLIC_FRTS_REGION_DESC),
    ]
    read_vbios_desc = FWSECLIC_READ_VBIOS_DESC(version=0x1, size=ctypes.sizeof(FWSECLIC_READ_VBIOS_DESC), flags=2)
    frst_reg_desc = FWSECLIC_FRTS_REGION_DESC(version=0x1, size=ctypes.sizeof(FWSECLIC_FRTS_REGION_DESC),
      frtsRegionOffset4K=0xdead, frtsRegionSize=0x100, frtsRegionMediaType=2)
    frts_cmd = FWSECLIC_FRTS_CMD(readVbiosDesc=read_vbios_desc, frtsRegionDesc=frst_reg_desc)
    assert int.from_bytes(frts_cmd, 'little') == 0x2000001000000dead0000001400000001000000020000000000000000000000000000001800000001
    assert int.from_bytes(frts_cmd.readVbiosDesc, 'little') == int.from_bytes(read_vbios_desc, 'little')
    assert int.from_bytes(frts_cmd.frtsRegionDesc, 'little') == int.from_bytes(frst_reg_desc, 'little')
    assert frts_cmd.readVbiosDesc.__class__ is FWSECLIC_READ_VBIOS_DESC
    assert frts_cmd.frtsRegionDesc.__class__ is FWSECLIC_FRTS_REGION_DESC

if __name__ == "__main__": unittest.main()
