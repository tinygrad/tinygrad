import ctypes, struct, subprocess, tempfile, unittest
from tinygrad.helpers import WIN
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, field, pointer
from tinygrad.runtime.support.autogen import gen

class TestAutogen(unittest.TestCase):
  def compile(self, src):
    with tempfile.NamedTemporaryFile(suffix=".so") as f:
      subprocess.check_output(('clang', '-x', 'c', '-fPIC', '-shared', '-', '-o', f.name), input=src.encode())
      return DLL("test", f.name)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_packed_struct_interop(self):
    class Baz(Struct): SIZE = 8
    Baz._fields_ = ['a', 'b', 'c', 'd']
    Baz.a, Baz.b, Baz.c, Baz.d = field(0, ctypes.c_int, 30), field(3, ctypes.c_int, 30, 6), field(7, ctypes.c_int, 2, 4), field(7, ctypes.c_int, 2, 6)
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
    dll = self.compile(src)
    b = Baz(0xAA000, 0x00BB0, 0, 1)
    @dll.bind((Baz,), ctypes.c_int)
    def test(_): ...
    self.assertEqual(test(b), b.a + b.b + b.c + b.d)

  # https://github.com/python/cpython/issues/90914
  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_bitfield_interop(self):
    class Baz(Struct): SIZE = 1
    Baz._fields_ = [chr(ord('a') + i) for i in range(8)]
    for i in range(8): setattr(Baz, chr(ord('a') + i), field(0, ctypes.c_bool, 1, i))
    src = '''#include <stdbool.h>
      struct baz {
        bool a:1, b:1, c:1, d:1, e:1, f:1, g:1, h:1;
      };

      int test(struct baz x) {
        return x.c;
      }
    '''
    dll = self.compile(src)
    @dll.bind((Baz,), ctypes.c_int)
    def test(_): ...
    for i in range(8): self.assertEqual(test(Baz(*(j==i for j in range(8)))), i==2)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_struct_interop(self):
    class Baz(Struct): SIZE = 32
    Baz._fields_ = [chr(ord('a') + i) for i in range(8)]
    for i in range(8): setattr(Baz, chr(ord('a') + i), field(i*4, ctypes.c_int))
    src = '''#include <stdio.h>
      struct baz {
        int a, b, c, d, e, f, g, h;
      };

      struct baz test(struct baz x) {
        return (struct baz){x.h, x.g, x.f, x.e, x.d, x.c, x.b, x.a};
      }
    '''
    dll = self.compile(src)
    @dll.bind((Baz,), Baz)
    def test(_): ...
    self.assertEqual(bytes(test(Baz(*range(8)))), struct.pack("8i", *range(7, -1, -1)))

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_array_interop(self):
    Int4 = Array(ctypes.c_int, 4)
    src = """
      void test(int arr[4]) {
        arr[0] += 10;
        arr[3] *= 10;
      }
    """
    dll = self.compile(src)
    @dll.bind((Int4,), None)
    def test(_): ...
    test(arr:=Int4([1,2,3,4]))
    self.assertEqual(arr[0], 11)
    self.assertEqual(arr[1], 2)
    self.assertEqual(arr[2], 3)
    self.assertEqual(arr[3], 40)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_aos_interop(self):
    class Item(Struct): SIZE = 4
    Item._fields_ = ['val']
    Item.val = field(0, ctypes.c_int)
    src = """
    struct item { int val; };
      int test(struct item arr[3]) {
        int ret = 0;
        for (int i = 0; i < 3; i++) ret += arr[i].val;
        return ret;
      }
    """
    dll = self.compile(src)
    @dll.bind((Array(Item, 3),), ctypes.c_int)
    def test(_): ...
    self.assertEqual(test(Array(Item, 3)([Item(10), Item(20), Item(30)])), 60)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_soa_interop(self):
    class Row(Struct): SIZE = 16
    Row._fields_ = ['data']
    Row.data = field(0, Array(ctypes.c_int, 3))
    src = """
    struct row { int data[3]; };
      struct row test(struct row x) {
        return (struct row){{ x.data[2], x.data[1], x.data[0] }};
      }
    """
    dll = self.compile(src)
    @dll.bind((Row,), Row)
    def test(_): ...
    r = test(Row([10, 20, 30]))
    self.assertIsInstance(r, Row)
    self.assertEqual(r.data[0], 30)
    self.assertEqual(r.data[1], 20)
    self.assertEqual(r.data[2], 10)

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_nested_struct_interop(self):
    class Inner(Struct): SIZE = 4
    Inner._fields_ = ['a']
    Inner.a = field(0, ctypes.c_int)
    class Outer(Struct): SIZE = 8
    Outer._fields_ = ['inner', 'b']
    Outer.inner = field(0, Inner)
    Outer.b = field(4, ctypes.c_int)
    src = """
      struct i { int a; };
      struct o { struct i i; int b; };
      struct o test(struct o x) {
        return (struct o){(struct i){ x.b }, x.i.a };
      }
    """
    dll = self.compile(src)
    @dll.bind((Outer,), Outer)
    def test(_): ...
    o = test(Outer(Inner(10), 20))
    self.assertEqual(o.inner.a, 20)
    self.assertEqual(o.b, 10)

  def test_pointer_interop(self):
    src = """
      int test(int *p) {
        return (*p)++;
      }
    """
    dll = self.compile(src)
    @dll.bind((Pointer(ctypes.c_int),), ctypes.c_int)
    def test(_): ...
    self.assertEqual(test(pointer(i:=ctypes.c_int(10))), 10)
    self.assertEqual(i.value, 11)

  def test_struct_pointer_interop(self):
    class Foo(Struct): SIZE = 8
    Foo._fields_ = ['a', 'b']
    Foo.a = field(0, ctypes.c_int)
    Foo.b = field(4, ctypes.c_int)
    src = """
      struct foo { int a, b; };
      struct foo *test(struct foo *f) {
        int x = f->a;
        f->a = f->b;
        f->b = x;
        return f;
      }
    """
    dll = self.compile(src)
    @dll.bind((Pointer(Foo),), Pointer(Foo))
    def test(_): ...
    inp = pointer(Foo(10, 20))
    out = test(inp)
    self.assertEqual(inp.value, out.value)
    self.assertEqual(out.contents.a, 20)
    self.assertEqual(out.contents.b, 10)

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

  def test_packed_fields(self):
    uint8_t = ctypes.c_ubyte
    uint16_t = ctypes.c_ushort
    uint32_t = ctypes.c_uint32

    class struct_die_info(Struct): pass
    struct_die_info._packed_ = True
    struct_die_info._fields_ = [
      ('die_id', uint16_t),
      ('die_offset', uint16_t),
    ]
    die_info = struct_die_info
    class struct_ip_discovery_header(Struct): pass
    class struct_ip_discovery_header_0(ctypes.Union): pass
    class struct_ip_discovery_header_0_0(Struct): pass
    uint8_t = ctypes.c_ubyte
    struct_ip_discovery_header_0_0._fields_ = [
      ('base_addr_64_bit', uint8_t,1),
      ('reserved', uint8_t,7),
      ('reserved2', uint8_t),
    ]
    struct_ip_discovery_header_0._anonymous_ = ['_0']
    struct_ip_discovery_header_0._packed_ = True
    struct_ip_discovery_header_0._fields_ = [
      ('padding', (uint16_t * 1)),
      ('_0', struct_ip_discovery_header_0_0),
    ]
    struct_ip_discovery_header._anonymous_ = ['_0']
    struct_ip_discovery_header._packed_ = True
    struct_ip_discovery_header._fields_ = [
      ('signature', uint32_t),
      ('version', uint16_t),
      ('size', uint16_t),
      ('id', uint32_t),
      ('num_dies', uint16_t),
      ('die_info', (die_info * 16)),
      ('_0', struct_ip_discovery_header_0),
    ]
    ip_discovery_header = struct_ip_discovery_header

    hdr = b'IPDS\x04\x00|\x1d\x80\x1a\xffd\x01\x00\x00\x00\x8c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00' # noqa: E501
    ihdr = ip_discovery_header.from_buffer_copy(hdr)

    assert ctypes.sizeof(ihdr) == 80
    assert ihdr.signature == 0x53445049
    assert ihdr.version == 0x0004
    assert ihdr.num_dies == 1
    assert ihdr.base_addr_64_bit == 1

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_gen_from_header(self):
    header_content = """
    typedef struct {
      int x;
      int y;
    } Point;

    typedef enum {
      RED = 0,
      GREEN = 1,
      BLUE = 2
    } Color;

    typedef struct {
      Point origin;
      int width;
      int height;
      Color color;
    } Rectangle;

    int add_points(Point a, Point b);
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.h') as f:
      f.write(header_content)
      f.flush()

      generated_code = gen(name="test_header", dll=None, files=[f.name])

      namespace = {}
      exec(generated_code, namespace)

      self.assertIn('Point', namespace)
      self.assertIn('Color', namespace)
      self.assertIn('Rectangle', namespace)
      self.assertIn('RED', namespace)
      self.assertIn('GREEN', namespace)
      self.assertIn('BLUE', namespace)

      self.assertEqual(namespace['RED'], 0)
      self.assertEqual(namespace['GREEN'], 1)
      self.assertEqual(namespace['BLUE'], 2)

      Point = namespace['Point']
      p = Point()
      self.assertIsInstance(p, Struct)
      self.assertTrue(hasattr(p, 'x'))
      self.assertTrue(hasattr(p, 'y'))

      Rectangle = namespace['Rectangle']
      rect = Rectangle()
      self.assertTrue(hasattr(rect, 'origin'))
      self.assertTrue(hasattr(rect, 'width'))
      self.assertTrue(hasattr(rect, 'height'))
      self.assertTrue(hasattr(rect, 'color'))

  @unittest.skipIf(WIN, "doesn't compile on windows")
  def test_struct_ordering(self):
    header_content = """
    struct A;
    struct C;
    typedef struct A A;

    struct B {
      struct C *c_ptr;
    };

    struct C {
      struct A *a_ptr;
    };

    struct A {
      int x;
      struct B *b_ptr;
    };
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.h') as f:
      f.write(header_content)
      f.flush()
      generated_code = gen(name="test_ordering", dll=None, files=[f.name])
      namespace = {}
      exec(generated_code, namespace)
      self.assertIn('struct_A', namespace)
      self.assertIn('struct_B', namespace)
      self.assertIn('struct_C', namespace)
      A, B, C = namespace['struct_A'], namespace['struct_B'], namespace['struct_C']
      a, b, c = A(), B(), C()
      self.assertTrue(hasattr(a, 'x'))
      self.assertTrue(hasattr(a, 'b_ptr'))
      self.assertTrue(hasattr(b, 'c_ptr'))
      self.assertTrue(hasattr(c, 'a_ptr'))

if __name__ == "__main__": unittest.main()
