import unittest, tinygrad.runtime.autogen.libc as libc
from tinygrad import Device, dtypes
from tinygrad.ops import Ops, UOp
from test.test_uops import to_uops_list
from tinygrad.runtime.ops_clang import ClangCompiler
from tinygrad.runtime.support.elf import elf_loader

# All relocations that reference GOT or PLT
EXTERNAL_RELOCATIONS = {libc.R_X86_64_GOT32, libc.R_X86_64_PLT32, libc.R_X86_64_GOTPCREL, libc.R_X86_64_GOTTPOFF,
                        libc.R_X86_64_GOTOFF64, libc.R_X86_64_GOTPC32, libc.R_X86_64_GOT64, libc.R_X86_64_GOTPCREL64,
                        libc.R_X86_64_GOTPC64, libc.R_X86_64_GOTPLT64, libc.R_X86_64_PLTOFF64, libc.R_X86_64_GOTPC32_TLSDESC,
                        libc.R_X86_64_GOTPCRELX, libc.R_X86_64_REX_GOTPCRELX}

@unittest.skipIf(Device.DEFAULT != "CLANG", "Clang only")
class TestClang(unittest.TestCase):
  def test_no_compiler_rt_link(self):
    dst = UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(), (), 0)
    src = UOp(Ops.DEFINE_GLOBAL, dtypes.double.ptr(), (), 1)
    idx = UOp(Ops.CONST, dtypes.int, (), 0)
    load = UOp(Ops.LOAD, dtypes.double, (src.index(idx),))
    cast = UOp(Ops.CAST, dtypes.half, (load,))
    store = UOp(Ops.STORE, dtypes.void, (dst.index(idx), cast))
    uops = to_uops_list([store], opts=Device[Device.DEFAULT].renderer)
    code = Device[Device.DEFAULT].renderer.render("test", uops)
    elf = ClangCompiler(args=['-c', '-march=x86-64-v4', '--target=x86_64-none-unknown-elf']).compile(code)
    _, _, relocs = elf_loader(elf)
    for _, _, r_type, _ in relocs:
      self.assertNotIn(r_type, EXTERNAL_RELOCATIONS)