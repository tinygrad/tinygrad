from tinygrad import Device
from tinygrad.runtime.ops_clang import ClangCompiler
import io, contextlib, unittest

@unittest.skipUnless(Device.DEFAULT == "CLANG", "compile with clang")
class TestCpuObjDump(unittest.TestCase):
  def test_intel_syntax(self):
    src = "int main() { return 0; }"
    lib = ClangCompiler().compile(src)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
      ClangCompiler().disassemble(lib)
    assert "%" not in out.getvalue(), "objdump disassembly should be intel syntax"

if __name__ == '__main__':
  unittest.main()
