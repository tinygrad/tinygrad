import os, sys, struct, tempfile, textwrap, unittest
from tinygrad.device import Compiler, CompileError

# a stand-in for runtime/support/compileserver.py speaking the same protocol: <u32 len><src> in, <u32 len><lib> out.
# the reply is an ELF-looking blob naming the kernel, so the name check can be exercised without a real compiler
FAKE_SERVER = textwrap.dedent('''
  import os, re, struct, sys, time
  mode = os.environ.get("FAKE_MODE", "ok")
  while (hdr := sys.stdin.buffer.read(4)):
    src = sys.stdin.buffer.read(struct.unpack("I", hdr)[0]).decode()
    name = "other_kernel" if mode == "wrong" else re.search(r"void (?:__launch_bounds__\\(\\d+\\) )?(\\w+)\\(", src).group(1)
    lib = b"\\x7fELF" + b"\\0" * 60 + f".text.{name}".encode() + b"\\0"
    out = struct.pack("I", len(lib)) + lib
    if mode == "slow":  # dribble the reply out so the reader sees short reads
      for i in range(0, len(out), 7): sys.stdout.buffer.write(out[i:i+7]); sys.stdout.buffer.flush(); time.sleep(0.001)
    else: sys.stdout.buffer.write(out); sys.stdout.buffer.flush()
''')

def kern(name:str) -> str: return f'extern "C" __global__ void __launch_bounds__(1) {name}(float* d) {{ d[0] = 1.0f; }}'

class FakeCompiler(Compiler):
  def __init__(self, script:str, mode:str):
    super().__init__(None)
    os.environ["FAKE_MODE"] = mode
    self.compiler_process = self.server(f"{sys.executable} {script}", "fake")
  def compile(self, src:str) -> bytes: return self.compile_server(src, self.compiler_process)

class TestCompileServer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(FAKE_SERVER)
    cls.script = f.name
  @classmethod
  def tearDownClass(cls): os.unlink(cls.script)
  def tearDown(self):
    p = self.c.compiler_process
    p.kill(), p.wait(), p.stdin.close(), p.stdout.close()

  def test_compile(self):
    self.c = FakeCompiler(self.script, "ok")
    self.assertIn(b".text.alpha\0", self.c.compile(kern("alpha")))

  def test_short_reads(self):
    self.c = FakeCompiler(self.script, "slow")
    for name in ("alpha", "beta"): self.assertIn(f".text.{name}\0".encode(), self.c.compile(kern(name)))

  def test_stale_reply_rejected_and_server_restarted(self):
    self.c = FakeCompiler(self.script, "ok")
    proc = self.c.compiler_process
    # an interrupted exchange leaves a reply in the pipe: send a request whose reply is never read
    proc.stdin.write(struct.pack("I", len(s:=kern("stale").encode())) + s)
    with self.assertRaises(CompileError): self.c.compile(kern("beta"))
    self.assertIsNot(self.c.compiler_process, proc)
    self.assertIn(b".text.gamma\0", self.c.compile(kern("gamma")))

  def test_wrong_kernel_rejected(self):
    self.c = FakeCompiler(self.script, "wrong")
    with self.assertRaises(CompileError): self.c.compile(kern("alpha"))

if __name__ == "__main__":
  unittest.main()
