import pathlib, subprocess, sys, unittest, zipfile
import tinygrad
from tinygrad.device import Device
from tinygrad.helpers import temp

class TestZipImport(unittest.TestCase):
  def test_import_from_zip(self):
    pkg = pathlib.Path(tinygrad.__file__).parent
    with zipfile.ZipFile(zippath:=temp("tinygrad_zipimport.zip"), "w") as zf:
      for f in pkg.rglob("*.py"):
        if "__pycache__" not in f.parts: zf.write(f, f.relative_to(pkg.parent).as_posix())
    code = "import sys; sys.path.insert(0, sys.argv[1]); from tinygrad.device import Device; import tinygrad.runtime.autogen as autogen; " \
           "autogen.libc; print(sorted(Device._devices))"
    proc = subprocess.run([sys.executable, "-I", "-c", code, zippath], capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0, proc.stderr)
    self.assertEqual(proc.stdout.strip(), str(sorted(Device._devices)))

if __name__ == '__main__':
  unittest.main()
