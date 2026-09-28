import unittest, os, subprocess
from tinygrad.device import Device
from tinygrad.helpers import WIN

@unittest.skipIf(Device.DEFAULT != "AMD", "only run on AMD")
class TestDevice(unittest.TestCase):
  def test_nonexistent_iface(self):
    result = subprocess.run(['python3', '-c', 'from tinygrad import Device; Device[Device.DEFAULT].iface'],
                            env={**os.environ, "DEV":"USA+AMD"}, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"did you mean: 'USB'", result.stderr)

  def test_dev_id_out_of_range(self):
    result = subprocess.run(['python3', '-c', 'from tinygrad import Device; Device[Device.DEFAULT]'],
                            env={**os.environ, "DEV":":99+AMD"}, capture_output=True)
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(b"invalid visibility filter", result.stderr)

  @unittest.skipIf(WIN, "skipping windows test") # TODO: subprocess causes memory violation?
  def test_env_overwrite_default_compiler(self):
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler
    try: _, _ = HIPCompiler(Device[Device.DEFAULT].arch), AMDLLVMCompiler(Device[Device.DEFAULT].arch)
    except Exception as e: self.skipTest(f"skipping compiler test: not all compilers: {e}")

    imports = ("from tinygrad import Device; from tinygrad.runtime.support.compiler_amd import HIPCompiler; "
               "from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler")
    subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, AMDLLVMCompiler)"'],
                      shell=True, check=True, env={**os.environ, "DEV": "AMD:LLVM"})
    subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, HIPCompiler)"'],
                      shell=True, check=True, env={**os.environ, "DEV": "AMD"})
    subprocess.run([f'python3 -c "{imports}; assert isinstance(Device[Device.DEFAULT].compiler, HIPCompiler)"'],
                      shell=True, check=True, env={**os.environ, "DEV": "AMD:HIP"})

if __name__ == "__main__":
  unittest.main()
