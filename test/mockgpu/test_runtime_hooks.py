import os, subprocess, sys, textwrap, unittest

class TestRuntimeHooks(unittest.TestCase):
  def test_mockpci_amd_discovery(self):
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
      from tinygrad.runtime.support.system import PCIDevice
      from tinygrad.runtime.support.am.amdev import AMDev
      dev = object.__new__(AMDev)
      dev.pci_dev = PCIDevice("AM", "mock:am:0")
      dev.vram = dev.pci_dev.map_bar(0)
      dev.mmio = dev.pci_dev.map_bar(5, fmt="I")
      dev.vf_rlc_gated = []
      dev._run_discovery()
      assert dev.vram_size == dev.pci_dev.bar_info(0)[1] > 0
    ''')], cwd=source_root,
                            env=os.environ | {'DEV': 'MOCKPCI+AMD', 'PYTHONPATH': '.'}, capture_output=True, text=True, timeout=60)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(result.stderr, '')

  def test_missing_file_close(self):
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
      from tinygrad.runtime.support.system import FileIOInterface
      fd = FileIOInterface('/definitely/not/a/mock/file')
      assert fd.fd is None
      fd.close()
      fd.close()
    ''')], cwd=source_root, env=os.environ | {'DEV': 'MOCKPCI+AMD', 'PYTHONPATH': '.'}, capture_output=True, text=True, timeout=60)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(result.stderr, '')

  def test_virtual_binary_read(self):
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
      from tinygrad.runtime.support.system import FileIOInterface
      from test.mockgpu.mockgpu import runtime
      config = FileIOInterface('/sys/bus/pci/devices/mock:am:0/config')
      config.write(bytes((127,)), binary=True, offset=0x34)
      assert config.read(1, binary=True, offset=0x34) == bytes((127,))
      config.close()
      assert not runtime.tracked_fds
      runtime.close()
    ''')], cwd=source_root, env=os.environ | {'DEV': 'MOCKPCI+AMD', 'PYTHONPATH': '.'}, capture_output=True, text=True, timeout=60)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(result.stderr, '')

  def test_backend_runtime_isolation(self):
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    for backend in ('MOCK+AMD', 'MOCK+NV'):
      with self.subTest(backend=backend):
        result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
          from tinygrad.runtime.support.system import FileIOInterface
          from tinygrad.runtime.support.memory import MMIOInterface
          import test.mockgpu.mockgpu as mockgpu
          from tinygrad.engine import realize
          from tinygrad.runtime.autogen import libc
          assert mockgpu.runtime.drivers
          assert libc.dll.ioctl is mockgpu.ioctl_bridge.native
          original_exec = realize.exec_kernel
          fd = FileIOInterface(mockgpu.runtime.drivers[0].tracked_files[0].path)
          assert fd.mock_runtime is mockgpu.runtime
          assert libc.dll.ioctl is mockgpu.ioctl_bridge.native
          assert realize.exec_kernel is original_exec
          del fd
          mockgpu.runtime.close()
          assert not mockgpu.runtime.tracked_fds
          seen = []
          mmio = MMIOInterface(0x1000, 4, view_factory=lambda addr, size: (seen.append((addr, size)) or memoryview(bytearray(size))))
          mmio[0] = 7
          assert seen == [(0x1000, 4)] and mmio[0] == 7
        ''')], cwd=source_root, env=os.environ | {'DEV': backend, 'PYTHONPATH': '.'}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, '')

if __name__ == '__main__': unittest.main()
