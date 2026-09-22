import os, subprocess, sys, textwrap, unittest

class TestRuntimeHooks(unittest.TestCase):
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
