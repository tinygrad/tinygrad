import ctypes, os, subprocess, sys, textwrap, unittest

class TestIoctlBridge(unittest.TestCase):
  def test_callback_ref_keeps_owner_alive(self):
    import gc
    from test.mockgpu.cfunc_ref import BoundCFunctionRef
    callback = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 7)
    ref = BoundCFunctionRef.bind(callback, 'probe')
    address = ref.address()
    del callback
    gc.collect()
    self.assertEqual(ref.address(), address)
    self.assertIsNotNone(ref.owner)

  def test_runtime_owners_are_independent(self):
    from test.mockgpu.mockgpu import MockRuntime
    first, second = MockRuntime(()), MockRuntime(())
    self.assertIsNot(first.tracked_fds, second.tracked_fds)
    self.assertIsNot(first.ioctl_bridge, second.ioctl_bridge)
    first.tracked_fds[7] = object()
    self.assertNotIn(7, second.tracked_fds)
    first.tracked_fds.clear()
    first.close()
    second.close()

  def test_runtime_close_releases_owned_fds(self):
    from test.mockgpu.mockgpu import MockRuntime
    runtime = MockRuntime(())
    class Owned:
      def __init__(self): self.closed = False
      def close(self, fd): self.closed = True
    owned = Owned()
    runtime.tracked_fds[9] = owned
    runtime.close()
    self.assertTrue(owned.closed)
    self.assertFalse(runtime.tracked_fds)
    runtime.close()

  def test_runtime_does_not_replace_process_hooks(self):
    from tinygrad.engine import realize
    from tinygrad.runtime.autogen import libc
    original_ioctl, original_exec = libc.dll.ioctl, realize.exec_kernel
    import test.mockgpu.mockgpu as mockgpu
    self.assertIs(libc.dll.ioctl, original_ioctl)
    self.assertIs(realize.exec_kernel, original_exec)
    self.assertIs(libc.dll.ioctl, mockgpu.ioctl_bridge.native)
    self.assertIsNot(libc.dll.ioctl, mockgpu.ioctl_bridge.callback)

  def test_runtime_launch_scope_rejects_reentry_and_recovers(self):
    from test.mockgpu.mockgpu import MockRuntime
    runtime = MockRuntime(())
    with runtime.launch_scope():
      with self.assertRaisesRegex(RuntimeError, 'submission already active'):
        with runtime.launch_scope(): pass
      with self.assertRaisesRegex(RuntimeError, 'active MockRuntime'):
        runtime.close()
    with runtime.launch_scope(): pass
    runtime.close()
  def test_checked_call_preserves_first_failure(self):
    self.run_code('''
      from tinygrad.runtime.support.system import FileIOInterface
      from tinygrad.runtime.autogen import libc
      from test.mockgpu.ioctl import IoctlBridge
      class Broken:
        def ioctl(self, fd, request, arg): raise ValueError('IR3 pc=0x10: unsupported instruction')
      bridge = IoctlBridge(libc.dll.ioctl, {123456: Broken()})
      recovered = []
      def launch():
        assert bridge.callback(123456, 42, None) == -1
        assert bridge.callback(-1, 0, None) == -1
      try: bridge.run(launch, recovered.append)
      except ValueError as error: assert str(error) == 'IR3 pc=0x10: unsupported instruction'
      else: raise AssertionError('callback failure did not cross the Python launch boundary')
      assert recovered == [123456]
      assert bridge.pending_error is None and not bridge.checking
      assert bridge.run(lambda: 7, recovered.append) == 7
      assert recovered == [123456]
    ''')

  def test_recovery_failure_preserves_callback_error(self):
    self.run_code('''
      from tinygrad.runtime.autogen import libc
      from test.mockgpu.ioctl import IoctlBridge
      class Broken:
        def ioctl(self, fd, request, arg): raise ValueError('original IR3 failure')
      bridge = IoctlBridge(libc.dll.ioctl, {123456: Broken()})
      def recover(fd): raise OSError('recovery failure')
      try: bridge.run(lambda: bridge.callback(123456, 42, None), recover)
      except ValueError as error:
        assert str(error) == 'original IR3 failure'
        assert isinstance(error.__cause__, OSError)
        assert str(error.__cause__) == 'recovery failure'
      else: raise AssertionError('original callback error was replaced')
      assert bridge.pending_error is None and not bridge.checking
    ''')

  def test_unimplemented_eventfd_fails_explicitly(self):
    from test.mockgpu.mockgpu import MockFileIOInterface
    with self.assertRaises(NotImplementedError): MockFileIOInterface.eventfd(0)

  def run_code(self, code):
    out = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=os.getcwd(),
                         env=os.environ | {'DEV': 'MOCK+QCOM;MOCK+AMD', 'PYTHONPATH': '.'}, capture_output=True, text=True, timeout=60)
    self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
    self.assertEqual(out.stderr, '')

  def test_native_callback_routes_and_errors(self):
    self.run_code('''
      import ctypes, errno, gc, os, termios
      from tinygrad.runtime.support.system import FileIOInterface
      from tinygrad.runtime.autogen import kgsl, kfd, libc
      from test.mockgpu.qcom.qcomdriver import _ioctl_request
      from test.mockgpu.ioctl import IoctlBridge
      import test.mockgpu.mockgpu as mockgpu
      q, a = FileIOInterface('/dev/kgsl-3d0'), FileIOInterface('/dev/kfd')
      bridge = mockgpu.ioctl_bridge
      signature = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, use_errno=True)
      native = signature(ctypes.cast(bridge.callback, ctypes.c_void_p).value)
      arg = kgsl.struct_kgsl_drawctxt_create()
      req = _ioctl_request(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE)
      assert native(q.fd, req, ctypes.addressof(arg)) == 0
      assert arg.drawctxt_id in mockgpu.tracked_fds[q.fd].driver.contexts
      version = kfd.struct_kfd_ioctl_get_version_args()
      assert native(a.fd, _ioctl_request(kfd.AMDKFD_IOC_GET_VERSION), ctypes.addressof(version)) == 0
      assert (version.major_version, version.minor_version) == (1, 14)
      assert native(q.fd, req, None) == -1 and ctypes.get_errno() == errno.EFAULT
      assert bridge.last_error[:2] == (q.fd, req)
      assert native(q.fd, 0, ctypes.addressof(arg)) == -1 and ctypes.get_errno() == errno.ENOTTY
      r, w = os.pipe()
      try:
        os.write(w, b'abc')
        available = ctypes.c_int()
        assert native(r, termios.FIONREAD, ctypes.addressof(available)) == 0 and available.value == 3
        assert bridge.last_error is None
        assert native(r, 0, None) == -1 and ctypes.get_errno() == errno.ENOTTY
      finally:
        os.close(r)
        os.close(w)
      fd = q.fd
      del q, a
      gc.collect()
      assert not mockgpu.tracked_fds
      assert native(fd, req, ctypes.addressof(arg)) == -1 and ctypes.get_errno() == errno.EBADF
    ''')

  def test_exception_diagnostic(self):
    self.run_code('''
      import ctypes, errno
      from tinygrad.runtime.support.system import FileIOInterface
      from tinygrad.runtime.autogen import libc
      from test.mockgpu.ioctl import IoctlBridge
      class Broken:
        def ioctl(self, fd, request, arg): raise ValueError('bad PM4 at dword 7')
      bridge = IoctlBridge(libc.dll.ioctl, {123456: Broken()})
      assert bridge.callback(123456, 42, None) == -1 and ctypes.get_errno() == errno.EIO
      assert bridge.last_error[:2] == (123456, 42)
      assert str(bridge.last_error[2]) == 'bad PM4 at dword 7'
      assert bridge.last_error[2].__traceback__ is None
      class MissingResult:
        def ioctl(self, fd, request, arg): return None
      bridge.tracked_fds[123456] = MissingResult()
      assert bridge.callback(123456, 42, None) == -1 and ctypes.get_errno() == errno.EIO
      assert isinstance(bridge.last_error[2], TypeError)
    ''')

  def test_hcq2_function_pointer(self):
    self.run_code('''
      import ctypes, gc
      from tinygrad.runtime.support.system import FileIOInterface
      from tinygrad.runtime.autogen import kgsl
      from tinygrad.runtime.support import hcq2
      from tinygrad.helpers import Context
      from test.mockgpu.qcom.qcomdriver import _ioctl_request
      import test.mockgpu.mockgpu as mockgpu
      f = FileIOInterface('/dev/kgsl-3d0')
      try:
        with Context(HCQ_RUNTIME_DEV='CPU'):
          ptr = hcq2.cfunc_buf('libc', 'ioctl', f.ioctl_function).host.view(fmt='Q')[0]
        assert ptr == f.ioctl_function.address()
        native = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)(ptr)
        arg = kgsl.struct_kgsl_drawctxt_create()
        assert native(f.fd, _ioctl_request(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE), ctypes.addressof(arg)) == 0
        assert arg.drawctxt_id in mockgpu.tracked_fds[f.fd].driver.contexts
      finally:
        hcq2.cfunc_buf.cache_clear()
        del f
        gc.collect()
      assert not mockgpu.tracked_fds
    ''')

  def test_compiled_hcq2_call(self):
    self.run_code('''
      import ctypes, gc
      from tinygrad import dtypes
      from tinygrad.helpers import Context
      from tinygrad.uop.ops import UOp, Ops, KernelInfo
      from tinygrad.engine.realize import lower_and_compile, run_linear
      from tinygrad.runtime.support.system import FileIOInterface
      from tinygrad.runtime.support import hcq2
      from tinygrad.runtime.autogen import kgsl
      from test.mockgpu.qcom.qcomdriver import _ioctl_request
      import test.mockgpu.mockgpu as mockgpu
      file = FileIOInterface('/dev/kgsl-3d0')
      try:
        with Context(HCQ_RUNTIME_DEV='CPU'):
          arg = kgsl.struct_kgsl_drawctxt_create()
          out = UOp.placeholder((1,), dtypes.int32, device='CPU', slot=1, volatile=True, tag='ioctl_result')
          ret = hcq2.ccall(file.ioctl_function, file.fd, UOp.const(_ioctl_request(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE), dtypes.uint32),
                          UOp.const(ctypes.addressof(arg), dtypes.uint64))
          call = hcq2.lower_call(UOp.sink(out.index(0).store(ret), arg=KernelInfo('ioctl')).call(aux=hcq2.HCQInfo(('CPU',))))
          linear = hcq2.hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(call,))), allow_cache=False)
          run_linear(linear, jit=True)
          buffers = [u.buffer for u in linear.src[0].without_after.src[1:] if u.op is Ops.BUFFER]
          assert next(b for b in buffers if b.dtype is dtypes.int).host.view(fmt='i')[0] == 0
          assert arg.drawctxt_id in mockgpu.tracked_fds[file.fd].driver.contexts
      finally:
        hcq2.cfunc_buf.cache_clear()
        del file
        gc.collect()
      assert not mockgpu.tracked_fds
    ''')

if __name__ == '__main__': unittest.main()
