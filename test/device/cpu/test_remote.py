import socket, threading, unittest
from types import SimpleNamespace
from tinygrad import Device, dtypes
from tinygrad.codegen import to_program
from tinygrad.dtype import AddrSpace
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad.runtime.support.system import RemoteCmd, RemotePCIDevice
from tinygrad.uop.ops import KernelInfo, UOp
from extra.remote.serve import serve, programs

@unittest.skipUnless(Device.DEFAULT == 'CPU', 'requires CPU')
class TestRemoteCPU(unittest.TestCase):
  def test_mixed_arguments(self):
    if Device['CPU'].renderer.target.renderer == 'LVP': self.skipTest('remote CPU uses the native calling convention')
    scalar = UOp.param(0, dtypes.int16, addrspace=AddrSpace.ALU)
    out = UOp.param(1, dtypes.int32, 1)
    bias = UOp.param(2, dtypes.int32, addrspace=AddrSpace.ALU)
    prg = to_program(out.index(0).store((scalar + bias).cast(dtypes.int32)).sink(arg=KernelInfo(name='remote_args'), tag=1),
                     Device['CPU'].renderer)
    result = UOp.new_buffer('CPU', 1, dtypes.int32).buffer.ensure_allocated()
    with socket.socket() as listener:
      listener.bind(('127.0.0.1', 0))
      listener.listen(1)
      with socket.create_connection(listener.getsockname()) as client, listener.accept()[0] as server:
        remote = RemotePCIDevice('CPU', 'remote:local:0', client)
        def run_server():
          try: serve(server)
          except ConnectionError: pass
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        count = len(programs)
        try:
          runtime = CPUProgram(SimpleNamespace(remote=remote, device='CPU'), prg.to_elf())
          for wait,bias_value in ((False, 7), (True, 9)):
            with self.subTest(wait=wait):
              args = [-3, result._buf, bias_value]
              elapsed = runtime(*args, wait=wait)
              remote.rpc(RemoteCmd.PING)
              self.assertEqual(result.as_memoryview().cast('i').tolist(), [bias_value - 3])
              if wait: self.assertGreaterEqual(elapsed, 0)
              else: self.assertIsNone(elapsed)
        finally:
          client.shutdown(socket.SHUT_RDWR)
          thread.join(timeout=10)
          del programs[count:]
        self.assertFalse(thread.is_alive())

if __name__ == '__main__': unittest.main()
