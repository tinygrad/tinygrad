import ctypes, struct, unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import dtypes, UOp
from tinygrad.device import Buffer, TinyELF
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target, WIN
from tinygrad.uop.ops import Ops, KernelInfo, ProgramInfo

class TestProgramArguments(unittest.TestCase):
  signature = (("scale", 2, dtypes.int, ()), ("input", 1, dtypes.float, (1,)),
               ("bias", 3, dtypes.long, ()), ("output", 0, dtypes.float, (1,)))

  def test_hip_interleaved_args_and_update(self):
    from tinygrad.runtime.ops_hip import HIPProgram
    program = HIPProgram.__new__(HIPProgram)
    program.dev, program.prg, program.signature = SimpleNamespace(device_id=0), None, self.signature
    with patch('tinygrad.runtime.ops_hip.hip.hipSetDevice', return_value=0), \
         patch('tinygrad.runtime.ops_hip.hip.hipModuleLaunchKernel', return_value=0):
      for scale,bias in ((-7, 2**35+13), (11, -2**34)):
        program(0x11, 0x22, vals=(scale, bias))
        self.assertEqual(struct.unpack('<i4xQqQ', bytes(program.c_args)), (scale, 0x22, bias, 0x11))

  @unittest.skipIf(WIN, "DSP requires POSIX")
  def test_dsp_interleaved_rpc_args(self):
    from tinygrad.runtime.ops_dsp import DSPProgram, DSPBuffer
    captured = []
    def capture(lib, sc, pra, fds, attrs):
      captured.append((ctypes.string_at(pra[0].buf.pv, pra[0].buf.len), ctypes.string_at(pra[1].buf.pv, pra[1].buf.len),
                       list(fds)[3:]))
    program = DSPProgram(SimpleNamespace(exec_lib=capture), TinyELF(b'', 'test', Target('DSP'), self.signature))
    bufs = [DSPBuffer(0, 16, SimpleNamespace(fd=10), 4), DSPBuffer(0, 32, SimpleNamespace(fd=20), 8)]
    program(*bufs, vals=(-7, 2**35+13))
    self.assertEqual(captured, [(struct.pack('<i4xi4xqi4x', -7, 32, 2**35+13, 16), struct.pack('<2I', 4, 8), [10, 20])])

  @unittest.skipIf(WIN, "QCOM requires POSIX")
  def test_qcom_interleaved_kernargs(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue
    out, inp = UOp.param(0, dtypes.float, (1,)), UOp.param(4, dtypes.float, (1,))
    scale = UOp.param(1, dtypes.int, addrspace=AddrSpace.ALU)
    bias = UOp.param(3, dtypes.long, addrspace=AddrSpace.ALU)
    sink = out[0].store(inp[0] * scale + bias).sink(arg=KernelInfo())
    prg = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(scale, inp, bias, out)),
                              UOp(Ops.SOURCE, arg=''), UOp(Ops.BINARY, arg=b'')),
              arg=replace(ProgramInfo.from_sink(sink), params=(scale, inp, bias, out)))
    bufs = [Buffer('CPU', 1, dtypes.float).allocate() for _ in range(3)]
    call = prg.call(UOp.from_buffer(bufs[0]), UOp.const(-7, dtypes.int), UOp.from_buffer(bufs[1]),
                    UOp.const(2**35+13, dtypes.long), UOp.from_buffer(bufs[2]))
    for nir in (False, True):
      with self.subTest(nir=nir):
        data = SimpleNamespace(NIR=nir, ibo_cnt=0, tex_cnt=0, consts_info=[], samplers=[], samp_off=0, buf_off=0,
                               buf_offs=[0, 8, 16, 24], wgsz=0xfc, tex_off=32, ibo_off=32, kernargs_alloc_size=32)
        lin = QCOMComputeQueue.kernargs(SimpleNamespace(devs=('CPU',)), call, prg, data)
        payload = b''.join(w.arg if w.op is Ops.BINARY else struct.pack('<'+w.dtype.fmt, w.val) for w in lin.src)
        self.assertEqual(payload, struct.pack('<i4xQqQ', -7, bufs[2].get_buf('CPU'), 2**35+13, bufs[0].get_buf('CPU')))

if __name__ == '__main__': unittest.main()
