import unittest

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.device import Device
from tinygrad.features.search import _compile_linearizer
from tinygrad.realize import create_schedule
from tinygrad.runtime.ops_cuda import CUDACompiler, PTXCompiler
from tinygrad.tensor import Tensor


class TestSearchCompileLinearizer(unittest.TestCase):
    @unittest.skipIf(Device.DEFAULT != "CUDA", "Only run on CUDA")
    def test_search_compile_linearizer_ptx(self):
        dev = Device["CUDA"]
        compiler = PTXCompiler(dev.arch)

        out1 = Tensor([True, True, False]) + Tensor([True, True, False])
        out2 = (True == Tensor([1, 2, 3])) == False # noqa: E712

        lin1 = Linearizer(create_schedule([out1.lazydata])[-1].ast[0])
        lin2 = Linearizer(create_schedule([out2.lazydata])[-1].ast[0])

        src_bytes1, _, _, _ = _compile_linearizer(compiler, lin1, "test")
        src_bytes2, _, _, _ = _compile_linearizer(compiler, lin2, "test")
        assert src_bytes1 and src_bytes2 is not None

    @unittest.skipIf(Device.DEFAULT != "CUDA", "Only run on CUDA")
    def test_search_compile_linearizer_cuda(self):
        dev = Device["CUDA"]
        compiler = CUDACompiler(dev.arch)

        out1 = Tensor([True, True, False]) + Tensor([True, True, False])
        out2 = (True == Tensor([1, 2, 3])) == False # noqa: E712

        lin1 = Linearizer(create_schedule([out1.lazydata])[-1].ast[0])
        lin2 = Linearizer(create_schedule([out2.lazydata])[-1].ast[0])

        src_bytes1, _, _, _ = _compile_linearizer(compiler, lin1, "test")
        src_bytes2, _, _, _ = _compile_linearizer(compiler, lin2, "test")
        assert src_bytes1 and src_bytes2 is not None

