import os
import unittest

from tinygrad import Tensor


class TestDeviceBehavior(unittest.TestCase):
    def test_device_behavior_gpu(self):
        os.environ["GPU"] = "1"
        self._should_pass()

    def test_device_behavior_clang(self):
        os.environ["CLANG"] = "1"
        self._should_pass()

    @staticmethod
    def _should_pass(qlen: int=3, qsz: int=1) -> None:
        queue = Tensor.zeros((qlen, qsz)).contiguous()
        buffered = Tensor.cat(*[Tensor([3]),  Tensor([1])]).unsqueeze(-1)
        replace = buffered.shape[0]
        queue[replace:] = queue[:-replace]
        queue[:replace] = buffered
        buffered = Tensor.cat(*[Tensor([4])]).unsqueeze(-1)
        replace = buffered.shape[0]
        queue[replace:] = queue[:-replace]
        queue[:replace] = buffered
        res = queue.tolist()
        assert res == [[4.0], [3.0], [1.0]], res
