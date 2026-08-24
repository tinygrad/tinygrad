import unittest

from tinygrad.helpers import DEV, IMAGE


@unittest.skipUnless(DEV.interface.startswith('MOCK') and DEV.device == 'QCOM' and DEV.renderer == 'QCOMCL',
                     'MOCK+QCOM:QCOMCL required')
class TestQCOMCLRuntime(unittest.TestCase):
  def test_machine_code_smoke(self):
    from tinygrad import Tensor

    self.assertEqual(((Tensor([1.5, -2.0]) + Tensor([2.5, 5.0])) * 2).tolist(), [8.0, 6.0])
    self.assertEqual((Tensor([1, 2, 3]) * Tensor([4, 5, 6]) + 1).tolist(), [5, 11, 19])
    self.assertEqual((Tensor(list(range(8)))[1:7:2] + 10).tolist(), [11, 13, 15])

    values = Tensor([-2, -1, 0, 1, 2])
    self.assertEqual((values > 0).where(values * 7 + 3, values * -1).tolist(), [2, 1, 0, 10, 17])
    self.assertEqual(Tensor(list(range(16))).reshape(4, 4).sum(axis=1).tolist(), [6, 22, 38, 54])

    if IMAGE:
      self.assertEqual((Tensor.ones(16, 16) @ Tensor.ones(16, 16)).tolist(), [[16.0] * 16 for _ in range(16)])


if __name__ == '__main__': unittest.main()
