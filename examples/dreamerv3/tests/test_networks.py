import unittest

import networks

from tinygrad import Tensor


class TestGRUCell(unittest.TestCase):
    def test_gru_simple(self):
        cell = networks.GRUCell(10, 20)
        inputs = Tensor.randn(1, 10)
        state = Tensor.zeros(1, 20)
        outputs, state = cell(inputs, state)
        self.assertEqual(outputs.shape, (1, 20))
        self.assertEqual(state.shape, (1, 20))
        inputs = Tensor.randn(1, 10)
        outputs, state = cell(inputs, state)
        self.assertEqual(outputs.shape, (1, 20))
        self.assertEqual(state.shape, (1, 20))


class TestConv2d(unittest.TestCase):
    def test_conv_encoder(self):
        inputs = Tensor.randn(1, 1, 64, 64, 3)
        conv = networks.ConvEncoder(inputs.shape[2:], 32)
        outputs = conv(inputs)
        self.assertEqual(outputs.shape, (1, 1, conv.outdim))

    def test_conv_decoder(self):
        inputs = Tensor.randn(1, 1, 4096)
        deconv = networks.ConvDecoder(4096, shape=(3, 64, 64), depth=32)
        outputs = deconv(inputs)
        self.assertEqual(outputs.shape, (1, 1, 64, 64, 3))

    def test_conv_encoder_decoder(self):
        inputs = Tensor.randn(1, 1, 64, 64, 3)
        conv = networks.ConvEncoder(inputs.shape[2:], 32)
        deconv = networks.ConvDecoder(conv.outdim, shape=(3, 64, 64), depth=32)
        outputs = deconv(conv(inputs))
        self.assertEqual(outputs.shape, (1, 1, 64, 64, 3))


class TestMLP(unittest.TestCase):
    def test_mlp_normal(self):
        inputs = Tensor.randn(1, 10)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="normal")
        outputs = mlp(inputs).sample()
        self.assertEqual(outputs.shape, (1, 20))

    def test_mlp_normal_std_fixed(self):
        inputs = Tensor.randn(1, 10)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="normal_std_fixed")
        outputs = mlp(inputs).sample()
        self.assertEqual(outputs.shape, (1, 20))

    def test_mlp_trunc_normal(self):
        inputs = Tensor.randn(1, 10)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="trunc_normal")
        outputs = mlp(inputs).sample()
        self.assertEqual(outputs.shape, (1, 20))

    def test_mlp_onehot(self):
        inputs = Tensor.randn(1, 10)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="onehot")
        outputs = mlp(inputs).sample()
        self.assertEqual(outputs.shape, (1, 20))

    def test_mlp_binary(self):
        inputs = Tensor.randn(20, 10)
        mlp = networks.MLP(10, (), layers=2, units=2, dist="binary")
        dist = mlp(inputs)
        entropy = dist.entropy()
        self.assertEqual(entropy.shape, (20, 1))
        outputs = dist.sample()
        self.assertEqual(outputs.shape, (20, 1))
        log_prob = dist.log_prob(outputs)
        self.assertEqual(log_prob.shape, (20, 1))

    def test_mlp_symlog_disc(self):
        inputs = Tensor.randn(1, 10)
        outputs = Tensor.randn(1, 20)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="symlog_disc")
        outputs = mlp(inputs).log_prob(outputs)
        self.assertEqual(outputs.shape, (1, 20))

    def test_mlp_symlog_mse(self):
        inputs = Tensor.randn(1, 10)
        outputs = Tensor.randn(1, 20)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="symlog_mse")
        outputs = mlp(inputs).log_prob(outputs)
        self.assertEqual(outputs.shape, (1, 20))


class TestMulti(unittest.TestCase):
    pass


class TestRSSM(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
