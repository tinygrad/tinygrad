import unittest

import networks

from tinygrad import Tensor


class TestGRUCell(unittest.TestCase):
    def test_gru_simple(self):
        cell = networks.GRUCell(10, 20)
        inputs = Tensor.randn(1, 10)
        state = Tensor.zeros(1, 20)
        outputs, state = cell(inputs, state)
        self.assertEqual(outputs.numpy().shape, (1, 20))
        self.assertEqual(state.numpy().shape, (1, 20))
        inputs = Tensor.randn(1, 10)
        outputs, state = cell(inputs, state)
        self.assertEqual(outputs.numpy().shape, (1, 20))
        self.assertEqual(state.numpy().shape, (1, 20))


class TestConv2d(unittest.TestCase):
    def test_conv_encoder(self):
        inputs = Tensor.randn(1, 1, 64, 64, 3)
        conv = networks.ConvEncoder(inputs.numpy().shape[2:], 32)
        outputs = conv(inputs)
        self.assertEqual(outputs.numpy().shape, (1, 1, conv.outdim))

    def test_conv_decoder(self):
        inputs = Tensor.randn(1, 1, 4096)
        deconv = networks.ConvDecoder(4096, shape=(3, 64, 64), depth=32)
        outputs = deconv(inputs)
        self.assertEqual(outputs.numpy().shape, (1, 1, 64, 64, 3))

    def test_conv_encoder_decoder(self):
        inputs = Tensor.randn(1, 1, 64, 64, 3)
        conv = networks.ConvEncoder(inputs.numpy().shape[2:], 32)
        deconv = networks.ConvDecoder(conv.outdim, shape=(3, 64, 64), depth=32)
        outputs = deconv(conv(inputs))
        self.assertEqual(outputs.numpy().shape, (1, 1, 64, 64, 3))


class TestMLP(unittest.TestCase):
    def test_mlp_normal(self):
        inputs = Tensor.randn(1, 10)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="normal")
        outputs = mlp(inputs).sample()
        self.assertEqual(outputs.numpy().shape, (1, 20))

    def test_mlp_normal_std_fixed(self):
        inputs = Tensor.randn(1, 10)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="normal_std_fixed")
        outputs = mlp(inputs).sample()
        self.assertEqual(outputs.numpy().shape, (1, 20))

    def test_mlp_trunc_normal(self):
        inputs = Tensor.randn(1, 10)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="trunc_normal")
        outputs = mlp(inputs).sample()
        self.assertEqual(outputs.numpy().shape, (1, 20))

    def test_mlp_onehot(self):
        inputs = Tensor.randn(1, 10)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="onehot")
        outputs = mlp(inputs).sample()
        self.assertEqual(outputs.numpy().shape, (1, 20))

    def test_mlp_binary(self):
        inputs = Tensor.randn(20, 10)
        outputs = Tensor.randn(1, 20)
        mlp = networks.MLP(10, (), layers=2, units=2, dist="binary")
        dist = mlp(inputs)
        log_prob = dist.log_prob(outputs)
        self.assertEqual(log_prob.numpy().shape, (1, 20))

    def test_mlp_symlog_disc(self):
        inputs = Tensor.randn(1, 10)
        outputs = Tensor.randn(1, 1)
        mlp = networks.MLP(10, (255,), layers=2, units=2, dist="symlog_disc")
        outputs = mlp(inputs).log_prob(outputs)
        self.assertEqual(outputs.numpy().shape, (1, 1))

    def test_mlp_symlog_mse(self):
        inputs = Tensor.randn(1, 10)
        outputs = Tensor.randn(1, 20)
        mlp = networks.MLP(10, 20, layers=2, units=2, dist="symlog_mse")
        outputs = mlp(inputs).log_prob(outputs)
        self.assertEqual(outputs.numpy().shape, (1, 20))

    def test_mlp_output_dict(self):
        inputs = Tensor.randn(1, 10)
        shape = {"state": 20, "action": 20}
        mlp = networks.MLP(10, shape, layers=2, units=2, dist="normal")
        outputs = mlp(inputs)
        self.assertEqual(list(outputs.keys()), ["state", "action"])
        self.assertEqual(outputs["state"].sample().numpy().shape, (1, 20))
        self.assertEqual(outputs["action"].sample().numpy().shape, (1, 20))


class TestMulti(unittest.TestCase):
    def test_multi_encoder(self):
        cnn_keys = "state"
        mlp_keys = "action"
        shapes = {"state": (64, 64, 3), "action": 10}
        mlp = networks.MultiEncoder(shapes, mlp_keys, cnn_keys)
        obs = {"state": Tensor.randn(1, 1, 64, 64, 3), "action": Tensor.randn(1, 1, 10)}
        outputs = mlp(obs)
        self.assertEqual(outputs.numpy().shape, (1, 1, mlp.outdim))

    def test_multi_decoder(self):
        cnn_keys = "state"
        mlp_keys = "action"
        shapes = {"state": (64, 64, 3), "action": 256}
        mlp = networks.MultiDecoder(4352, shapes, mlp_keys, cnn_keys, image_dist="normal", vector_dist="normal")
        inputs = Tensor.randn(1, 1, 4352)
        outputs = mlp(inputs)
        self.assertEqual(outputs["state"].sample().numpy().shape, (1, 1, 64, 64, 3))
        self.assertEqual(outputs["action"].sample().numpy().shape, (1, 1, 256))

    def test_multi_encoder_decoder(self):
        cnn_keys = "state"
        mlp_keys = "action"
        shapes = {"state": (64, 64, 3), "action": 10}
        encoder = networks.MultiEncoder(shapes, mlp_keys, cnn_keys)
        decoder = networks.MultiDecoder(encoder.outdim, shapes, mlp_keys, cnn_keys, image_dist="normal", vector_dist="normal")
        obs = {"state": Tensor.randn(1, 1, 64, 64, 3), "action": Tensor.randn(1, 1, 10)}
        outputs = decoder(encoder(obs))
        self.assertEqual(outputs["state"].sample().numpy().shape, (1, 1, 64, 64, 3))
        self.assertEqual(outputs["action"].sample().numpy().shape, (1, 1, 10))

    def test_multi_encoder_decoder_B_T(self):
        B = 2
        T = 4
        cnn_keys = "state"
        mlp_keys = "action"
        shapes = {"state": (64, 64, 3), "action": 10}
        encoder = networks.MultiEncoder(shapes, mlp_keys, cnn_keys)
        decoder = networks.MultiDecoder(encoder.outdim, shapes, mlp_keys, cnn_keys, image_dist="normal", vector_dist="normal")
        obs = {"state": Tensor.randn(B, T, 64, 64, 3), "action": Tensor.randn(B, T, 10)}
        outputs = decoder(encoder(obs))
        self.assertEqual(outputs["state"].sample().numpy().shape, (B, T, 64, 64, 3))
        self.assertEqual(outputs["action"].sample().numpy().shape, (B, T, 10))


class TestRSSM(unittest.TestCase):
    def test_rssm_initial(self):
        B = 2
        rssm = networks.RSSM(num_actions=10, embed=20)
        state = rssm.initial(B)
        self.assertEqual(state["stoch"].numpy().shape, (B, 30, 30))
        self.assertEqual(state["deter"].numpy().shape, (B, 200))
        feat = rssm.get_feat(state)
        self.assertEqual(feat.numpy().shape, (B, 1100))

    def test_rssm_img_step(self):
        B = 2
        rssm = networks.RSSM(num_actions=10, embed=20)
        prev_state = rssm.initial(B)
        prev_action = Tensor.randn(B, 10)
        prior = rssm.img_step(prev_state, prev_action)
        self.assertEqual(prior["stoch"].numpy().shape, (B, 30, 30))
        self.assertEqual(prior["deter"].numpy().shape, (B, 200))
        prior = rssm.img_step(prior, prev_action)
        self.assertEqual(prior["stoch"].numpy().shape, (B, 30, 30))
        self.assertEqual(prior["deter"].numpy().shape, (B, 200))

    def test_rssm_obs_step(self):
        B = 2
        rssm = networks.RSSM(num_actions=10, embed=20)
        is_first = Tensor.ones(B)
        embed = Tensor.randn(B, 20)
        post, prior = rssm.obs_step(None, None, embed, is_first)
        self.assertEqual(post["stoch"].numpy().shape, (B, 30, 30))
        self.assertEqual(post["deter"].numpy().shape, (B, 200))
        is_first = Tensor([0, 1])
        prev_action = Tensor.randn(B, 10)
        post, _ = rssm.obs_step(post, prev_action, embed, is_first)
        self.assertEqual(post["stoch"].numpy().shape, (B, 30, 30))
        self.assertEqual(post["deter"].numpy().shape, (B, 200))
        is_first = Tensor.zeros(B)
        post, _ = rssm.obs_step(post, prev_action, embed, is_first)
        self.assertEqual(post["stoch"].numpy().shape, (B, 30, 30))
        self.assertEqual(post["deter"].numpy().shape, (B, 200))

    def test_rssm_imagine_with_action(self):
        B = 2
        T = 4
        rssm = networks.RSSM(num_actions=10, embed=20)
        state = rssm.initial(B)
        action = Tensor.randn(B, T, 10)
        prior = rssm.imagine_with_action(action, state)
        self.assertEqual(prior["stoch"].numpy().shape, (B, T, 30, 30))
        self.assertEqual(prior["deter"].numpy().shape, (B, T, 200))

    def test_rssm_observe(self):
        B = 2
        T = 4
        rssm = networks.RSSM(num_actions=10, embed=20)
        embed = Tensor.randn(B, T, 20)
        action = Tensor.randn(B, T, 10)
        is_first = Tensor.ones(B, T)
        post, prior = rssm.observe(embed, action, is_first)
        self.assertEqual(post["stoch"].numpy().shape, (B, T, 30, 30))
        self.assertEqual(post["deter"].numpy().shape, (B, T, 200))
        is_first = Tensor.zeros(B, T)
        post, prior = rssm.observe(embed, action, is_first, post)
        self.assertEqual(post["stoch"].numpy().shape, (B, T, 30, 30))
        self.assertEqual(post["deter"].numpy().shape, (B, T, 200))

    def test_rssm_kl_loss(self):
        B = 4
        T = 2
        rssm = networks.RSSM(num_actions=10, embed=20)
        embed = Tensor.randn(B, T, 20)
        action = Tensor.randn(B, T, 10)
        is_first = Tensor.ones(B, T)
        post, prior = rssm.observe(embed, action, is_first)
        loss, value, dyn_loss, rep_loss = rssm.kl_loss(post, prior, 1.0, 0.5, 0.1)
        self.assertEqual(loss.numpy().shape, (B, T))
        self.assertEqual(value.numpy().shape, (B, T))
        self.assertEqual(dyn_loss.numpy().shape, (B, T))
        self.assertEqual(rep_loss.numpy().shape, (B, T))


if __name__ == "__main__":
    unittest.main()
