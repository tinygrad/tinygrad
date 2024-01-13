import tempfile
import unittest
from pprint import pprint

import gymnasium as gym
import models
import numpy as np
import utils

from tinygrad import Tensor


class TestRewardEMA(unittest.TestCase):
    def test_reward_ema(self):
        reward_ema = models.RewardEMA(device="cuda", alpha=0.9)
        mean, std = reward_ema(Tensor([1.0, 2.0, 3.0]))
        print(mean.item(), std.item())
        mean, std = reward_ema(Tensor([1.0, 2.0, 3.0]))
        print(mean.item(), std.item())


class TestWorldModel(unittest.TestCase):
    def test_world_model_init(self):
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)})
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        print(f"world model parameters: {sum(param.numel() for param in world_model.parameters())}")

    def test_world_model_preprocess(self):
        B = 8
        T = 6
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)})
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        data = {
            "image": np.random.randint(0, 255, (B, T, 64, 64, 3)),
            "action": np.random.randint(0, 3, (B, T), dtype=np.int32),
            "reward": np.random.rand(B, T),
            "discount": np.ones((B, T)),
            "is_first": np.ones((B, T)),
            "is_terminal": np.zeros((B, T)),
        }
        data = world_model.preprocess(data)

    def test_world_model_video_pred(self):
        B = 8
        T = 6
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)})
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        data = {
            "image": np.random.randint(0, 255, (B, T, 64, 64, 3)),
            "action": np.random.randint(0, 3, (B, T)),
            "reward": np.random.rand(B, T),
            "discount": np.ones((B, T)),
            "is_first": np.ones((B, T)),
            "is_terminal": np.zeros((B, T)),
        }
        video_pred = world_model.video_pred(data)
        logger = utils.Logger(tempfile.gettempdir(), 0)
        logger.offline_video("video", video_pred, 0)

    def test_world_model_train(self):
        B = 4
        T = 2
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)})
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        data = {
            "image": np.random.randint(0, 255, (B, T, 64, 64, 3)),
            "action": np.random.randint(0, 3, (B, T)),
            "reward": np.random.rand(B, T),
            "discount": np.ones((B, T)),
            "is_first": np.ones((B, T)),
            "is_terminal": np.zeros((B, T)),
        }
        print("Train step: 0")
        post, context, metrics = world_model._train(data)
        pprint(metrics)
        self.assertEqual(post["stoch"].numpy().shape, (B, T, 32, 32))
        self.assertEqual(post["deter"].numpy().shape, (B, T, 512))
        self.assertEqual(context["embed"].numpy().shape, (B, T, 4096))
        self.assertEqual(context["feat"].numpy().shape, (B, T, 1536))
        print("Train step: 1")
        post, context, metrics = world_model._train(data)
        pprint(metrics)
        print("Train step: 2")
        post, context, metrics = world_model._train(data)
        pprint(metrics)


class TestImagBehavior(unittest.TestCase):
    def test_imag_behavior_init(self):
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)})
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        imag_behavior = models.ImagBehavior(config, world_model)
        print(f"actor parameters: {sum(param.numel() for param in imag_behavior.actor_parameters())}")
        print(f"value parameters: {sum(param.numel() for param in imag_behavior.value_parameters())}")

    def test_imag_behavior_funcs(self):
        B = 8
        T = 6
        H = 5
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)})
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        imag_behavior = models.ImagBehavior(config, world_model)
        start = world_model.dynamics.initial(B * T)
        start = {k: v.reshape((B, T) + v.shape[1:]) for k, v in start.items()}
        feats, states, actions = imag_behavior._imagine(start, imag_behavior.actor, H)
        self.assertEqual(feats.numpy().shape, (H, B * T, 1536))
        self.assertEqual(states["stoch"].numpy().shape, (H, B * T, 32, 32))
        self.assertEqual(states["deter"].numpy().shape, (H, B * T, 512))
        self.assertEqual(actions.numpy().shape, (H, B * T, world_model.num_actions))
        rewards = Tensor.uniform((H, B * T))
        target, weights, base = imag_behavior._compute_target(feats, states, rewards)
        actor_loss, metrics = imag_behavior._compute_actor_loss(feats, actions, target, weights, base)
        actor_loss.mean().backward()  # checks backward pass
        metrics["actor_loss"] = actor_loss.mean().item()
        pprint(metrics)

    def test_imag_behavior_train(self):
        B = 8
        T = 6
        obs_space = gym.spaces.Dict({"image": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)})
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        imag_behavior = models.ImagBehavior(config, world_model)
        start = world_model.dynamics.initial(B * T)
        start = {k: v.reshape((B, T) + v.shape[1:]) for k, v in start.items()}

        def reward(f, s, a):
            return world_model.heads["reward"](world_model.dynamics.get_feat(s)).mode.squeeze(-1)

        feat, state, action, weights, metrics = imag_behavior._train(start, reward)
        pprint(metrics)


if __name__ == "__main__":
    unittest.main()
