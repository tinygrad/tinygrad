from tinygrad import Tensor
import unittest
import models
import utils
import gymnasium as gym
import numpy as np
import tempfile


class TestRewardEMA(unittest.TestCase):
    def test_reward_ema(self):
        reward_ema = models.RewardEMA(device="cuda", alpha=0.9)
        mean, std = reward_ema(Tensor([1.0, 2.0, 3.0]))
        print(mean.item(), std.item())
        mean, std = reward_ema(Tensor([1.0, 2.0, 3.0]))
        print(mean.item(), std.item())


class TestWorldModel(unittest.TestCase):
    def test_world_model_init(self):
        obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
                ),
            }
        )
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        print(
            f"DreamerV3 world model has {sum(param.numel() for param in world_model.parameters())} variables."
        )

    def test_world_model_preprocess(self):
        B = 6
        T = 6
        obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
                ),
            }
        )
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
        B = 6
        T = 6
        obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
                ),
            }
        )
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
        B = 2
        T = 2
        obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
                ),
            }
        )
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
        post, context, metrics = world_model._train(data)
        print(*list(metrics.items()), sep="\n")
        self.assertEqual(post["stoch"].numpy().shape, (B, T, 32, 32))
        self.assertEqual(post["deter"].numpy().shape, (B, T, 512))
        self.assertEqual(context["embed"].numpy().shape, (B, T, 4096))
        self.assertEqual(context["feat"].numpy().shape, (B, T, 4096))

class TestImagBehavior(unittest.TestCase):
    def test_imag_behavior_init(self):
        B = 2
        T = 2
        obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
                ),
            }
        )
        act_space = gym.spaces.Discrete(3)
        config = utils.load_config()
        world_model = models.WorldModel(obs_space, act_space, 0, config)
        imag_behavior = models.ImagBehavior(config, world_model)
        


if __name__ == "__main__":
    unittest.main()
