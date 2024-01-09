from tinygrad import Tensor
import unittest
import models
import gymnasium as gym
import numpy as np


class TestRewardEMA(unittest.TestCase):
    def test_reward_ema(self):
        reward_ema = models.RewardEMA(device="cuda", alpha=0.9)
        mean, std = reward_ema(Tensor([1.0, 2.0, 3.0]))
        print(mean.item(), std.item())
        mean, std = reward_ema(Tensor([1.0, 2.0, 3.0]))
        print(mean.item(), std.item())


class TestWorldModel(unittest.TestCase):
    def test_world_model(self):
        obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
                ),
            }
        )
        world_model = models.WorldModel(obs_space, None, 0, {})
        world_model(Tensor([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
