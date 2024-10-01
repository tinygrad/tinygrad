#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tinygrad import Tensor, TinyJit
np.bool = np.bool_

def train(continuous, env, agent, max_episode, warmup, save_model_dir, max_episode_length, logger, save_per_epochs):
    Tensor.training = True
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    s_t = None
    print(f'max_episode: {max_episode}. save_per_epochs: {save_per_epochs}')
    while episode < max_episode:
        while True:
            if s_t is None:
                s_t = env.reset()
                if isinstance(s_t, tuple):
                    # print(f'Tuple detected for s_t: {s_t}')
                    s_t = np.array(s_t[0]).astype(np.float32)
                agent.reset(s_t)

            # agent pick action ...
            # args.warmup: time without training but only filling the memory
            if step <= warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(s_t)

            #print(f'action: {action}')

            # env response with next_observation, reward, terminate_info
            if not continuous:
                action = action.reshape(1,).astype(int)[0]
            s_t1, r_t, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            #print(f's_t1, r_t, done: {s_t1}, {r_t}, {done}')
            #print(f's_t1 type: {type(s_t1)}')
            s_t1 = np.array(s_t1)

            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True

            # agent observe and update policy
            agent.observe(r_t, s_t1, done)
            if step > warmup:
                agent.update_policy()

            # update
            step += 1
            episode_steps += 1
            episode_reward += r_t
            s_t = s_t1
            # s_t = deepcopy(s_t1)

            if done:  # end of an episode
                print("Ep:{0} | R:{1:.4f}".format(episode, episode_reward))
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(episode, episode_reward)
                )

                agent.memory.append(
                    s_t,
                    agent.select_action(s_t),
                    0., True
                )

                # reset
                s_t = None
                episode_steps =  0
                episode_reward = 0.
                episode += 1
                # break to next episode
                break
        # [optional] save intermideate model every run through of 32 episodes
        if step > warmup and episode > 0 and episode % save_per_epochs == 0:
            agent.save_model(save_model_dir)
            logger.info("### Model Saved before Ep:{0} ###".format(episode))

@TinyJit
# @Tensor.test()
def test(env, agent, model_path, test_episode, max_episode_length, logger):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()

    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    episode_steps = 0
    episode_reward = 0.
    s_t = None
    for i in range(test_episode):
        while True:
            if s_t is None:
                s_t = env.reset()
                agent.reset(s_t)

            action = policy(s_t)
            s_t, r_t, done, _, _ = env.step(action)
            s_t = np.array(s_t)
            episode_steps += 1
            episode_reward += r_t
            if max_episode_length and episode_steps >= max_episode_length - 1:
                done = True
            if done:  # end of an episode
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(i+1, episode_reward)
                )
                s_t = None
                break
