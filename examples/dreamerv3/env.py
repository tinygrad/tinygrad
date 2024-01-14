import io
import os
import pathlib
from collections import OrderedDict

import gymnasium as gym
import numpy as np


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


class AtariEnv(gym.Env):
    def __init__(self, env):
        env = gym.make(env, obs_type="rgb", frameskip=4, repeat_action_probability=0.0, render_mode="rgb_array")
        env = gym.wrappers.ResizeObservation(env, (64, 64))
        env = gym.wrappers.TimeLimit(env, 108000)
        self.env = env

    @property
    def observation_space(self):
        return gym.spaces.Dict({"image": self.env.observation_space})

    @property
    def action_space(self):
        return self.env.action_space

    def _wrap_obs(self, obs, terminated, truncated, is_first):
        obs = {"image": obs}
        obs["is_last"] = terminated or truncated
        obs["is_terminal"] = terminated
        obs["is_first"] = is_first
        return obs

    def step(self, action):
        if isinstance(action, dict):
            action = action["action"]
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = action.argmax(-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        obs = self._wrap_obs(obs, terminated, truncated, False)
        return obs, reward, done, info

    def reset(self):
        obs, _ = self.env.reset()
        obs = self._wrap_obs(obs, False, False, True)
        return obs


def make_envs(config):
    if config.env.startswith("ALE"):
        # [2]: "We down-scale the 84 × 84 grayscale images to 64 × 64 pixels so that
        # we can apply the convolutional architecture of DreamerV1."
        # ...
        # "We follow the evaluation protocol of Machado et al. (2018) with 200M
        # environment steps, action repeat of 4, a time limit of 108,000 steps per
        # episode that correspond to 30 minutes of game play, no access to life
        # information, full action space, and sticky actions. Because the world
        # model integrates information over time, DreamerV2 does not use frame
        # stacking."
        # However, in Danijar's repo, Atari100k experiments are configured as:
        # noop=30, 64x64x3 (no grayscaling), sticky actions=False,
        # full action space=False,
        envs = [AtariEnv(config.env) for _ in range(config.num_envs)]
    else:
        raise ValueError(f"Unknown env: {config.env}.")
    return envs


def simulate(agent, envs, cache, directory, logger, is_eval=False, limit=None, steps=0, episodes=0, state=None):
    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, reward = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary
        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            for index, result in zip(indices, results):
                t = result
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, index, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        action, agent_state = agent(obs, done, agent_state)
        if isinstance(action, dict):
            action = [{k: np.array(action[k][i].numpy()) for k in action} for i in range(len(envs))]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for index, (a, result) in enumerate(zip(action, results)):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, index, transition)

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                save_episodes(directory, {i: cache[i]})
                length = len(cache[i]["reward"]) - 1
                score = float(np.array(cache[i]["reward"]).sum())
                video = cache[i]["image"]
                # record logs given from environments
                for key in list(cache[i].keys()):
                    if "log_" in key:
                        logger.scalar(key, float(np.array(cache[i][key]).sum()))
                        # log items won't be used later
                        cache[i].pop(key)

                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar("dataset_size", step_in_dataset)
                    logger.scalar("train_return", score)
                    logger.scalar("train_length", length)
                    logger.scalar("train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    if "eval_lengths" not in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    logger.video("eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar("eval_return", score)
                        logger.scalar("eval_length", length)
                        logger.scalar("eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


def make_dataset(episodes, config):
    generator = sample_episodes(episodes, config.batch_length)
    dataset = from_generator(generator, config.batch_size)
    return dataset


def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array([len(next(iter(episode.values()))) for episode in episodes.values()])
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {k: v[index : min(index + length, total)].copy() for k, v in episode.items() if "log_" not in k}
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {k: np.append(ret[k], v[index : min(index + possible, total)].copy(), axis=0) for k, v in episode.items() if "log_" not in k}
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            try:
                data[key] = np.stack(data[key], 0)
            except:
                breakpoint()
        yield data


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if not dataset_size or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size:
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True
