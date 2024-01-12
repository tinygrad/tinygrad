import argparse
import json
import pathlib
import random
import time

import numpy as np
import yaml
from tensorboardX import SummaryWriter

from tinygrad import Tensor, nn


def symlog(x):
    return Tensor.sign(x) * Tensor.log(Tensor.abs(x) + 1.0)


def symexp(x):
    return Tensor.sign(x) * (Tensor.exp(Tensor.abs(x)) - 1.0)


def cumprod(x: Tensor, axis):
    dtype = x.dtype
    # why implement cumprod when you can use math instead?
    return x.log().cumsum(axis).exp().cast(dtype)


def quantile(input, q):
    # TODO: optimize this to not use numpy
    return Tensor(np.quantile(input.numpy(), q.numpy()), dtype=input.dtype)


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        def inp(x):
            return (_input[x] for _input in inputs)

        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = Tensor.cat(outputs, last, dim=-1)
    outputs = Tensor.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = outputs.flip(1)
    outputs = outputs[:]
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * Tensor.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = Tensor.zeros_like(value[-1])
    next_values = Tensor.cat(value[1:], bootstrap[None], dim=0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        if "e-" in default or "." in default:
            try:
                return float(x)
            except ValueError:
                pass
        if "e" in default:
            try:
                return int(x)
            except ValueError:
                pass
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:

        def inp(x):
            return (_input[x] for _input in inputs)

        last = fn(last, *inp(index))
        if flag:
            if isinstance(last, dict):
                outputs = {key: value.unsqueeze(0) for key, value in last.items()}
            else:
                outputs = []
                for _last in last:
                    if isinstance(_last, dict):
                        outputs.append(
                            {key: value.unsqueeze(0) for key, value in _last.items()}
                        )
                    else:
                        outputs.append(_last.unsqueeze(0))
            flag = False
        else:
            if isinstance(last, dict):
                for key in last.keys():
                    outputs[key] = Tensor.cat(
                        outputs[key], last[key].unsqueeze(0), dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if isinstance(last[j], dict):
                        for key in last[j].keys():
                            outputs[j][key] = Tensor.cat(
                                outputs[j][key], last[j][key].unsqueeze(0), dim=0
                            )
                    else:
                        outputs[j] = Tensor.cat(outputs[j], last[j].unsqueeze(0), dim=0)
    if isinstance(outputs, dict):
        outputs = [outputs]
    return outputs


def weight_init(m):
    if isinstance(m, nn.Linear):
        out_num, in_num = m.weight.shape
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        m.weight = Tensor.normal(m.weight.shape, mean=0.0, std=std).clip(
            -2.0 * std, 2.0 * std
        )
        if m.bias is not None:
            m.bias = Tensor.zeros_like(m.bias)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        out_num, in_num = m.weight.shape[:2]
        denoms = (in_num + out_num) * space / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        m.weight = Tensor.normal(m.weight.shape, mean=0.0, std=std).clip(
            -2.0 * std, 2.0 * std
        )
        if m.bias is not None:
            m.bias = Tensor.zeros_like(m.bias)
    elif isinstance(m, nn.LayerNorm):
        m.weight = 1.0
        m.bias = 0.0


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            out_num, in_num = m.weight.shape
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            m.weight = Tensor.uniform(m.weight.shape, low=-limit, high=limit)
            if m.bias is not None:
                m.bias = Tensor.zeros_like(m.bias)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            out_num, in_num = m.weight.shape[:2]
            denoms = (in_num + out_num) * space / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            m.weight = Tensor.uniform(m.weight.shape, low=-limit, high=limit)
            if m.bias is not None:
                m.bias = Tensor.zeros_like(m.bias)
        elif isinstance(m, nn.LayerNorm):
            m.weight = 1.0
            m.bias = 0.0

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": Tensor.mean(tensor).numpy(),
        "std": Tensor.std(tensor).numpy(),
        "min": Tensor.min(tensor).numpy(),
        "max": Tensor.max(tensor).numpy(),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    Tensor.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _sum_rightmost(value, dim):
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def numel(shape):
    return int(np.prod(shape)) if shape else 1


def get_act(act):
    if isinstance(act, str):
        if act == "none":
            return lambda x: x
        else:
            act = getattr(Tensor, act)
    return act


def clip_grad_norm_(parameters, max_norm):
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return Tensor(0.0)

    def l2_norm(x):
        return Tensor.sqrt(Tensor.sum(Tensor.square(x)))

    norms = [l2_norm(g) for g in grads]
    total_norm = l2_norm(Tensor.stack(norms))
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = Tensor.maximum(clip_coef, 1.0)
    for g in grads:
        g *= clip_coef
    return total_norm


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr: float = 1e-3,
        eps: float = 1e-5,
        grad_clip: float = 100,
        opt: str = "adam",
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.grad_clip = grad_clip
        # in case we get strings e.g. 1e-5
        lr = float(lr)
        eps = float(eps)
        if opt == "adam":
            self.opt = nn.optim.Adam(self.parameters, lr=lr, eps=eps)
        elif opt == "sgd":
            self.opt = nn.optim.SGD(self.parameters, lr=lr)
        elif opt == "momentum":
            self.opt = nn.optim.SGD(self.parameters, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        grad_norm = clip_grad_norm_(self.parameters, self.grad_clip)
        self.opt.step()
        return {f"{self.name}_grad_norm": grad_norm.item()}


def load_config():
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text()
    )
    parser = argparse.ArgumentParser()
    for key, value in sorted(configs.items(), key=lambda x: x[0]):
        arg_type = args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_known_args()


class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)
