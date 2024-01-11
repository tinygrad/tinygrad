import math
import re

import distributions
import numpy as np
import utils

from tinygrad import Tensor, nn


class GRUCell:
    def __init__(self, inp_size, size, norm=True):
        self._inp_size = inp_size
        self._size = size
        self.layers = [nn.Linear(inp_size + size, 3 * size, bias=False)]
        if norm:
            self.layers.append(nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def __call__(self, inputs: Tensor, state: Tensor):
        parts = Tensor.cat(inputs, state, dim=-1).sequential(self.layers)
        reset, cand, update = Tensor.split(parts, [self._size] * 3, -1)
        reset = Tensor.sigmoid(reset)
        cand = Tensor.tanh(reset * cand)
        update = Tensor.sigmoid(update - 1)
        output = update * cand + (1 - update) * state
        return output, output


class Conv2dSamePad(nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def __call__(self, x):
        ih, iw = x.shape[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride, d=self.dilation
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride, d=self.dilation
        )

        if pad_h > 0 or pad_w > 0:
            wleft = pad_w // 2
            wright = pad_w - wleft
            hleft = pad_h // 2
            hright = pad_h - hleft
            padding_values = ((0, 0), (0, 0), (wleft, wright), (hleft, hright))
            x = Tensor.pad(x, padding_values)

        return x.conv2d(
            self.weight,
            self.bias,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvEncoder:
    def __init__(
        self,
        input_shape,
        depth=32,
        norm=True,
        act="silu",
        kernel_size=4,
        minres=4,
    ):
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        act = utils.get_act(act)

        layers = []
        for _ in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=True,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act)
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        [utils.weight_init(layer) for layer in layers]
        self.layers = layers

    def __call__(self, obs: Tensor):
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = x.sequential(self.layers)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape(x.shape[0], -1)
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(obs.shape[:2] + (-1,))


class ConvDecoder:
    def __init__(
        self,
        feat_size,
        shape=(64, 64, 3),
        depth=32,
        act="elu",
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch
        act = utils.get_act(act)

        self._linear_layer = nn.Linear(feat_size, out_ch)
        utils.uniform_weight_init(outscale)(self._linear_layer)
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act)
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
        [utils.weight_init(m) for m in layers[:-1]]
        utils.uniform_weight_init(outscale)(layers[-1])
        self.layers = layers

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def __call__(self, features):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = x.sequential(self.layers)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = Tensor.sigmoid(mean)
        else:
            mean = mean + 0.5
        return mean


class MLP:
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="silu",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._dist = dist
        self._std = std if isinstance(std, str) else Tensor([std], device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device
        act = utils.get_act(act)

        self.layers = []
        for i in range(layers):
            self.layers.append(nn.Linear(inp_dim, units, bias=False))
            if norm:
                self.layers.append(nn.LayerNorm(units, eps=1e-03))
            if act:
                self.layers.append(act)
            if i == 0:
                inp_dim = units
        [utils.weight_init(layer) for layer in self.layers]

        if isinstance(self._shape, dict):
            self.mean_layer = dict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape).item())
                utils.uniform_weight_init(outscale)(self.mean_layer[name])
            if isinstance(self._std, str) and self._std == "learned":
                assert dist in ("normal", "trunc_normal"), dist
                self.std_layer = dict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape).item())
                    utils.uniform_weight_init(outscale)(self.std_layer[name])
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape).item())
            utils.uniform_weight_init(outscale)(self.mean_layer)
            if isinstance(self._std, str) and self._std == "learned":
                assert dist in ("normal", "trunc_normal"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape).item())
                utils.uniform_weight_init(outscale)(self.std_layer)

    def __call__(self, features):
        x = features
        if self._symlog_inputs:
            x = utils.symlog(x)
        out = x.sequential(self.layers)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if isinstance(self._std, str) and self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if isinstance(self._std, str) and self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if self._dist == "normal":
            std = (self._max_std - self._min_std) * Tensor.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = distributions.Normal(Tensor.tanh(mean), std)
            dist = distributions.ContDist(
                distributions.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "normal_std_fixed":
            dist = distributions.Normal(mean, self._std)
            dist = distributions.ContDist(
                distributions.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "trunc_normal":
            mean = Tensor.tanh(mean)
            std = 2 * Tensor.sigmoid(std / 2) + self._min_std
            dist = distributions.SafeTruncatedNormal(mean, std, -1, 1)
            dist = distributions.ContDist(
                distributions.Independent(dist, 1), absmax=self._absmax
            )
        elif self._dist == "onehot":
            dist = distributions.OneHotCategorical(
                mean, unimix_ratio=self._unimix_ratio
            )
        elif dist == "binary":
            dist = distributions.Independent(
                distributions.Bernoulli(logits=mean), len(shape)
            )
        elif dist == "symlog_disc":
            dist = distributions.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = distributions.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class ImgChLayerNorm:
    def __init__(self, ch, eps=1e-03):
        self.norm = nn.LayerNorm(ch, eps=eps)

    def __call__(self, x: Tensor):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class MultiEncoder:
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act: bool = Tensor.silu,
        norm: bool = True,
        cnn_depth: int = 32,
        kernel_size: int = 4,
        minres: int = 4,
        mlp_layers: int = 4,
        mlp_units: int = 256,
        symlog_inputs: bool = False,
    ):
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: (v,) if isinstance(v, int) else v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        # print("Encoder CNN shapes:", self.cnn_shapes)
        # print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, norm, act, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def __call__(self, obs):
        outputs = []
        if self.cnn_shapes:
            cnn_inputs = [obs[k] for k in self.cnn_shapes]
            inputs = Tensor.cat(*cnn_inputs, dim=-1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            mlp_inputs = [obs[k] for k in self.mlp_shapes]
            inputs = Tensor.cat(*mlp_inputs, dim=-1)
            outputs.append(self._mlp(inputs))
        outputs = Tensor.cat(*outputs, dim=-1)
        return outputs


class MultiDecoder:
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act="elu",
        norm: bool = True,
        cnn_depth: int = 32,
        kernel_size: int = 4,
        minres: int = 4,
        mlp_layers: int = 4,
        mlp_units: int = 256,
        cnn_sigmoid: bool = False,
        image_dist: str = "mse",
        vector_dist: str = "symlog_mse",
        outscale: float = 1.0,
    ):
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {
            k: (v,) if isinstance(v, int) else v
            for k, v in shapes.items()
            if k not in excluded
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        # print("Decoder CNN shapes:", self.cnn_shapes)
        # print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def __call__(self, features):
        dists = {}
        if self.cnn_shapes:
            outputs = self._cnn(features)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = Tensor.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return distributions.ContDist(
                distributions.Independent(distributions.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return distributions.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class RSSM:
    def __init__(
        self,
        num_actions,
        embed,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=30,
        act="silu",
        norm=True,
        mean_act="none",
        std_act="sigmoid",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        device=None,
    ):
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        act = utils.get_act(act)

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act)
        self._img_in_layers = inp_layers
        utils.weight_init(self._img_in_layers)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        utils.weight_init(self._cell)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act)
        self._img_out_layers = img_out_layers
        utils.weight_init(self._img_out_layers)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act)
        self._obs_out_layers = obs_out_layers
        utils.weight_init(self._obs_out_layers)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
        utils.uniform_weight_init(1.0)(self._obs_stat_layer)
        utils.uniform_weight_init(1.0)(self._imgs_stat_layer)

        if self._initial == "learned":
            self.W = Tensor.zeros(1, self._deter, device=self._device)
            self.W.requires_grad = True

    def initial(self, batch_size):
        deter = Tensor.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=Tensor.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=Tensor.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=Tensor.zeros([batch_size, self._stoch]).to(self._device),
                std=Tensor.zeros([batch_size, self._stoch]).to(self._device),
                stoch=Tensor.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = Tensor.tanh(self.W).repeat((batch_size, 1))
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        def swap(x):
            return x.permute([1, 0] + list(range(2, len(x.shape))))

        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = utils.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (time, batch, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        def swap(x):
            return x.permute([1, 0] + list(range(2, len(x.shape))))

        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = utils.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return Tensor.cat(stoch, state["deter"], dim=-1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = distributions.Independent(
                distributions.OneHotCategorical(logit, unimix_ratio=self._unimix_ratio),
                1,
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = distributions.ContDist(
                distributions.Independent(distributions.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        B = is_first.shape[0]
        # initialize all prev_state
        if prev_state is None or Tensor.sum(is_first) == B:
            prev_state = self.initial(B)
            prev_action = Tensor.zeros((B, self._num_actions)).to(self._device)
        # overwrite the prev_state only where is_first=True
        elif Tensor.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(B)
            for key, val in prev_state.items():
                is_first_r = Tensor.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = Tensor.cat(prior["deter"], embed, dim=-1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = x.sequential(self._obs_out_layers)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = Tensor.cat(prev_stoch, prev_action, dim=-1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = x.sequential(self._img_in_layers)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, deter)
        # (batch, deter) -> (batch, hidden)
        x = x.sequential(self._img_out_layers)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = deter.sequential(self._img_out_layers)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = Tensor.split(x, [self._stoch] * 2, -1)
            mean = self._mean_act(mean)
            std = self._std_act(std) + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        def kld(x, y):
            return distributions.kl_divergence(x, y).mean(-1)

        def dist(x):
            return self.get_dist(x)

        def sg(x):
            return {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        rep_loss = Tensor.maximum(rep_loss, free)
        dyn_loss = Tensor.maximum(dyn_loss, free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss
