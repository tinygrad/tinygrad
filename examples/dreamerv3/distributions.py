from pytest import param
from tinygrad import Tensor, dtypes

from utils import one_hot, symexp, symlog


class Distribution:
    def __init__(self):
        pass

    def sample(self, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def mode(self):
        raise NotImplementedError


class SampleDist(Distribution):
    def __init__(self, dist: Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    def mean(self):
        samples = self._dist.sample(self._samples)
        return Tensor.mean(samples, 0)

    @property
    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[Tensor.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -Tensor.mean(logprob, 0)


class Categorical(Distribution):
    def __init__(self, probs=None, logits=None):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
            self.logits = Tensor.log(probs)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            self.probs = Tensor.softmax(self.logits, -1)
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]

    def sample(self, sample_shape):
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = Tensor.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))


class OneHotDist(Categorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = Tensor.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = Tensor.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = one_hot(
            Tensor.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = Tensor.softmax(logits, -1)
        self.width = (high - low) / 255
        self.buckets = Tensor.arange(low, high, self.width).to(device)
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(Tensor.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(Tensor.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = Tensor.sum((self.buckets <= x[..., None]).to(dtypes.int32), dim=-1) - 1
        above = len(self.buckets) - Tensor.sum(
            (self.buckets > x[..., None]).to(dtypes.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = Tensor.clip(below, 0, len(self.buckets) - 1)
        above = Tensor.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = Tensor.where(equal, 1, Tensor.abs(self.buckets[below] - x))
        dist_to_above = Tensor.where(equal, 1, Tensor.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - self.logits.exp().sum(-1, keepdim=True).log()
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = Tensor.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = Tensor.abs(self._mode - symlog(value))
            distance = Tensor.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (
                self.absmax / Tensor.clip(Tensor.abs(out), min=self.absmax)
            ).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (
                self.absmax / Tensor.clip(Tensor.abs(out), min=self.absmax)
            ).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = Tensor.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -Tensor.softplus(_logits)
        log_probs1 = -Tensor.softplus(-_logits)

        return Tensor.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class Normal(Distribution):
    def __init__(self, loc, scale, threshold=1):
        self.mean = loc
        self._scale = scale
        self._threshold = threshold


class UnnormalizedHuber(Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            Tensor.sqrt((event - self.mean) ** 2 + self._threshold**2)
            - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = Tensor.clip(
                event, self._low + self._clip, self._high - self._clip
            )
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector:
    def __init__(self, dist: Distribution):
        self._dist = dist

    def _forward(self, x):
        return Tensor.tanh(x)

    def _inverse(self, y):
        y = Tensor.where(
            (Tensor.abs(y) <= 1.0), Tensor.clip(y, -0.99999997, 0.99999997), y
        )
        y = Tensor.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = Tensor.log(2.0)
        return 2.0 * (log2 - x - Tensor.softplus(-2.0 * x))


class Independent(Distribution):
    def __init__(self, dist, reinterpreted_batch_ndims):
        self._dist = dist
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def sample(self, sample_shape=()):
        return self._dist.sample(sample_shape)

    def log_prob(self, value):
        return self._dist.log_prob(value)
