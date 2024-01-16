from utils import _sum_rightmost, numel, symexp, symlog

from tinygrad import Tensor, dtypes


class Distribution:
    def __init__(self):
        pass

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def mode(self) -> Tensor:
        raise NotImplementedError

    def sample(self, sample_shape=()) -> Tensor:
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError

    def entropy(self) -> Tensor:
        raise NotImplementedError


class SampleDist(Distribution):
    def __init__(self, dist: Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def mean(self):
        samples = self._dist.sample(self._samples)
        return samples.mean()

    @property
    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[Tensor.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -Tensor.mean(logprob, 0)

    def log_prob(self, value):
        return self._dist.log_prob(value)

    def sample(self, sample_shape=()):
        return self._dist.sample(sample_shape)


class Categorical(Distribution):
    def __init__(self, probs=None, logits=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if len(probs.shape) < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
            self.logits = Tensor.log(probs)
        else:
            if len(logits.shape) < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.exp().sum(-1, keepdim=True).log()
            self.probs = Tensor.softmax(self.logits, -1)
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.shape[-1]
        self._batch_shape = self._param.shape[:-1] if len(self._param.shape) > 1 else ()

    @property
    def mode(self):
        return self.probs.argmax(axis=-1)

    def sample(self, sample_shape=()):
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = Tensor.multinomial(probs_2d, numel(sample_shape), True).T
        output_shape = sample_shape + self._batch_shape
        output_shape = output_shape if len(output_shape) > 0 else (1,)
        return samples_2d.reshape(output_shape)

    def entropy(self):
        p_log_p = self.logits * self.probs
        return -p_log_p.sum(-1)

    def log_prob(self, value):
        value = value.cast(dtypes.int32).unsqueeze(-1)
        # Tensor.gather uses mlops.Eq, which does not support backward() yet
        # return self.logits.gather(value, dim=-1).squeeze(-1)
        value = Tensor.one_hot(value, self._num_events)
        return (self.logits * value).sum(-1)


class OneHotCategorical(Categorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = Tensor.softmax(logits, -1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = Tensor.log(probs)
            probs = None
        super().__init__(logits=logits, probs=probs)

    @property
    def mode(self):
        _mode = Tensor.one_hot(Tensor.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode.detach() + self.logits - self.logits.detach()

    def sample(self, sample_shape=()):
        probs = self.probs
        num_events = self._num_events
        indices = super().sample(sample_shape)
        sample = Tensor.one_hot(indices, num_events)
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        return sample.detach() + probs - probs.detach()

    def entropy(self):
        return super().entropy()

    def log_prob(self, value):
        return (self.logits * value).sum(-1)


class DiscDist:
    def __init__(self, logits, low=-20.0, high=20.0, transfwd=symlog, transbwd=symexp):
        self.logits = logits
        self.probs = Tensor.softmax(logits, -1)
        self.width = (high - low) / 255
        self.buckets = Tensor.arange(low, high, self.width)
        self.num_buckets = self.buckets.shape[0]
        self.transfwd = transfwd
        self.transbwd = transbwd

    @property
    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(Tensor.sum(_mean, axis=-1, keepdim=True))

    @property
    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(Tensor.sum(_mode, axis=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = Tensor.sum((self.buckets <= x[..., None]).cast(dtypes.int32), axis=-1) - 1
        above = self.num_buckets - Tensor.sum((self.buckets > x[..., None]).cast(dtypes.int32), axis=-1)
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = Tensor.clip(below, 0, self.num_buckets - 1)
        above = Tensor.clip(above, 0, self.num_buckets - 1)
        equal = below == above

        dist_to_below = Tensor.where(equal, 1, Tensor.abs(self.buckets[below] - x))
        dist_to_above = Tensor.where(equal, 1, Tensor.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            Tensor.one_hot(below, num_classes=self.num_buckets) * weight_below[..., None]
            + Tensor.one_hot(above, num_classes=self.num_buckets) * weight_above[..., None]
        )
        log_pred = self.logits - self.logits.exp().sum(-1, keepdim=True).log()
        return (target * log_pred).sum(-1)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    @property
    def mode(self):
        return self._mode

    @property
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

    @property
    def mode(self):
        return symexp(self._mode)

    @property
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
    def __init__(self, dist: Distribution, absmax: int = None):
        self._dist = dist
        self.absmax = absmax

    @property
    def mean(self):
        out = self._dist.mean
        if self.absmax is not None:
            out = Tensor.clip(out, -self.absmax, self.absmax).detach()
        return out

    @property
    def mode(self):
        out = self._dist.mode
        if self.absmax is not None:
            out = Tensor.clip(out, -self.absmax, self.absmax).detach()
        return out

    def entropy(self):
        return self._dist.entropy()

    def sample(self, sample_shape=()):
        out = self._dist.sample(sample_shape)
        if self.absmax is not None:
            out = Tensor.clip(out, -self.absmax, self.absmax)
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli(Distribution):
    def __init__(self, probs=None, logits=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if not isinstance(probs, Tensor):
                probs = Tensor(probs)
            self.probs = probs
            probs = probs.clip(min=1e-7, max=1.0 - 1e-7)
            self.logits = Tensor.log(probs) - Tensor.log(-probs)
        else:
            if not isinstance(logits, Tensor):
                logits = Tensor(logits)
            # Normalize
            self.logits = logits
            self.probs = Tensor.sigmoid(self.logits)
        self._param = self.probs if probs is not None else self.logits
        self._batch_shape = self._param.shape

    @property
    def mean(self):
        return self.probs

    @property
    def mode(self):
        _mode = Tensor.trunc(self.mean)
        return _mode.detach() + self.mean - self.mean.detach()

    def entropy(self):
        return -self.log_prob(self.probs)

    def sample(self, sample_shape=()):
        output_shape = sample_shape + self._batch_shape
        output_shape = output_shape if len(output_shape) > 0 else (1,)
        eps = Tensor.rand(output_shape)
        return (eps < self.probs).cast(self.probs.dtype)

    def log_prob(self, x):
        x = x.unsqueeze(-1)
        log_probs0 = -Tensor.softplus(self.logits)
        log_probs1 = -Tensor.softplus(-self.logits)

        return log_probs0 * (1 - x) + log_probs1 * x


class Normal(Distribution):
    def __init__(self, loc, scale, threshold=1):
        self._loc = loc
        self._scale = scale
        self._threshold = threshold

    @property
    def mean(self):
        return self._loc

    @property
    def mode(self):
        return self._loc

    def entropy(self):
        return 0.5 * self._scale.log()

    def sample(self, sample_shape=()):
        return self._loc + self._scale * Tensor.randn(sample_shape)

    def log_prob(self, event):
        return -(Tensor.sqrt((event - self._loc) ** 2 + self._threshold**2) - self._threshold)


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
            clipped = Tensor.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class Independent(Distribution):
    def __init__(self, dist, reinterpreted_batch_ndims):
        self._dist = dist
        self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

    @property
    def mean(self):
        return self._dist.mean

    @property
    def mode(self):
        return self._dist.mode

    def sample(self, sample_shape=()):
        return self._dist.sample(sample_shape)

    def log_prob(self, event):
        log_prob = self._dist.log_prob(event)
        return _sum_rightmost(log_prob, self._reinterpreted_batch_ndims)

    def entropy(self):
        entropy = self._dist.entropy()
        return _sum_rightmost(entropy, self._reinterpreted_batch_ndims)


class Uniform(Distribution):
    def __init__(self, low, high):
        self._low = low
        self._high = high
        self._batch_shape = self._low.shape if isinstance(low, Tensor) else ()

    @property
    def mean(self):
        return (self._low + self._high) / 2.0

    def entropy(self):
        return Tensor.log(self._high - self._low)

    def sample(self, sample_shape=()):
        output_shape = sample_shape + self._batch_shape
        output_shape = output_shape if len(output_shape) > 0 else (1,)
        return self._low + (self._high - self._low) * Tensor.rand(output_shape)

    def log_prob(self, value):
        lb = (self.low <= value).cast(self.low.dtype)
        ub = (self.high >= value).cast(self.low.dtype)
        return Tensor.log(lb.mul(ub)) - Tensor.log(self.high - self.low)


def _unwrap_dist(dist):
    if isinstance(dist, (Independent, SampleDist, ContDist)):
        return _unwrap_dist(dist._dist)
    else:
        return dist


def kl_divergence(dist1, dist2):
    if isinstance(dist1, Independent) and isinstance(dist2, Independent):
        return _kl_independent_independent(dist1, dist2)
    if isinstance(dist1, Categorical) and isinstance(dist2, Categorical):
        return _kl_categorical_categorical(dist1, dist2)
    else:
        raise NotImplementedError


def _kl_independent_independent(dist1: Independent, dist2: Independent):
    if dist1._reinterpreted_batch_ndims != dist2._reinterpreted_batch_ndims:
        raise NotImplementedError
    kldiv = kl_divergence(dist1._dist, dist2._dist)
    return _sum_rightmost(kldiv, dist1._reinterpreted_batch_ndims)


def _kl_categorical_categorical(p: Categorical, q: Categorical):
    return (p.probs * (p.logits - q.logits)).sum(-1)
