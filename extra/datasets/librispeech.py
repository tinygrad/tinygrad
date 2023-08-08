import functools
import json
import math
import pathlib
import numpy as np
import soundfile

"""
The dataset has to be downloaded manually from https://www.openslr.org/12/ and put in `extra/datasets/librispeech`.
For mlperf validation the dev-clean dataset is used.

Then all the flacs have to be converted to wav using something like:
```sh
for file in $(find * | grep flac); do ffmpeg -i $file -ar 16k "$(dirname $file)/$(basename $file .flac).wav"; done
```

Then this [file](https://github.com/mlcommons/inference/blob/master/speech_recognition/rnnt/dev-clean-wav.json) has to also be put in `extra/datasets/librispeech`.
"""
BASEDIR = pathlib.Path(__file__).parent / "librispeech"
with open(BASEDIR / "dev-clean-wav.json") as f:
  ci = json.load(f)

#stft and mel functions were ripped out of librosa, which pulled too many dependencies
def mel_frequencies(n_mels, fmin, fmax):
    #use approximation, not exact, should be good enough, need to check
  def hz_to_mel(frequencies):
    frequencies = np.asanyarray(frequencies)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels
  
  def mel_to_hz(mels):
    mels = np.asanyarray(mels)
    freqs = 200.0 / 3 * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0.0) / (200.0 / 3)
    logstep = np.log(6.4) / 27.0

    if mels.ndim:
      log_t = mels >= min_log_mel
      freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
      freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs

  min_mel = hz_to_mel(fmin)
  max_mel = hz_to_mel(fmax)
  mels = np.linspace(min_mel, max_mel, n_mels)
  hz = mel_to_hz(mels)
  return hz

def mel(sr, n_fft, n_mels):
  n_mels = int(n_mels)
  weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
  fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
  mel_f = mel_frequencies(n_mels=n_mels + 2, fmin=0, fmax=float(sr)/2)
  fdiff = np.diff(mel_f)
  ramps = np.subtract.outer(mel_f, fftfreqs)

  for i in range(n_mels):
    lower = -ramps[i] / fdiff[i]
    upper = ramps[i + 2] / fdiff[i + 1]
    weights[i] = np.maximum(0, np.minimum(lower, upper))

  if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
    print('Empty filters in mel frequency basis. Some channels will produce empty responses.')

  return weights

def frame(x, frame_length, hop_length):
  out_strides = x.strides + tuple([x.strides[-1]])
  out_strides = x.strides + tuple([x.strides[-1]])

  x_shape_trimmed = list(x.shape)
  x_shape_trimmed[-1] -= frame_length - 1
  out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
  xw = np.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)
  xw = np.moveaxis(xw, -1, -2)

  slices = [slice(None)] * xw.ndim
  slices[-1] = slice(0, None, hop_length)
  return xw[tuple(slices)]

def tinystft(y, n_fft, hop_length, window):
  import warnings
  warnings.simplefilter("ignore", np.ComplexWarning)
  
  def pad_center(data: np.ndarray, size: int) -> np.ndarray:
    axis = -1
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))
    assert lpad >= 0, f"Target size ({size:d}) must be at least input size ({n:d})"
    return np.pad(data, lengths, "constant")
  def expand_to(x, ndim, axes):
    shape = [1] * ndim
    for i, axi in enumerate([axes]):
      shape[axi] = x.shape[i]
    return x.reshape(shape)
  fft_window = pad_center(window, size=n_fft)
  fft_window = expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

  assert n_fft <= y.shape[-1], f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}"

  padding = [(0, 0) for _ in range(y.ndim)]

  start_k = int(np.ceil(n_fft // 2 / hop_length))

  tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

  if tail_k <= start_k:
    start = 0
    extra = 0
    padding[-1] = (n_fft // 2, n_fft // 2)
    y = np.pad(y, padding, mode="constant")
  else:
    start = start_k * hop_length - n_fft // 2
    padding[-1] = (n_fft // 2, 0)

    y_pre = np.pad(
      y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
      padding,
      mode="constant",
    )
    y_frames_pre = frame(y_pre, frame_length=n_fft, hop_length=hop_length)
    y_frames_pre = y_frames_pre[..., :start_k]

    extra = y_frames_pre.shape[-1]

    if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
      padding[-1] = (0, n_fft // 2)
      y_post = np.pad(
        y[..., (tail_k) * hop_length - n_fft // 2 :], padding, mode="constant"
      )
      y_frames_post = frame(
        y_post, frame_length=n_fft, hop_length=hop_length
      )
      extra += y_frames_post.shape[-1]
    else:
      post_shape = list(y_frames_pre.shape)
      post_shape[-1] = 0
      y_frames_post = np.empty_like(y_frames_pre, shape=post_shape, dtype=np.float32)

  y_frames = frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)

  shape = list(y_frames.shape)
  shape[-2] = 1 + n_fft // 2
  shape[-1] += extra
  stft_matrix = np.zeros(shape, order="F", dtype=np.float32)

  if extra > 0:
    off_start = y_frames_pre.shape[-1]
    stft_matrix[..., :off_start] = np.fft.rfft(fft_window * y_frames_pre, axis=-2)

    off_end = y_frames_post.shape[-1]
    if off_end > 0:
        stft_matrix[..., -off_end:] = np.fft.rfft(fft_window * y_frames_post, axis=-2)
  else:
    off_start = 0

  MAX_MEM_BLOCK = 2**8 * 2**10
  n_columns = int(
    MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize)
  )
  n_columns = max(n_columns, 1)
  for bl_s in range(0, y_frames.shape[-1], n_columns):
    bl_t = min(bl_s + n_columns, y_frames.shape[-1])
    stft_matrix[..., bl_s + off_start : bl_t + off_start] = np.fft.rfft(
        fft_window * y_frames[..., bl_s:bl_t], axis=-2
    )

  return stft_matrix

@functools.lru_cache(None)
def get_window(n_fft): return (1 - np.cos(2 * math.pi * np.arange(n_fft) / n_fft)) / 2

FILTER_BANK = np.expand_dims(mel(sr=16000, n_fft=512, n_mels=80), 0)
WINDOW = get_window(320)

def feature_extract(x, x_lens):
  x_lens = np.ceil((x_lens / 160) / 3).astype(np.int32)

  # pre-emphasis
  x = np.concatenate((np.expand_dims(x[:, 0], 1), x[:, 1:] - 0.97 * x[:, :-1]), axis=1)

  # stft
  x = tinystft(x, n_fft=512, window=WINDOW, hop_length=160, win_length=320, center=True, pad_mode="reflect")
  x = np.stack((x.real, x.imag), axis=-1)

  # power spectrum
  x = (x**2).sum(-1)

  # mel filter bank
  x = np.matmul(FILTER_BANK, x)

  # log
  x = np.log(x + 1e-20)

  # feature splice
  seq = [x]
  for i in range(1, 3):
    tmp = np.zeros_like(x)
    tmp[:, :, :-i] = x[:, :, i:]
    seq.append(tmp)
  features = np.concatenate(seq, axis=1)[:, :, ::3]

  # normalize
  features_mean = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
  features_std = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
  for i in range(features.shape[0]):
    features_mean[i, :] = features[i, :, :x_lens[i]].mean(axis=1)
    features_std[i, :] = features[i, :, :x_lens[i]].std(axis=1, ddof=1)
  features_std += 1e-5
  features = (features - np.expand_dims(features_mean, 2)) / np.expand_dims(features_std, 2)

  return features.transpose(2, 0, 1), x_lens.astype(np.float32)

def load_wav(file):
  sample = soundfile.read(file)[0].astype(np.float32)
  return sample, sample.shape[0]

def iterate(bs=1, start=0):
  print(f"there are {len(ci)} samples in the dataset")
  for i in range(start, len(ci), bs):
    samples, sample_lens = zip(*[load_wav(BASEDIR / v["files"][0]["fname"]) for v in ci[i : i + bs]])
    samples = list(samples)
    # pad to same length
    max_len = max(sample_lens)
    for j in range(len(samples)):
      samples[j] = np.pad(samples[j], (0, max_len - sample_lens[j]), "constant")
    samples, sample_lens = np.array(samples), np.array(sample_lens)

    yield feature_extract(samples, sample_lens), np.array([v["transcript"] for v in ci[i : i + bs]])

if __name__ == "__main__":
  X, Y = next(iterate())
  print(X[0].shape, Y.shape)
