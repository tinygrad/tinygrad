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

#stft function was ripped out of librosa, which pulled too many dependencies
def stft(y, n_fft = 2048, hop_length = None, win_length = None, pad_mode = "constant"):
    win_length = win_length if win_length is not None else n_fft
    hop_length = hop_length if hop_length is not None else int(win_length // 4)
    fft = np.fft
    dtype = np.complex64
    assert hop_length > 0 and isinstance(hop_length, int), f"hop_length={hop_length} must be a positive integer"
    assert pad_mode in ["constant", "reflect"], f"pad_mode='{pad_mode}' is not supported. Choose 'constant' or 'reflect'."

    fft_window = np.hanning(win_length)
    axis = -1
    n = fft_window.shape[axis]
    lpad = int((n_fft - n) // 2)
    lengths = [(0, 0)] * fft_window.ndim
    lengths[axis] = (lpad, int(n_fft - n - lpad))
    fft_window = np.pad(fft_window, lengths, "constant")    
    shape = [1] * (1 + y.ndim)
    shape[-2] = fft_window.shape[0]
    fft_window = fft_window.reshape(shape)

    padding = [(0, 0) for _ in range(y.ndim)]
    start = int(np.ceil(n_fft // 2 / hop_length))
    padding[-1] = (n_fft // 2, n_fft // 2)
    y = np.pad(y, padding, mode=pad_mode)

    assert n_fft <= y.shape[-1], f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}"
    # split it into frames
    out_strides = y[..., start:].strides + tuple([y[..., start:].strides[-1]])
    x_shape_trimmed = list(y[..., start:].shape)
    x_shape_trimmed[-1] -= n_fft - 1
    out_shape = tuple(x_shape_trimmed) + tuple([n_fft])
    y_frames = np.lib.stride_tricks.as_strided(y[..., start:], strides=out_strides, shape=out_shape)
    y_frames = np.moveaxis(y_frames, -1, -2)
    slices = [slice(None)] * y_frames.ndim
    slices[-1] = slice(0, None, hop_length)
    y_frames = y_frames[tuple(slices)]

    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    stft_matrix = np.zeros(shape, dtype=dtype, order="F")
    MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10
    n_columns = int(MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize))
    n_columns = max(n_columns, 1)
    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])
        stft_matrix[..., bl_s:bl_t] = fft.rfft(fft_window * y_frames[..., bl_s:bl_t], axis=-2)
    return stft_matrix

#was generated with mel(sr=16000, n_fft=512, n_mels=80)
FILTER_BANK = np.expand_dims(np.load(pathlib.Path(__file__).parent / "rnnt_mel_filters.npz")["mel_80"], 0)

def feature_extract(x, x_lens):
  x_lens = np.ceil((x_lens / 160) / 3).astype(np.int32)

  # pre-emphasis
  x = np.concatenate((np.expand_dims(x[:, 0], 1), x[:, 1:] - 0.97 * x[:, :-1]), axis=1)

  # stft
  x = stft(x, n_fft=512, hop_length=160, win_length=320, pad_mode="reflect")
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
