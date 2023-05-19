import json
import os
import numpy as np
import librosa

BASEDIR = "/home/woze/projects/tinygrad/datasets/"
ci = json.load(open(os.path.join(BASEDIR, "dev-clean-wav.json")))

FILTER_BANK = np.expand_dims(librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000), 0)

def feature_extract(file):
  x = np.expand_dims(librosa.load(file, sr=16000)[0], 0).astype(np.float32)

  # pre-emphasis
  x = np.concatenate((np.expand_dims(x[:, 0], 1), x[:, 1:] - 0.97 * x[:, :-1]), axis=1)

  # stft
  x = librosa.stft(x, n_fft=512, window="hann", hop_length=160, win_length=320, center=True)
  x = np.stack((x.real, x.imag), axis=-1)

  # power spectrum
  x = (x**2).sum(-1)

  # mel filter bank
  x = np.matmul(FILTER_BANK, x)

  # log
  x = np.log(x + 1e-20)

  # feature splice
  frames = librosa.util.frame(x, frame_length=3, hop_length=1)
  features = np.concatenate((frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]), axis=1)[:, :, ::3]

  # normalize
  features_mean = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
  features_std = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
  for i in range(features.shape[0]):
    features_mean[i, :] = features[i, :, :].mean(axis=1)
    features_std[i, :] = features[i, :, :].std(axis=1, ddof=1)
  features_std += 1e-5
  features = (features - np.expand_dims(features_mean, 2)) / np.expand_dims(features_std, 2)

  return features.transpose(2, 0, 1)

def iterate(bs=1):
  print(f"there are {len(ci)} samples in the dataset")
  for i in range(0, len(ci), bs):
    features = [feature_extract(os.path.join(BASEDIR, v["files"][0]["fname"])) for v in ci[i : i + bs]]
    transcripts = [v["transcript"] for v in ci[i : i + bs]]
    yield np.concatenate(features, axis=1), np.array(transcripts)

if __name__ == "__main__":
  X, Y = next(iterate())
  print(X.shape, Y.shape)
