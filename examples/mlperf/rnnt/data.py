#%%
import json
import pathlib
import numpy as np
import librosa
import soundfile
from tinygrad import Tensor

"""
The dataset has to be downloaded manually from https://www.openslr.org/12/ and put in `extra/datasets/librispeech`.
For mlperf validation the dev-clean dataset is used.

Then all the flacs have to be converted to wav using something like:
```fish
for file in $(find * | grep flac); do ffmpeg -i $file -ar 16k "$(dirname $file)/$(basename $file .flac).wav"; done
```

Then this [file](https://github.com/mlcommons/inference/blob/master/speech_recognition/rnnt/dev-clean-wav.json) has to also be put in `extra/datasets/librispeech`.
"""

#%%

BASEDIR = pathlib.Path(__file__).parent/"../../../extra/datasets/librispeech/"

def load_data(max_s = 15):
  maxX = maxY = 0
  with open(BASEDIR / "dev-clean-wav.json") as f:
    ci = json.load(f)
    ci = list(filter(lambda x: x['files'][0]['duration'] < max_s,ci)) # filtering for samples under 15s as is allowed in reference
    
    feature_rate = 0.03
    ci_ = []
    for item in ci:
      maxX = max(maxX, round(item['files'][0]['duration'] / feature_rate + 1))
      maxY = max(maxY, len(item["transcript"]))
      ci_.append(item)
    ci = ci_
  return ci, maxX, maxY

FILTER_BANK = np.expand_dims(librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000), 0)
WINDOW = librosa.filters.get_window("hann", 320)

def feature_extract(x, x_lens):
  x_lens = np.ceil((x_lens / 160) / 3).astype(np.int32)

  # pre-emphasis
  x = np.concatenate((np.expand_dims(x[:, 0], 1), x[:, 1:] - 0.97 * x[:, :-1]), axis=1)

  # stft
  x = librosa.stft(x, n_fft=512, window=WINDOW, hop_length=160, win_length=320, center=True, pad_mode="reflect")
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


i2c = [" ","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","'"]
c2i = {c:i for i,c in enumerate(i2c)}

def text_encode(y:np.ndarray,maxlen):
  ylens = np.array([len(t) for t in y])
  y = [np.array([c2i[c] for c in t]) for t in y]
  y = np.array([np.pad(t, (0, maxlen - len(t)), "constant") for t in y])
  return y,ylens

def text_decode(y:np.ndarray):
  return ["".join([i2c[c] for c in t]) for t in y]

def iterate(ci, bs, maxx, maxy):

  for i in range(0, len(ci), bs):
    samples, sample_lens = zip(*[load_wav(BASEDIR / v["files"][0]["fname"]) for v in ci[i : i + bs]])
    samples = list(samples)
    # pad to same length
    max_len = max(sample_lens)
    for j in range(len(samples)):
      samples[j] = np.pad(samples[j], (0, max_len - sample_lens[j]), "constant")
    samples, sample_lens = np.array(samples), np.array(sample_lens)
    files = ci[i:i+bs]
    if (len(files)< bs): return
    
    x,xlens = feature_extract(samples, sample_lens)

    x = np.pad(x, ((0, maxx - x.shape[0]), (0, 0), (0, 0)), "constant")
    y = np.array([v["transcript"] for v in files])
    y,ylens = text_encode(y, maxy)

    yield list(map(lambda x:Tensor(x).realize(), [x,y,xlens,ylens]))
    
#%%
    
ci, maxx, maxy = load_data()
x,y,xlens,ylens = next(iterate(ci, 2,maxx,maxy))

