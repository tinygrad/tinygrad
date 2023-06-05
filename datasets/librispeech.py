import json
import pathlib
import numpy as np
import librosa
import soundfile
import requests
import hashlib
import tarfile
from tqdm import tqdm

"""
The dataset has to be downloaded manually from https://www.openslr.org/12/ and put in `datasets/librispeech`.
For mlperf validation the dev-clean dataset is used.

Then all the flacs have to be converted to wav using something like:
```fish
for file in $(find * | grep flac); do ffmpeg -i $file -ar 16k "$(dirname $file)/$(basename $file .flac).wav"; done
```

Then this [file](https://github.com/mlcommons/inference/blob/master/speech_recognition/rnnt/dev-clean-wav.json) has to also be put in `datasets/librispeech`.
"""
BASEDIR = pathlib.Path(__file__).parent.parent / "datasets/librispeech"

FILTER_BANK = np.expand_dims(librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000), 0)
WINDOW = librosa.filters.get_window("hann", 320)

DATASET_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
HASH = "42e2234ba48799c1f50f24a7926300a1"

DEV_CLEAN_WAV = "https://raw.githubusercontent.com/mlcommons/inference/master/speech_recognition/rnnt/dev-clean-wav.json"

def dataset_preprocessing():
  BASEDIR.mkdir(parents=True, exist_ok=True)

  # Download and verify dev-clean.tar.gz
  if not (BASEDIR / "dev-clean.tar.gz").exists():
    with requests.get(DATASET_URL, stream=True) as r:
      r.raise_for_status()
      total_size = int(r.headers.get('content-length', 0))

      with open(BASEDIR / "dev-clean.tar.gz", "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, unit_divisor=1024, 
        desc='Downloading dev-clean.tar.gz') as progress_bar:
        for chunk in r.iter_content(chunk_size=8192):
          f.write(chunk)
          progress_bar.update(len(chunk))

    print("Verifying hash...")
    with open(BASEDIR / "dev-clean.tar.gz", "rb") as f:
      assert hashlib.md5(f.read()).hexdigest() == HASH
  else:
    print("File dev-clean.tar.gz already exists. Skipping download.")

  # Extract dev-clean.tar.gz
  if not (BASEDIR / "LibriSpeech").exists():
    with tarfile.open(BASEDIR / "dev-clean.tar.gz", "r:gz") as f, tqdm(
        total=len(f.getmembers()), unit='file', 
        desc='Extracting dev-clean.tar.gz') as progress_bar:
      f.extractall(BASEDIR)
      progress_bar.update(len(f.getmembers()))

    # Move extracted files to librispeech from LibriSpeech
    extracted_path = BASEDIR / "LibriSpeech"
    for file_path in extracted_path.glob("**/*"):
        new_file_path = BASEDIR / file_path.relative_to(extracted_path)
        new_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.rename(new_file_path)
    
    # Remove LibriSpeech folder
    extracted_path.rmdir()
  else:
    print("Folder LibriSpeech already exists. Skipping extraction.")

  # Convert flac to wav
  flac_files = list(BASEDIR.glob("**/*.flac"))
  wav_files = list(BASEDIR.glob("**/*.wav"))
  if not flac_files:
    raise Exception("No flac files found. Did you download and extract the dataset?")
  elif len(wav_files) > 0:
    print("Wav files already exist. Skipping conversion.")
  else:
    wav_folder = BASEDIR / "dev-clean-wav"
    wav_folder.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(flac_files), unit='file', desc='Converting .flac to .wav') as progress_bar:
      for file in flac_files:
        # relative_path = file.relative_to(BASEDIR / "LibriSpeech")
        relative_path = file.relative_to(BASEDIR / "LibriSpeech" / "dev-clean").parent / file.name
        print(f"Relative Path: {relative_path}")
        wav_file = wav_folder / relative_path.with_suffix(".wav")
        print(f"Wav File: {wav_file}")
        soundfile.write(wav_file, soundfile.read(file)[0], 16000)
        progress_bar.update(1)

  # Download dev-clean-wav.json
  with requests.get(DEV_CLEAN_WAV, stream=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get('content-length', 0))

    with open(BASEDIR / "dev-clean-wav.json", "wb") as f, tqdm(
      total=total_size, unit='B', unit_scale=True, unit_divisor=1024, 
      desc='Downloading dev-clean-wav.json') as progress_bar:
      for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
        progress_bar.update(len(chunk))

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

def iterate(bs=1, start=0, val=True):
  if val:
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
  else:
    print(f"there are {len(ci)} samples in the dataset")
    for i in range(start, len(ci), bs):
      samples, sample_lens = zip(*[load_wav(BASEDIR / v["files"][0]["fname"]) for v in ci[i : i + bs]])
      samples = list(samples)
      # pad to same length
      max_len = max(sample_lens)
      for j in range(len(samples)):
        samples[j] = np.pad(samples[j], (0, max_len - sample_lens[j]), "constant")
      samples, sample_lens = np.array(samples), np.array(sample_lens)

      LABELS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
      transcript_labels = [[LABELS.index(c) for c in v["transcript"]] for v in ci[i:i + bs]]
      yield feature_extract(samples, sample_lens), np.array(transcript_labels).astype(np.float32), np.array([v["transcript"] for v in ci[i : i + bs]])

if __name__ == "__main__":
  dataset_preprocessing()
  with open(BASEDIR / "dev-clean-wav.json", encoding="utf-8") as f:
    ci = json.load(f)
  X, Y = next(iterate())
  print(X[0].shape, Y.shape)
