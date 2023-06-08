import json
import numpy as np
import librosa
import soundfile
import requests
import hashlib
import tarfile

from pathlib import Path
from tqdm import tqdm
from concurrent import futures

"""
This script downloads, verifies and extracts these datasets from https://www.openslr.org/12/ and converts their flacs to wav:
1. train-clean-100.tar.gz   [6.3G]    (training set of 100 hours "clean" speech)
2. train-clean-360.tar.gz   [23G]     (training set of 360 hours "clean" speech)
3. train-other-500.tar.gz   [30G]     (training set of 500 hours "other" speech)
4. dev-clean.tar.gz         [337M]    (development set, "clean" speech)

Additionally, the script generates the JSON files for the training and validation datasets:
1. train-clean-100-wav.json
2. train-clean-360-wav.json
3. train-other-500-wav.json
4. dev-clean-wav.json


TODO:
1. Add code to generate JSON files

File download url and md5 hash:
http://www.openslr.org/resources/12/train-clean-100.tar.gz
2a93770f6d5c6c964bc36631d331a522

http://www.openslr.org/resources/12/train-clean-360.tar.gz
c0e676e450a7ff2f54aeade5171606fa

http://www.openslr.org/resources/12/train-other-500.tar.gz
d1a0fd59409feb2c614ce4d30c387708

http://www.openslr.org/resources/12/dev-clean.tar.gz
42e2234ba48799c1f50f24a7926300a1
"""

BASEDIR = Path(__file__).parent.parent / "datasets/"

DATASETS = {
  "train-clean-100": {"url": "http://www.openslr.org/resources/12/train-clean-100.tar.gz", "md5": "2a93770f6d5c6c964bc36631d331a522"},
  "train-clean-360": {"url": "http://www.openslr.org/resources/12/train-clean-360.tar.gz", "md5": "c0e676e450a7ff2f54aeade5171606fa"},
  "train-other-500": {"url": "http://www.openslr.org/resources/12/train-other-500.tar.gz", "md5": "d1a0fd59409feb2c614ce4d30c387708"},
  "dev-clean": {"url": "http://www.openslr.org/resources/12/dev-clean.tar.gz", "md5": "42e2234ba48799c1f50f24a7926300a1"}
}

FILTER_BANK = np.expand_dims(librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000), 0)
WINDOW = librosa.filters.get_window("hann", 320)

NUM_THREADS = 8


def extract_file(tar_file, member, destination):
    # Remove the leading "LibriSpeech/" folder from the extracted member path
    member.name = "/".join(member.name.split("/")[1:])
    tar_file.extract(member, destination)
    return member


def dataset_download(dataset, url, hash, num_threads=4):
    BASEDIR.mkdir(parents=True, exist_ok=True)

    dataset_path = BASEDIR / f"{dataset}.tar.gz"

    if dataset_path.exists():
        print("File already exists. Skipping download.")
    else:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))

            with open(dataset_path, "wb") as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                    desc=f"Downloading {dataset}") as progress_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        print("Verifying hash...")
        with open(dataset_path, "rb") as f:
            assert hashlib.md5(f.read()).hexdigest() == hash

    if not (BASEDIR / "LibriSpeech").exists():
        with tarfile.open(dataset_path, "r:gz") as tar, \
                futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

            members = tar.getmembers()
            progress_bar = tqdm(total=len(members), unit='file', desc=f"Extracting {dataset}")

            futures_list = [executor.submit(extract_file, tar, member, BASEDIR / "LibriSpeech" / dataset)
                            for member in members]

            for _ in futures.as_completed(futures_list):
                progress_bar.update(1)

            progress_bar.close()
    else:
        print("Directory already exists. Skipping extraction.")


def dataset_convert(dataset):
  # Convert flac to wav
  flac_files = list((BASEDIR / "LibriSpeech" / dataset).glob("**/*.flac"))
  wav_files = list((BASEDIR / "LibriSpeech" / dataset).glob("**/*.wav"))
  if not flac_files:
    raise Exception("No flac files found. Did you download and extract the dataset?")
  elif len(wav_files) > 0:
    print("Wav files already exist. Skipping conversion.")
  else:
    wav_folder = BASEDIR / "LibriSpeech" / f"{dataset}-wav"
    wav_folder.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(flac_files), unit='file', desc='Converting .flac to .wav') as progress_bar:
      for file in flac_files:
        relative_path = file.relative_to(BASEDIR / "LibriSpeech" / dataset).parent / file.name
        (wav_folder / relative_path).mkdir(parents=True, exist_ok=True)
        print("relative:", relative_path)
        print("wav:", wav_folder / relative_path)
        wav_file = (wav_folder / relative_path).with_suffix(".wav")
        soundfile.write(wav_file, soundfile.read(file)[0], 16000)
        progress_bar.update(1)


def generate_json(dataset):
    wav_files = list(BASEDIR.glob("**/*.wav"))
    if not wav_files:
        raise Exception("No wav files found. Did you convert the dataset to .wav?")
    else:
        json_data = []
        with tqdm(total=len(wav_files), unit='file', desc='Generating JSON files') as progress_bar:
            for file in wav_files:
                relative_path = file.relative_to(BASEDIR / "LibriSpeech" / dataset).parent / file.name
                json_data.append({
                    "wav_path": str(file),
                    "relative_path": str(relative_path),
                    "transcript": ""
                })
                progress_bar.update(1)

        json_output_file = BASEDIR / "LibriSpeech" / f"{dataset}.json"
        with open(json_output_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"JSON file generated: {json_output_file}")


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
    print(f"Number of samples in the dataset: {len(ci)}")
    for i in range(start, len(ci), bs):
      samples, sample_lens = zip(*[load_wav(BASEDIR / "LibriSpeech" / v["files"][0]["fname"]) for v in ci[i : i + bs]])
      samples = list(samples)
      # pad to same length
      max_len = max(sample_lens)
      for j in range(len(samples)):
        samples[j] = np.pad(samples[j], (0, max_len - sample_lens[j]), "constant")
      samples, sample_lens = np.array(samples), np.array(sample_lens)

      yield feature_extract(samples, sample_lens), np.array([v["transcript"] for v in ci[i : i + bs]])
  else:
    print(f"Number of samples in the dataset: {len(ci)}")
    for i in range(start, len(ci), bs):
      samples, sample_lens = zip(*[load_wav(BASEDIR / "LibriSpeech" / v["files"][0]["fname"]) for v in ci[i : i + bs]])
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
  for dataset, info in DATASETS.items():
    # dataset_download(dataset, info["url"], info["md5"], NUM_THREADS)
    dataset_convert(dataset)
    generate_json(dataset)


  # with open(BASEDIR / "LibriSpeech" / "dev-clean-wav.json", encoding="utf-8") as f:
  #   ci = json.load(f)
  # X, Y = next(iterate())
  # print(f"Shape of X: {X[0].shape} ; Shape of Y: {Y.shape}")
