import os
import json
import pathlib
import numpy as np
import librosa
import soundfile as sf
import tarfile
from extra.utils import download_file
from typing import List

"""
The dataset has to be downloaded manually from https://www.openslr.org/12/ and put in `datasets/librispeech`.
For mlperf validation the dev-clean dataset is used.

Then all the flacs have to be converted to wav using something like:
```fish
for file in $(find * | grep flac); do ffmpeg -i $file -ar 16k "$(dirname $file)/$(basename $file .flac).wav"; done
```

Then this [file](https://github.com/mlcommons/inference/blob/master/speech_recognition/rnnt/dev-clean-wav.json) has to also be put in `datasets/librispeech`.
"""
BASEDIR = pathlib.Path(__file__).parent.parent / "datasets"

# with open(BASEDIR / "dev-clean-wav.json") as f:
#   ci = json.load(f)


def get_train_files() -> list[dict]:
    fp = BASEDIR / "librispeech" / "train"/ "train-clean-100.tar.gz"
    fp_target = BASEDIR / "librispeech" / "train"
    file_name = "train_clean_100.json"
    if not os.path.exists(fp):
        download_file(
            "https://www.openslr.org/resources/12/train-clean-100.tar.gz", fp
        )
    if not os.path.exists(fp_target/"LibriSpeech"):
        tarfile_ = tarfile.open(fp, "r")
        tarfile_.extractall(fp_target)
        tarfile_.close()
    # fp.unlink()
    if not os.path.exists(fp_target/"LibriSpeech"/f"{file_name}"):
        generate_transcripts(
            fp_target, file_name
        )  # TODO: check if the transcripts already exists, and if yes skip this step.
    return load_transcripts(fp_target / file_name)


def get_validation_files() -> List[dict]:
    fp = BASEDIR / "librispeech" / "val"/ "dev-clean.tar.gz"
    fp_target = BASEDIR / "librispeech" / "val"
    print("fp_target", fp_target)
    file_name = "dev_clean.json"
    if not os.path.exists(fp):
        download_file("https://www.openslr.org/resources/12/dev-clean.tar.gz", fp)
    if not os.path.exists(fp_target/"LibriSpeech"):
        tarfile_ = tarfile.open(fp, "r")
        tarfile_.extractall(fp_target)
        tarfile_.close()
    # fp.unlink()
    if not os.path.exists(fp_target/"LibriSpeech"/f"{file_name}"):
        generate_transcripts(
            fp_target, file_name
        )  # TODO: check if the transcripts already exists, and if yes skip this step.
    return load_transcripts(fp_target / "LibriSpeech" / file_name)


def generate_transcripts(file_path: str, file_name: str):
    transcripts: list = []
    transcript_file_paths = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(file_path/"LibriSpeech"/"dev-clean", topdown=True)
        for name in files
        if name.endswith(".trans.txt")
    ]
    for t in transcript_file_paths:
        with open(t, "r") as transcript_file:
            for line in transcript_file:
                if line:
                    # decoded_line = line.encode("utf-8").strip()
                    file_id, transcript = line.strip().split(" ", 1)
                    speaker_id, chapter_id = [int(el) for el in file_id.split("-")[:2]]
                    transcripts.append(
                        {
                            "id": file_id,
                            "file_path": os.path.join(file_path,f"LibriSpeech/dev-clean/{speaker_id}/{chapter_id}/{file_id}.flac"),
                            "transcript": transcript,
                        }
                    )
    final_transcript_file = file_path / "LibriSpeech" / file_name
    with open(final_transcript_file, "w") as f:
        json.dump(transcripts, f)


def load_transcripts(file_path: str) -> List[dict]:
    with open(file_path, "r") as f_:
        return json.load(f_)
        f_.close()


FILTER_BANK = np.expand_dims(
    librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000), 0
)
WINDOW = librosa.filters.get_window("hann", 320)


def feature_extract(x : np.ndarray[np.float32] , x_lens : np.ndarray[np.float32]):
    x_lens = np.ceil((x_lens / 160) / 3).astype(np.int32)

    # pre-emphasis
    x = np.concatenate(
        (np.expand_dims(x[:, 0], 1), x[:, 1:] - 0.97 * x[:, :-1]), axis=1
    )

    # stft
    x = librosa.stft(
        x,
        n_fft=512,
        window=WINDOW,
        hop_length=160,
        win_length=320,
        center=True,
        pad_mode="reflect",
    )
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
        features_mean[i, :] = features[i, :, : x_lens[i]].mean(axis=1)
        features_std[i, :] = features[i, :, : x_lens[i]].std(axis=1, ddof=1)
    features_std += 1e-5
    features = (features - np.expand_dims(features_mean, 2)) / np.expand_dims(
        features_std, 2
    )

    return features.transpose(2, 0, 1), x_lens.astype(np.float32)


def load_audio_file(file):
    sample = sf.read(file)[0].astype(np.float32)
    return sample, sample.shape[0]


def iterate(bs=1, start=0, mode="val"):
    assert mode in ["train", "val"], "mode must be either train or val."
    transcripts = get_train_files() if mode == "train" else get_validation_files()
    print(f"there are {len(transcripts)} samples in the dataset")
    for i in range(start, len(transcripts), bs):
        samples, sample_lens = zip(
            *[
                load_audio_file(BASEDIR / v["file_path"])
                for v in transcripts[i : i + bs]
            ]
        )
        samples = list(samples)
        # pad to same length
        max_len = max(sample_lens)
        for j in range(len(samples)):
            samples[j] = np.pad(samples[j], (0, max_len - sample_lens[j]), "constant")
        samples, sample_lens = np.array(samples), np.array(sample_lens)

        yield feature_extract(samples, sample_lens), np.array(
            [v["transcript"] for v in transcripts[i : i + bs]]
        )


if __name__ == "__main__":
    result = next(iterate(mode="val"))
    print(result)
