import ast
import pathlib
import sys, difflib, string
import unittest

import numpy as np
from tinygrad.tensor import Tensor
from examples.whisper import Whisper, load_model, load_wav, N_SAMPLES, N_FRAMES, pad_or_trim, prep_audio

from extra.datasets.librispeech import ci, BASEDIR
from examples.mlperf.metrics import word_error_rate

def transcribe_wav(model, fn, logprob_threshold=-1.0, no_speech_threshold=0.6):
  mel = prep_audio(load_wav(fn), padding=N_SAMPLES)
  content_frames = mel.shape[-1] - N_FRAMES
  initial_tokens = [model.encoding._special_tokens["<|startoftranscript|>"], model.encoding._special_tokens["<|en|>"],
                    model.encoding._special_tokens["<|transcribe|>"]]
  seek, texts = 0, []
  while seek < content_frames:
    mel_segment = mel[:, seek:seek+N_FRAMES]
    mel_segment = pad_or_trim(mel_segment, N_FRAMES)
    segment_size = min(N_FRAMES, content_frames - seek)
    texts += model.decode_segment(mel_segment, initial_tokens)
    seek += segment_size
  return "".join(texts)

class TestBaseWhisper(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = load_model("base.en")

  @classmethod
  def tearDownClass(cls):
    del cls.model

  def test_transcribe(self):
    diff = difflib.Differ()
    for c in ci:
      fn = BASEDIR / c["files"][0]["fname"]
      print("-" * 128, f"{fn.stem}\n", sep="\n")
      predicted = "".join(transcribe_wav(TestBaseWhisper.model, fn)).translate(str.maketrans("", "", string.punctuation)).lower()
      transcript = c["transcript"].translate(str.maketrans("", "", string.punctuation))
      sys.stdout.writelines(list(diff.compare([predicted + "\n"], [transcript + "\n"])))
      print(f"\nword error rate: {word_error_rate([predicted], [transcript])[0]:.4f}")

if __name__ == '__main__':
  unittest.main()
