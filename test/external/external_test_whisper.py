import ast
import pathlib
import sys, difflib, string
import unittest

import numpy as np
import librosa
from tinygrad.tensor import Tensor
from examples.whisper import Whisper, load_model, transcribe

from extra.datasets.librispeech import ci, BASEDIR
from examples.mlperf.metrics import word_error_rate

class TestWhisperValidationSet(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model = load_model()

  @classmethod
  def tearDownClass(cls):
    del cls.model

  def test_transcribe(self):
    diff = difflib.Differ()
    for c in ci:
      fn = BASEDIR / c["files"][0]["fname"]
      print("-" * 128, f"{fn.stem}\n", sep="\n")
      waveform, _ = librosa.load(fn, mono=True, sr=None)
      predicted = "".join(transcribe(TestWhisperValidationSet.model, waveform)).translate(str.maketrans("", "", string.punctuation)).lower()
      transcript = c["transcript"].translate(str.maketrans("", "", string.punctuation))
      sys.stdout.writelines(list(diff.compare([predicted + "\n"], [transcript + "\n"])))
      print(f"\nword error rate: {word_error_rate([predicted], [transcript])[0]:.4f}")

if __name__ == '__main__':
  unittest.main()
