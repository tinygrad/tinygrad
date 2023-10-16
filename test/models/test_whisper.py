import unittest
import pathlib
import librosa
import numpy as np
from tinygrad.ops import Device
from examples.whisper import init_whisper, transcribe_waveform, prep_audio, RATE

@unittest.skipUnless(Device.DEFAULT in ["METAL", "GPU"], "Some non-metal backends spend too long trying to allocate a 20GB array")
class TestWhisper(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    model, enc = init_whisper("tiny.en")
    cls.model = model
    cls.enc = enc

  @classmethod
  def tearDownClass(cls):
    del cls.model
    del cls.enc

  def test_pad_less_thirty_sec(self):
    sec, shp = 20, 3000
    waveform = np.zeros((1, sec*RATE))
    log_spec_len = prep_audio(waveform=waveform).shape[2]
    self.assertEqual(shp, log_spec_len)

  def test_pad_thirty_sec(self):
    sec, shp = 30, 3000
    waveform = np.zeros((1, sec*RATE))
    log_spec_len = prep_audio(waveform=waveform).shape[2]
    self.assertEqual(shp, log_spec_len)

  def test_pad_more_than_thirty_sec(self):
    sec, shp = 61, 9000
    waveform = np.zeros((1, sec*RATE))
    log_spec_len = prep_audio(waveform=waveform).shape[2]
    self.assertEqual(shp, log_spec_len)

  def test_transcribe_file_less_thirty_sec(self):
    # Audio generated with the command on MacOS:
    # say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
    # We use the WAVE type because it's easier to decode in CI test environments
    filename = str(pathlib.Path(__file__).parent / "whisper/test.wav")
    waveform, _ = librosa.load(filename, sr=RATE)
    transcription = transcribe_waveform(waveform, self.model, self.enc)
    self.assertEqual("<|startoftranscript|><|notimestamps|> Could you please let me out of the box?<|endoftext|>",  transcription)

  def test_transcribe_file_more_than_thirty_sec(self):
    # Audio generated with the command on MacOS padded with blanck audio:
    # say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
    # We use the WAVE type because it's easier to decode in CI test environments
    filename = str(pathlib.Path(__file__).parent / "whisper/test.wav")
    waveform, _ = librosa.load(filename, sr=RATE)
    waveform = np.pad(waveform, (0, RATE*30)) # add 30 second black audio
    transcription = transcribe_waveform(waveform, self.model, self.enc)
    print(transcription)
    self.assertEqual("<|startoftranscript|><|notimestamps|> Could you please let me out of the box? [no audio]<|endoftext|>",  transcription)
