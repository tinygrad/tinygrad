import unittest
import pathlib
from examples.whisper import init_whisper, load_file_waveform, transcribe_file, transcribe_waveform
from tinygrad.helpers import CI
from tinygrad.ops import Device

@unittest.skipIf(CI and Device.DEFAULT in ["LLVM", "CLANG", "CPU"], "Not working on LLVM, slow on others")
class TestWhisper(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    model, enc = init_whisper("tiny.en", batch_size=2)
    cls.model = model
    cls.enc = enc

  @classmethod
  def tearDownClass(cls):
    del cls.model
    del cls.enc

  def test_transcribe_file(self):
    # Audio generated with the command on MacOS:
    # say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
    # We use the WAVE type because it's easier to decode in CI test environments
    filename = str(pathlib.Path(__file__).parent / "whisper/test.wav")
    transcription = transcribe_file(self.model, self.enc, filename)
    self.assertEqual("Could you please let me out of the box?",  transcription)

  def test_transcribe_batch(self):
    file1 = str(pathlib.Path(__file__).parent / "whisper/test.wav")
    file2 = str(pathlib.Path(__file__).parent / "whisper/test2.wav")

    waveforms = [load_file_waveform(file1), load_file_waveform(file2)]

    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual("Could you please let me out of the box?",  transcriptions[0])
    self.assertEqual("a slightly longer audio file so that we can test batch transcriptions of varying length.",  transcriptions[1])
