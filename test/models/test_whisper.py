import unittest
import pathlib
from examples.whisper import init_whisper, load_file_waveform, transcribe_file, transcribe_waveform
from tinygrad.helpers import CI
from tinygrad.ops import Device

# Audio generated with the command on MacOS:
# say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
# We use the WAVE type because it's easier to decode in CI test environments
TEST_FILE_1 = str(pathlib.Path(__file__).parent / "whisper/test.wav")
TRANSCRIPTION_1 = "Could you please let me out of the box?"
TEST_FILE_2 = str(pathlib.Path(__file__).parent / "whisper/test2.wav")
TRANSCRIPTION_2 = "a slightly longer audio file so that we can test batch transcriptions of varying length."

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

  def test_transcribe_file1(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_1),  TRANSCRIPTION_1)

  def test_transcribe_file2(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_2),  TRANSCRIPTION_2)

  def test_transcribe_batch12(self):
    waveforms = [load_file_waveform(TEST_FILE_1), load_file_waveform(TEST_FILE_2)]
    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[1])

  def test_transcribe_batch21(self):
    waveforms = [load_file_waveform(TEST_FILE_2), load_file_waveform(TEST_FILE_1)]
    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[1])
