import unittest
import pathlib
from tinygrad.ops import Device
from examples.whisper import init_whisper, transcribe_file

@unittest.skipUnless(Device.DEFAULT == "METAL", "Some non-metal backends spend too long trying to allocate a 20GB array")
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

  def test_transcribe_file(self):
    # Audio generated with the command on MacOS:
    # say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
    # We use the WAVE type because it's easier to decode in CI test environments
    filename = str(pathlib.Path(__file__).parent / "whisper/test.wav")
    transcription = transcribe_file(self.model, self.enc, filename)
    self.assertEqual("<|startoftranscript|><|notimestamps|> Could you please let me out of the box?<|endoftext|>",  transcription)
