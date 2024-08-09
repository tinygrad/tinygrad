import unittest
import pathlib
from examples.whisper import init_whisper, transcribe_file
from tinygrad.helpers import CI, fetch
from tinygrad import Device, dtypes
from test.helpers import is_dtype_supported

# Audio generated with the command on MacOS:
# say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
# We use the WAVE type because it's easier to decode in CI test environments
TEST_FILE_1 = str(pathlib.Path(__file__).parent / "whisper/test.wav")
TRANSCRIPTION_1 = "Could you please let me out of the box?"
TEST_FILE_2 = str(pathlib.Path(__file__).parent / "whisper/test2.wav")
TRANSCRIPTION_2 = "a slightly longer audio file so that we can test batch transcriptions of varying length."
# TODO this file will possibly not survive long. find another 1-2 minute sound file online to transcribe
TEST_FILE_3_URL = 'https://homepage.ntu.edu.tw/~karchung/miniconversations/mc45.mp3'
TRANSCRIPTION_3 = "Just lie back and relax. Is the level of pressure about right? Yes, it's fine, and I'd like conditioner please. Sure. I'm going to start the second lathering now. Would you like some Q-tips? How'd you like it cut? I'd like my bangs and the back trimmed, and I'd like the rest thinned out a bit and layered. Where would you like the part? On the left, right about here. Here, have a look. What do you think? It's fine. Here's a thousand anti-dollars. It's 30-ant extra for the rants. Here's your change and receipt. Thank you, and please come again. So how do you like it? It could have been worse, but you'll notice that I didn't ask her for her card. Hmm, yeah. Maybe you can try that place over there next time."   # noqa: E501

@unittest.skipIf(CI and Device.DEFAULT in ["CLANG"], "slow")
@unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16 support")
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
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_1),  [TRANSCRIPTION_1])

  @unittest.skipIf(CI, "too many tests for CI")
  def test_transcribe_file2(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_2),  [TRANSCRIPTION_2])

  @unittest.skipIf(CI, "too many tests for CI")
  def test_transcribe_batch12(self):
    transcriptions = transcribe_file(self.model, self.enc, TEST_FILE_1, TEST_FILE_2 )
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[1])

  def test_transcribe_batch21(self):
    transcriptions = transcribe_file(self.model, self.enc, TEST_FILE_2, TEST_FILE_1)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[1])

  @unittest.skipIf(CI, "too long for CI")
  def test_transcribe_long(self):

    transcription = transcribe_file(self.model, self.enc, fetch(TEST_FILE_3_URL))[0]
    print(transcription)
    self.assertEqual(TRANSCRIPTION_3, transcription)

  @unittest.skipIf(CI, "too long for CI")
  def test_transcribe_long_no_batch(self):
    with self.assertRaises(Exception):
      transcribe_file(self.model, self.enc, fetch(TEST_FILE_3_URL), TEST_FILE_1)

if __name__ == '__main__':
  unittest.main()
