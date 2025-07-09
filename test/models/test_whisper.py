import unittest
import pathlib
from examples.whisper import init_whisper, load_file_waveform, transcribe_file, transcribe_waveform
from tinygrad.helpers import CI, fetch
from tinygrad import Device, dtypes
from tinygrad.device import is_dtype_supported

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
TRANSCRIPTION_3_SEEK = [{'seek': 0, 'start': 0.0, 'end': 2.44, 'text': ' Just lie back and relax.'}, {'seek': 0, 'start': 2.44, 'end': 5.44, 'text': ' Is the level of pressure about right?'}, {'seek': 0, 'start': 5.44, 'end': 9.44, 'text': " Yes, it's fine, and I'd like conditioner, please."}, {'seek': 0, 'start': 9.44, 'end': 10.84, 'text': ' Sure.'}, {'seek': 0, 'start': 10.84, 'end': 13.84, 'text': " I'm going to start the second lathering now."}, {'seek': 0, 'start': 13.84, 'end': 16.88, 'text': ' Would you like some Q-tips?'}, {'seek': 0, 'start': 16.88, 'end': 18.92, 'text': " How'd you like it cut?"}, {'seek': 0, 'start': 18.92, 'end': 21.72, 'text': " I'd like my bangs in the back trimmed,"}, {'seek': 0, 'start': 21.72, 'end': 24.76, 'text': " and I'd like the rest thinned out a bit and layered."}, {'seek': 0, 'start': 24.76, 'end': 26.72, 'text': ' Where would you like the part?'}, {'seek': 0, 'start': 26.72, 'end': 29.76, 'text': ' On the left, right about here.'}, {'seek': 2976, 'start': 30.76, 'end': 32.08, 'text': ' Here, have a look.'}, {'seek': 2976, 'start': 32.08, 'end': 33.68, 'text': ' What do you think?'}, {'seek': 2976, 'start': 33.68, 'end': 34.68, 'text': " It's fine."}, {'seek': 2976, 'start': 34.68, 'end': 37.04, 'text': " Here's $1,000."}, {'seek': 2976, 'start': 37.04, 'end': 39.760000000000005, 'text': " It's 30-ant extra for the rinse."}, {'seek': 2976, 'start': 39.760000000000005, 'end': 41.72, 'text': " Here's your change and receipt."}, {'seek': 2976, 'start': 41.72, 'end': 44.68, 'text': ' Thank you, and please come again.'}, {'seek': 2976, 'start': 44.68, 'end': 47.400000000000006, 'text': ' So how do you like it?'}, {'seek': 2976, 'start': 47.400000000000006, 'end': 49.44, 'text': " It could have been worse, but you'll"}, {'seek': 2976, 'start': 49.44, 'end': 52.68000000000001, 'text': " notice that I didn't ask her for her card."}, {'seek': 2976, 'start': 52.68000000000001, 'end': 55.68000000000001, 'text': ' Hmm, yeah.'}, {'seek': 2976, 'start': 55.68000000000001, 'end': 59.28, 'text': ' Maybe you can try that place over there next time.'}] # noqa: E501

@unittest.skipIf(CI and Device.DEFAULT in ["CPU"], "slow")
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

  @classmethod
  def reinit_model(cls, batch_size: int):
    if cls.model.batch_size == batch_size: return
    del cls.model
    del cls.enc
    cls.model, cls.enc = init_whisper("tiny.en", batch_size=batch_size)

  def test_transcribe_file1(self):
    self.__class__.reinit_model(2)
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_1),  TRANSCRIPTION_1)

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too many tests for CI")
  def test_transcribe_file2(self):
    self.__class__.reinit_model(2)
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_2),  TRANSCRIPTION_2)

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too many tests for CI")
  def test_transcribe_batch12(self):
    self.__class__.reinit_model(2)
    waveforms = [load_file_waveform(TEST_FILE_1), load_file_waveform(TEST_FILE_2)]
    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[1])

  def test_transcribe_batch21(self):
    self.__class__.reinit_model(2)
    waveforms = [load_file_waveform(TEST_FILE_2), load_file_waveform(TEST_FILE_1)]
    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[1])

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too long for CI")
  def test_transcribe_long(self):
    self.__class__.reinit_model(1)
    waveform = [load_file_waveform(fetch(TEST_FILE_3_URL))]
    transcription = transcribe_waveform(self.model, self.enc, waveform)
    self.assertEqual(TRANSCRIPTION_3, transcription)

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too long for CI")
  def test_transcribe_long_no_batch(self):
    self.__class__.reinit_model(2)
    waveforms = [load_file_waveform(fetch(TEST_FILE_3_URL)), load_file_waveform(TEST_FILE_1)]

    trancriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(trancriptions))
    self.assertEqual(TRANSCRIPTION_3, trancriptions[0])
    self.assertEqual(TRANSCRIPTION_1, trancriptions[1])

  def dicts_are_equal(self, d1, d2, tol=0.05):
    return (d1['seek'] == d2['seek'] and d1['text'] == d2['text'] and abs(d1['start'] - d2['start']) <= tol and abs(d1['end'] - d2['end']) <= tol)

  def assert_transcriptions_equal(self, actual, expected, tol=0.05):
    self.assertEqual(len(actual), len(expected), "Transcription lengths differ.")
    for i, (a, e) in enumerate(zip(actual, expected)):
      self.assertTrue(self.dicts_are_equal(a, e, tol), msg=f"Mismatch at index {i}:\nActual:   {a}\nExpected: {e}")

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too long for CI")
  def test_transcribe_seek(self):
    self.__class__.reinit_model(1)
    waveform = [load_file_waveform(fetch(TEST_FILE_3_URL))]
    transcription = transcribe_waveform(self.model, self.enc, waveform, do_seek=True)
    self.assert_transcriptions_equal(transcription, TRANSCRIPTION_3_SEEK)

if __name__ == '__main__':
  unittest.main()