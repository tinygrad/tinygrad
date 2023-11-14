from contextlib import contextmanager
from typing import Tuple

import pyaudio

@contextmanager
def audio_stream(is_input: bool, sample_rate: int, n_frame_chunk: int):
  try:
    p = pyaudio.PyAudio()
    stream = p.open(rate=sample_rate, channels=1, format=pyaudio.paInt16, input=is_input, output=(not is_input), frames_per_buffer=n_frame_chunk)
    yield stream
  finally:
    stream.close()
    p.terminate()
