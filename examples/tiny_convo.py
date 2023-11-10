from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from examples.whisper import init_whisper, prep_audio, Whisper

import multiprocessing
import numpy as np
import pyaudio
import tiktoken

SAMPLE_RATE = 16000
N_FRAME_CHUNK = 1600

class AudioListener:
  def __init__(self, sample_rate: int = SAMPLE_RATE, n_frame_chunk: int = N_FRAME_CHUNK):
    self.sample_rate = sample_rate
    self.n_frame_chunk = n_frame_chunk

  def start(self, audio_queue: multiprocessing.Queue):
    self.process = multiprocessing.Process(target=self.listen, args=(audio_queue,))
    self.process.daemon = True
    self.process.start()

  def listen(self, audio_queue: multiprocessing.Queue):
    paudio = pyaudio.PyAudio()
    stream = paudio.open(
      format=pyaudio.paInt16,
      channels=1,
      rate=self.sample_rate,
      input=True,
      frames_per_buffer=self.n_frame_chunk
    )

    while True:
      # TODO: make audio settings as ArgumentParser arg
      audio = stream.read(self.n_frame_chunk)
      audio = ((np.frombuffer(audio, np.int16)/32768).astype(np.float32)*3)

      audio_queue.put(audio)


if __name__ == "__main__":
  # TODO: either refactor whisper example or put it on some helper class
  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en")

  # TODO: make this take arguments later
  audio_listener = AudioListener()
  audio_queue = multiprocessing.Queue()

  audio_listener.start(audio_queue)

  lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  audio_buffer = None
  did_read = False

  while True:
    while not audio_queue.empty():
      audio = audio_queue.get()
      if audio_buffer is None:
        audio_buffer = audio
      else:
        audio_buffer = np.concatenate([audio_buffer, audio])

    if audio_buffer is not None:
      log_spec = prep_audio(audio_buffer)
      encoded_audio = model.encoder(Tensor(log_spec)).realize()

      out = model.decoder(Tensor([lst]), encoded_audio).realize()
      idx = int(out[0,-1].argmax().numpy().item())
      lst.append(idx)
      dec = enc.decode(lst)

      print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
      if dec.endswith("<|endoftext|>"):
        lst.pop()
