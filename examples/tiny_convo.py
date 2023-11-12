from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from examples.whisper import init_whisper, prep_audio, Whisper
from tqdm import tqdm

import multiprocessing
import numpy as np
import pyaudio
import tiktoken
import os

SAMPLE_RATE = 16000
N_FRAME_CHUNK = 1600
RECORD_SECONDS = 10
NUM_RUNS = int(SAMPLE_RATE / N_FRAME_CHUNK * RECORD_SECONDS)

# TODO: move this somewhere because process cannot pickle this

def listen(q: multiprocessing.Queue):
  p = pyaudio.PyAudio()
  stream = p.open(
      format=pyaudio.paInt16,
      channels=1,
      rate=SAMPLE_RATE,
      input=True,
      frames_per_buffer=N_FRAME_CHUNK
  )
  print("Start listening...")

  for _ in range(0, NUM_RUNS):
    au_data = stream.read(N_FRAME_CHUNK)
    au = ((np.frombuffer(au_data, np.int16)/32768).astype(np.float32)*3)
    q.put(au)

  print("Done listening!")

  stream.close()
  p.terminate()

if __name__ == "__main__":
  q = multiprocessing.Queue()

  p = multiprocessing.Process(target=listen, args=(q,))
  p.daemon = True

  p.start()

  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en")
  lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  au_buffer = None

  # listen
  while True:
    au = q.get()

    if au_buffer is None:
      au_buffer = au
    else:
      au_buffer = np.concatenate([au_buffer, au])

    if au_buffer.shape[0] == N_FRAME_CHUNK * NUM_RUNS:
      break

  # decode what user is saying
  for _ in range(NUM_RUNS):
    log_spec = prep_audio(au_buffer)
    encoded_audio = model.encoder(Tensor(log_spec)).realize()

    out = model.decoder(Tensor([lst]), encoded_audio).realize()
    idx = int(out[0,-1].argmax().numpy().item())
    lst.append(idx)
    dec = enc.decode(lst)

    print(dec)

    if dec.endswith("<|endoftext|>"):
      lst.pop()
      break

  # TODO: remove special tokens
  # ask LLaMA

  # use VITS for response
