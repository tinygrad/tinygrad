import argparse
import socket
import numpy as np 
from whisper import prep_audio, init_whisper, listener
import multiprocessing
from os import getenv
from tinygrad.tensor import Tensor

RATE = 16000
CHUNK = 1600
RECORD_SECONDS = 5

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", type=str, default="localhost")
  parser.add_argument("--port", type=int, default=5000)
  args = parser.parse_args()
  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en")
  q = multiprocessing.Queue()
  p = multiprocessing.Process(target=listener, args=(q,))
  p.daemon = True
  p.start()

  lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  total = None
  did_read = False
  p = 0

  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect((args.host, args.port))
  prev = ""
  while True:
    while not q.empty() or total is None:
      waveform = q.get()
      if total is None: total = waveform
      else: total = np.concatenate([total, waveform])
      did_read = True
    if did_read:
      log_spec = prep_audio(total)
      encoded_audio = model.encoder(Tensor(log_spec)).realize()
    out = model.decoder(Tensor([lst]), encoded_audio).realize()
    lst.append(int(out[0,-1].argmax().numpy().item()))
    dec = enc.decode(lst)
    if dec.endswith("<|endoftext|>"):
      user_input = dec.split("[BLANK_AUDIO]")[1:-1]
      print(user_input)
      if len(user_input) > 0 and user_input[-1] != prev:
        print(user_input[-1])
        s.send(user_input[-1].encode()) 
        print(s.recv(1024).decode())
        prev = user_input[-1]
      lst.pop()
