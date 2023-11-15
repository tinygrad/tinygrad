import argparse
import pyaudio
import numpy as np
import multiprocessing as mp
from whisper import Whisper, prep_audio, init_whisper
from tiktoken import Encoding
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv

# Whisper constants
RATE = 16000
CHUNK = 10000

def clean_text(enc: Encoding, txt: str) -> str:
  for t in enc.special_tokens_set:
    txt = txt.replace(t, "")
  return txt.replace("[BLANK_AUDIO]", "").strip()

def voice2text(model: Whisper, enc: Encoding, waveform: bytes, tokens: list[int]):
  encoded_audio = model.encoder(Tensor(prep_audio(waveform))).realize()
  out = model.decoder(Tensor([tokens]), encoded_audio).realize()
  tokens.append(int(out[0,-1].argmax().numpy().item()))
  return enc.decode(tokens)

def llama_generate():
  pass

def text2voice():
  pass

def listener(q: mp.Queue):
    try:
      p = pyaudio.PyAudio()
      stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
      print("Listening")
      while True:
        q.put(((np.frombuffer(stream.read(CHUNK), np.int16)/32768).astype(np.float32)*3))
    except Exception as e:
      print(f"Error in audio_listener: {e}")
      q.put(None)
    finally:
      stream.stop_stream()
      stream.close()
      p.terminate()

if __name__ == "__main__":
  # Parse CLI arguments
  parser = argparse.ArgumentParser("Have a tiny conversation with tinygrad")

  # Whisper args
  parser.add_argument("--whisper_model_name", type=str, default="tiny.en")

  arguments = parser.parse_args()

  # Init whisper
  model, enc = init_whisper(arguments.whisper_model_name)

  # Start child process for mic input
  q = mp.Queue()
  p = mp.Process(target=listener, args=(q,))
  p.daemon = True
  p.start()

  tokens = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  total = np.array([])
  prev_length = 0
  while (data := q.get()) is not None:
    total = np.concatenate([total, data])
    txt = voice2text(model, enc, total, tokens)
    print(txt)
    if not txt.endswith("<|endoftext|>"): continue # Didn't reach the end of users' speech
    txt = clean_text(enc, txt)
    # TODO: generate response from llama

    # TODO: convert llama output to voice

    prev_length = len(txt)
    tokens.pop()
