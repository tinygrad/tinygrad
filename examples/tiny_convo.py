from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import Variable
from examples.whisper import init_whisper, prep_audio
from pathlib import Path
from examples.llama import LLaMa

import multiprocessing
import numpy as np
import pyaudio
import sys

# Whisper
SAMPLE_RATE = 16000
N_FRAME_CHUNK = 1600
RECORD_SECONDS = 3
NUM_RUNS = int(SAMPLE_RATE / N_FRAME_CHUNK * RECORD_SECONDS)

# LLaMA
# TODO: reuse it from examples/llama.py file
LLAMA_SUFFIX = {"1": "", "2": "-2", "code": "-code"}["1"]
MODEL_PATH = Path(__file__).parents[1] / f"weights/LLaMA{LLAMA_SUFFIX}/7B"
TOKENIZER_PATH = (MODEL_PATH if MODEL_PATH.is_dir() else MODEL_PATH.parent) / "tokenizer.model"
N_LLAMA_COUNT = 1000
MAX_CONTEXT = 1024
TEMP = 0.7

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

def whisper_decode(au_buffer: np.ndarray) -> str:
  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en")
  lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  idx_2_spec_toks = {v: k for k, v in enc._special_tokens.items()}
  output_history = ""

  for _ in range(NUM_RUNS):
    log_spec = prep_audio(au_buffer)
    encoded_audio = model.encoder(Tensor(log_spec)).realize()

    out = model.decoder(Tensor([lst]), encoded_audio).realize()
    idx = int(out[0,-1].argmax().numpy().item())
    lst.append(idx)
    
    lst_no_special_tokens = [token for token in lst if token not in idx_2_spec_toks]

    unmod_dec = enc.decode(lst)
    dec = enc.decode(lst_no_special_tokens)

    sys.stdout.write(dec[len(output_history):])
    sys.stdout.flush()

    output_history = dec

    if unmod_dec.endswith("<|endoftext|>"):
      break

  return dec


if __name__ == "__main__":
  Tensor.no_grad = True

  q = multiprocessing.Queue()

  p = multiprocessing.Process(target=listen, args=(q,))
  p.daemon = True

  p.start()

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
  user_prompt = whisper_decode(au_buffer)

  # ask LLaMA
  outputted = user_prompt
  llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_gen="1", model_size="7B", quantize=False)
  llama_tokens = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(user_prompt)
  start_pos = 0

  for i in range(N_LLAMA_COUNT):
    probs = llama.model(Tensor([llama_tokens[start_pos:]]), Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT).bind(start_pos), TEMP).realize()
    probs_np = probs.numpy()
    tok = int(np.random.choice(len(probs_np), p=probs_np))

    start_pos = len(llama_tokens)
    llama_tokens.append(tok)

    cur = llama.tokenizer.decode(llama_tokens)
    sys.stdout.write(cur[len(outputted):])
    sys.stdout.flush()
    outputted = cur

  # use VITS for response
