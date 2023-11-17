from tinygrad.helpers import getenv, dtypes
from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import Variable
from examples.whisper import init_whisper, prep_audio
from pathlib import Path
from examples.llama import LLaMa
from examples.vits import MODELS, download_if_not_present, get_hparams_from_file, TextMapper, VITS_PATH, load_model
from examples.tinyconvo.helpers import audio_stream
from typing import Tuple

import argparse
import multiprocessing
import numpy as np
import re
import sys
import time

# Whisper
SAMPLE_RATE = 16000
N_FRAME_CHUNK = 1600
RECORD_SECONDS = 3 # TODO: remove this
NUM_RUNS = int(SAMPLE_RATE / N_FRAME_CHUNK * RECORD_SECONDS)

# LLaMA
# TODO: reuse it from examples/llama.py file
LLAMA_SUFFIX = {"1": "", "2": "-2", "code": "-code"}["2"]
MODEL_PATH = Path(__file__).parents[1] / f"weights/LLaMA{LLAMA_SUFFIX}/7B"
TOKENIZER_PATH = (MODEL_PATH if MODEL_PATH.is_dir() else MODEL_PATH.parent) / "tokenizer.model"
N_LLAMA_COUNT = 30
MAX_CONTEXT = 1024
TEMP = 0.7

# VITS
NOISE_SCALE = 0.667
LENGTH_SCALE = 1
NOISE_SCALE_W = 0.8

def listen(q: multiprocessing.Queue, n_frame_chunk: int):
  with audio_stream(True, SAMPLE_RATE, n_frame_chunk) as stream:
    print("Start listening...")

    while True:
      au_data = stream.read(n_frame_chunk)
      au = ((np.frombuffer(au_data, np.int16)/32768).astype(np.float32)*3)
      q.put(au)

def whisper_decode(encoded_audio: Tensor, model, enc, lst) -> Tuple[str, int]:
  out = model.decoder(Tensor([lst]), 0, encoded_audio, streaming=True).realize()
  idx = int(out[0,-1].argmax().numpy().item())
  lst.append(idx)
  dec = enc.decode([token for token in lst if token not in {v: k for k, v in enc._special_tokens.items()}])
  return dec, lst[-1]

# TODO: refactor LLaMA example to expose API to perform decoding
def llama_response(llama, prompt: str) -> str:
  outputted = prompt
  llama_tokens = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(prompt)
  start_pos = 0

  llama_response = ""

  for i in range(N_LLAMA_COUNT):
    probs = llama.model(Tensor([llama_tokens[start_pos:]]), start_pos, TEMP).realize()
    probs_np = probs.numpy()
    tok = int(np.random.choice(len(probs_np), p=probs_np))

    start_pos = len(llama_tokens)
    llama_tokens.append(tok)

    cur = llama.tokenizer.decode(llama_tokens)
    llama_response += cur[len(outputted):]

    sys.stdout.write(cur[len(outputted):])
    sys.stdout.flush()
    outputted = cur

  print("\n")
  return llama_response

# TODO: refactor VITS example to expose API to perform TTS
def tts(text_to_synthesize: str, net_g, text_mapper) -> np.ndarray:
  text_to_synthesize = text_mapper.filter_oov(text_to_synthesize.lower())
  stn_tst = text_mapper.get_text(text_to_synthesize, hps.data.add_blank, hps.data.text_cleaners)
  x_tst, x_tst_lengths = stn_tst.unsqueeze(0), Tensor([stn_tst.shape[0]], dtype=dtypes.int64)
  sid = None

  audio_tensor = net_g.infer(x_tst, x_tst_lengths, sid, NOISE_SCALE, LENGTH_SCALE, NOISE_SCALE_W, emotion_embedding=None,
                             max_y_length_estimate_scale=None)[0, 0].realize()
  audio_data = (np.clip(audio_tensor.numpy(), -1.0, 1.0) * 32767).astype(np.int16)

  return audio_data

def play_audio(audio: np.ndarray):
  with audio_stream(False, SAMPLE_RATE, 0) as stream:
    stream.write(audio.tobytes())


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(prog="tiny convo", description="Run a conversation with tinygrad using Whisper + LLaMA + VITS", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  arg_parser.add_argument("--n_frame_chunk", type=int, default=N_FRAME_CHUNK, help="the number of chunks to read when speaking")

  args = arg_parser.parse_args()

  Tensor.no_grad = True

  # whisper
  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en")
  lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]

  # llama
  llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_gen="1", model_size="7B", quantize=False)

  # VITS
  model_config = MODELS["mmts-tts"]
  config_path = model_config[0]
  download_if_not_present(config_path, model_config[2])
  hps = get_hparams_from_file(config_path)

  symbols = [x.replace("\n", "") for x in open(download_if_not_present(VITS_PATH / "vocab_mmts-tts.txt", "https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/vocab.txt"), encoding="utf-8").readlines()]
  text_mapper = TextMapper(apply_cleaners=True, symbols=symbols)
  net_g = load_model(text_mapper.symbols, hps, model_config)

  # start listening process
  q = multiprocessing.Queue()

  p = multiprocessing.Process(target=listen, args=(q, args.n_frame_chunk))
  p.daemon = True

  p.start()

  au_buffer = None
  prompt_buffer = ""

  # listen
  try:
    while True:
      while not q.empty():
        au = q.get()

        if au_buffer is None:
          au_buffer = au
        else:
          au_buffer = np.concatenate([au_buffer, au])

      if au_buffer is not None:
        log_spec = prep_audio(au_buffer.reshape(1, -1), batch_size=1)
        encoded_audio = model.encoder(Tensor(log_spec)).realize()
        prompt, last_tok = whisper_decode(encoded_audio, model, enc, lst)

        sys.stdout.write(prompt[len(prompt_buffer):])
        sys.stdout.flush()

        prompt_buffer = prompt

        if last_tok == enc._special_tokens["<|endoftext|>"]:
          print("\nDone listening")
          print("Getting response...")

          prompt = re.sub(re.compile(r'\\[.*?\\]'), "", prompt)
          prompt = re.sub(re.compile(r'\\(.*?\\)'), "", prompt)
          prompt = re.sub(re.compile(r"[^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-]"), " ", prompt)
          response = llama_response(llama, prompt)
          
          print("Generating audio response...")
          audio_response = tts(response, net_g, text_mapper)

          print("Playing audio response")
          play_audio(audio_response)

          lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
          au_buffer = None
          prompt_buffer = ""

  except KeyboardInterrupt:
    sys.exit(0)
