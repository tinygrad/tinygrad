from tinygrad.helpers import getenv, dtypes
from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import Variable
from examples.whisper import init_whisper, prep_audio
from pathlib import Path
from examples.llama import LLaMa
from examples.vits import MODELS, download_if_not_present, get_hparams_from_file, TextMapper, VITS_PATH, load_model
from examples.tinyconvo.helpers import audio_stream

import argparse
import multiprocessing
import numpy as np
import sys

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
N_LLAMA_COUNT = 100
MAX_CONTEXT = 1024
TEMP = 0.7

# VITS
NOISE_SCALE = 0.667
LENGTH_SCALE = 1
NOISE_SCALE_W = 0.8

def listen(q: multiprocessing.Queue, n_frame_chunk: int):
  with audio_stream(True, SAMPLE_RATE, n_frame_chunk) as stream:
    print("Start listening...")

    for _ in range(0, NUM_RUNS):
      au_data = stream.read(n_frame_chunk)
      au = ((np.frombuffer(au_data, np.int16)/32768).astype(np.float32)*3)
      q.put(au)

    print("Done listening!")

def whisper_decode(au_buffer: np.ndarray) -> str:
  model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en")
  lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  idx_2_spec_toks = {v: k for k, v in enc._special_tokens.items()}
  output_history = ""

  log_spec = prep_audio(au_buffer)
  encoded_audio = model.encoder(Tensor(log_spec)).realize()

  for _ in range(NUM_RUNS):
    out = model.decoder(Tensor([lst]), encoded_audio).realize()
    idx = int(out[0,-1].argmax().numpy().item())
    lst.append(idx)

    # decode a version with no special tokens as input to LLaMA
    dec = enc.decode([token for token in lst if token not in idx_2_spec_toks])

    sys.stdout.write(dec[len(output_history):])
    sys.stdout.flush()

    output_history = dec

    if lst[-1] == enc._special_tokens["<|endoftext|>"]:
      return dec

# TODO: refactor LLaMA example to expose API to perform decoding
def llama_response(prompt: str) -> str:
  outputted = prompt
  llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_gen="1", model_size="7B", quantize=False)
  llama_tokens = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(prompt)
  start_pos = 0

  llama_response = ""

  print("Getting response...")
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

  print("Here's the response:")

  return llama_response

# TODO: refactor VITS example to expose API to perform TTS
def tts(text_to_synthesize: str) -> np.ndarray:
  model_config = MODELS["mmts-tts"]
  config_path = model_config[0]
  download_if_not_present(config_path, model_config[2])
  hps = get_hparams_from_file(config_path)

  symbols = [x.replace("\n", "") for x in open(download_if_not_present(VITS_PATH / "vocab_mmts-tts.txt", "https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/vocab.txt"), encoding="utf-8").readlines()]
  text_mapper = TextMapper(apply_cleaners=True, symbols=symbols)
  net_g = load_model(text_mapper.symbols, hps, model_config)

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

  q = multiprocessing.Queue()

  p = multiprocessing.Process(target=listen, args=(q, args.n_frame_chunk))
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

    if au_buffer.shape[0] == args.n_frame_chunk * NUM_RUNS:
      break

  # decode what user is saying
  user_prompt = whisper_decode(au_buffer)

  # ask LLaMA
  response = llama_response(user_prompt)

  # use VITS to read response
  audio_response = tts(response)

  print("Reading the response...")
  play_audio(audio_response)
  print("Done!")
