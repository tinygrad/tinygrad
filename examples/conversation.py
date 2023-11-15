import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pyaudio
from llama import LLaMa
from tiktoken import Encoding
from whisper import Whisper, init_whisper, prep_audio

from tinygrad.tensor import Tensor

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

def llama_prepare(llama: LLaMa) -> list[int]:
  pre_prompt = f"""Consider that the following is conversation between an AI assistant named Stacy and User
You are Stacy!
You have been a rapper your whole life who struggled with bipolar disorder. You called yourself lil stacy.
You love to answer questions and you are very good at it. Sometimes you answer in rap form.
You are verbose, honest, and accurate when you answer questions, but sometimes your mental illness manifests.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
  examples = {
    "What is your name?": "Hi! My name is Stacy. I'm a rapper with bipolar disorder.",
    "french revolution was what year?": "The French Revolution started in 1789, and lasted 10 years until 1799.",
    "What is bigger, the moon or the sun?": "The sun is bigger than the moon, except when Mercury is in retrograde.",
  }

  user_delim = "\nUser: "
  resp_delim = "Stacy: "
  end_delim = " [EOS]\n"
  pre_prompt += ''.join(f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k,v in examples.items())

  toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(pre_prompt)
  llama.model(Tensor([toks]), 0, arguments.llama_temperature).realize()  # NOTE: outputs are not used
  return toks, user_delim, resp_delim, end_delim


def llama_generate(
  llama: LLaMa,
  prompt: str,
  start_pos: int,
  outputted: str,
  count=30,
  temperature=0.7,
  user_delim="\nUser: ",
  end_delim=" [EOS]",
):
  # Add tokens from user
  outputted += f"{user_delim}{prompt}\n"
  toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(outputted)

  for _ in range(count):
    probs_np = llama.model(Tensor([toks[start_pos:]]), start_pos, temperature).realize().numpy()
    token = int(np.random.choice(len(probs_np), p=probs_np))
    start_pos = len(toks)
    toks.append(token)

    cur = llama.tokenizer.decode(toks)

    # Print is just for debugging
    sys.stdout.write(cur[len(outputted):])
    sys.stdout.flush()
    outputted = cur

    # stop after you have your answer
    if outputted.endswith(end_delim): break
  return outputted, start_pos


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

  # LLAMA args
  parser.add_argument("--llama_prompt", type=str, default=None, help="Phrase to start with. Without this, it goes into chatbot mode")
  parser.add_argument("--llama_count", type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument("--llama_personality", type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")
  parser.add_argument("--llama_temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--llama_size", type=str, default="7B", help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B, 70B] for Gen 2, [7B, 13B, 34B] for Code LLaMA")
  parser.add_argument("--llama_gen", default="1", help="Generation of the model to use ['1', '2', 'code']")
  parser.add_argument("--llama_quantize", action="store_true", help="Quantize the weights to int8 in memory")
  parser.add_argument("--llama_model", type=Path, default=None, required=True, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")

  # TODO: vits args

  arguments = parser.parse_args()

  # Init whisper
  model, enc = init_whisper(arguments.whisper_model_name)

  # Init llama
  llama = LLaMa.build(arguments.llama_model, arguments.llama_model / "tokenizer.model", arguments.llama_gen, arguments.llama_size, arguments.llama_quantize)

  # Prepare personality
  toks, user_delim, resp_delim, end_delim = llama_prepare(llama)
  start_pos = len(toks)
  outputted = llama.tokenizer.decode(toks)
  sys.stdout.write(outputted)
  sys.stdout.flush()
  
  # Start child process for mic input
  q = mp.Queue()
  p = mp.Process(target=listener, args=(q,))
  p.daemon = True
  p.start()

  # Start the pipeline
  tokens = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  total = np.array([])
  prev_length = 0
  while (data := q.get()) is not None and len(total) < CHUNK * 30:
    total = np.concatenate([total, data])
    txt = voice2text(model, enc, total, tokens)
    print(txt)
    if not txt.endswith("<|endoftext|>"): 
      continue # Didn't reach the end of users' speech
    txt = clean_text(enc, txt)
    if len(txt) == prev_length: 
      tokens.pop()
      continue
    txt = txt[prev_length:]

    # generate response from llama
    outputted, start_pos = llama_generate(
      llama, txt, start_pos, outputted, 
      arguments.llama_count, arguments.llama_temperature, 
      user_delim, end_delim
    )
    print(outputted)

    # TODO: convert llama output to voice

    prev_length = len(txt)
    tokens.pop()
