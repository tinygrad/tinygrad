import argparse
import multiprocessing as mp
import os
import re
import sys
import time
from contextlib import contextmanager
from itertools import cycle
from pathlib import Path

import numpy as np
import pyaudio
import yaml
from llama import LLaMa
from tiktoken import Encoding
from vits import MODELS as VITS_MODELS
from vits import VITS_PATH, Y_LENGTH_ESTIMATE_SCALARS, HParams, Synthesizer, TextMapper, get_hparams_from_file, load_model
from whisper import Whisper, init_whisper, prep_audio

from tinygrad.helpers import Timing, dtypes, fetch
from tinygrad.tensor import Tensor

# Whisper constants
RATE = 16000
CHUNK = 1600

def clean_text(enc: Encoding, txt: str) -> str:
  for t in enc.special_tokens_set:
    txt = txt.replace(t, "")
  txt = re.sub(r"\(.+\)", "", txt)
  return txt.replace("[BLANK_AUDIO]", "").strip()


def voice2text(model: Whisper, enc: Encoding, encoded_audio: Tensor, tokens: list[int]):
  out = model.decoder(Tensor([tokens]), 0, encoded_audio, streaming=True).realize()
  tokens.append(int(out[0,-1].argmax().numpy().item()))
  return enc.decode(tokens)


def llama_prepare(llama: LLaMa, temperature: float, pre_prompt_path: Path) -> tuple[list[int], str, str, str]:
  config = yaml.safe_load(open(str(pre_prompt_path)).read())
  pre_prompt, user_delim, resp_delim, end_delim = config["pre_prompt"], config["user_delim"], config["resp_delim"], config["end_delim"]
  pre_prompt += ''.join(f"{user_delim}{i['user_prompt']}\n{resp_delim}{i['resp_prompt']}{end_delim}" for i in config["examples"])

  toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(pre_prompt)
  llama.model(Tensor([toks]), 0, temperature).realize()  # NOTE: outputs are not used
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
  print() # because the output is flushed
  return outputted, start_pos


def tts(
  text_to_synthesize: str,
  synth: Synthesizer,
  hps: HParams,
  emotion_embedding: Path,
  speaker_id: int,
  model_to_use: str,
  noise_scale: float,
  noise_scale_w: float,
  length_scale: float,
  estimate_max_y_length: bool,
  text_mapper: TextMapper,
  model_has_multiple_speakers: bool,
  batch_size = 500
):
  if model_to_use == "mmts-tts": text_to_synthesize = text_mapper.filter_oov(text_to_synthesize.lower())

  # Convert the input text to a tensor.
  stn_tst = text_mapper.get_text(text_to_synthesize, hps.data.add_blank, hps.data.text_cleaners)
  init_shape = stn_tst.shape
  assert init_shape[0] < batch_size, "text is too long"
  stn_tst = stn_tst.pad(((0, batch_size - init_shape[0]),), 1)
  x_tst, x_tst_lengths = stn_tst.unsqueeze(0), Tensor([init_shape[0]], dtype=dtypes.int64)
  sid = Tensor([speaker_id], dtype=dtypes.int64) if model_has_multiple_speakers else None

  # Perform inference.
  audio_tensor = synth.infer(x_tst, x_tst_lengths, sid, noise_scale, length_scale, noise_scale_w, emotion_embedding=emotion_embedding,
                             max_y_length_estimate_scale=Y_LENGTH_ESTIMATE_SCALARS[model_to_use] if estimate_max_y_length else None, batch_size=batch_size)[0, 0]
  # Save the audio output.
  audio_data = (np.clip(audio_tensor.numpy(), -1.0, 1.0) * 32767).astype(np.int16)
  return audio_data


def init_vits(
  model_to_use: str,
  emotion_path: Path,
  speaker_id: int,
  seed: int,
):
  model_config = VITS_MODELS[model_to_use]

  # Load the hyperparameters from the config file.
  hps = get_hparams_from_file(fetch(model_config[0]))

  # If model has multiple speakers, validate speaker id and retrieve name if available.
  model_has_multiple_speakers = hps.data.n_speakers > 0
  if model_has_multiple_speakers:
    if speaker_id >= hps.data.n_speakers: raise ValueError(f"Speaker ID {speaker_id} is invalid for this model.")
    if hps.__contains__("speakers"): # maps speaker ids to names
      speakers = hps.speakers
      if isinstance(speakers, list): speakers = {speaker: i for i, speaker in enumerate(speakers)}

  # Load emotions if any. TODO: find an english model with emotions, this is untested atm.
  emotion_embedding = None
  if emotion_path is not None:
    if emotion_path.endswith(".npy"): emotion_embedding = Tensor(np.load(emotion_path), dtype=dtypes.int64).unsqueeze(0)
    else: raise ValueError("Emotion path must be a .npy file.")

  # Load symbols, instantiate TextMapper and clean the text.
  if hps.__contains__("symbols"): symbols = hps.symbols
  elif model_to_use == "mmts-tts": symbols = [x.replace("\n", "") for x in fetch("https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/vocab.txt").open(encoding="utf-8").readlines()]
  else: symbols = ['_'] + list(';:,.!?¡¿—…"«»“” ') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
  text_mapper = TextMapper(apply_cleaners=True, symbols=symbols)

  # Load the model.
  Tensor.no_grad = True
  if seed is not None:
    Tensor.manual_seed(seed)
    np.random.seed(seed)
  net_g = load_model(text_mapper.symbols, hps, model_config)

  return net_g, emotion_embedding, text_mapper, hps, model_has_multiple_speakers


@contextmanager
def output_stream(num_channels: int, sample_rate: int):
  try:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=num_channels, rate=sample_rate, output=True)
    yield stream
  finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

@contextmanager
def log_writer():
  try:
    logs = []
    yield logs
  finally:
    sep = "="*os.get_terminal_size()[1]
    print(sep[:-1])
    print("CHAT LOG")
    print(*logs, sep="\n")
    print(sep)


def listener(q: mp.Queue, event: mp.Event):
  try:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    spinner = cycle(['-', '/', '|', '\\'])
    n = 0
    while True:
      data = stream.read(CHUNK) # read data to avoid overflow
      if event.is_set():
        if n % 4 == 0:
          sys.stdout.write(f"listening {next(spinner)}\r")
          sys.stdout.flush()
        q.put(((np.frombuffer(data, np.int16)/32768).astype(np.float32)*3))
        n += 1
  finally:
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
  Tensor.no_grad = True
  # Parse CLI arguments
  parser = argparse.ArgumentParser("Have a tiny conversation with tinygrad")

  # Whisper args
  parser.add_argument("--whisper_model_name", type=str, default="tiny.en")

  # LLAMA args
  parser.add_argument("--llama_pre_prompt_path", type=Path, default=Path(__file__).parent / "conversation_data" / "pre_prompt_stacy.yaml", help="Path to yaml file which contains all pre-prompt data needed. ")
  parser.add_argument("--llama_count", type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument("--llama_personality", type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")
  parser.add_argument("--llama_temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--llama_quantize", action="store_true", help="Quantize the weights to int8 in memory")
  parser.add_argument("--llama_model", type=Path, default=None, required=True, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")
  parser.add_argument("--llama_gen", type=str, default="tiny", required=False, help="Generation of the model to use")
  parser.add_argument("--llama_size", type=str, default="1B", required=False, help="Size of model to use")
  parser.add_argument("--llama_tokenizer", type=Path, default=None, required=True, help="Path to llama tokenizer.model")

  # vits args
  parser.add_argument("--vits_model_to_use", default="vctk", help="Specify the model to use. Default is 'vctk'.")
  parser.add_argument("--vits_speaker_id", type=int, default=6, help="Specify the speaker ID. Default is 6.")
  parser.add_argument("--vits_noise_scale", type=float, default=0.667, help="Specify the noise scale. Default is 0.667.")
  parser.add_argument("--vits_noise_scale_w", type=float, default=0.8, help="Specify the noise scale w. Default is 0.8.")
  parser.add_argument("--vits_length_scale", type=float, default=1, help="Specify the length scale. Default is 1.")
  parser.add_argument("--vits_seed", type=int, default=None, help="Specify the seed (set to None if no seed). Default is 1337.")
  parser.add_argument("--vits_num_channels", type=int, default=1, help="Specify the number of audio output channels. Default is 1.")
  parser.add_argument("--vits_sample_width", type=int, default=2, help="Specify the number of bytes per sample, adjust if necessary. Default is 2.")
  parser.add_argument("--vits_emotion_path", type=Path, default=None, help="Specify the path to emotion reference.")
  parser.add_argument("--vits_estimate_max_y_length", type=str, default=False, help="If true, overestimate the output length and then trim it to the correct length, to prevent premature realization, much more performant for larger inputs, for smaller inputs not so much. Default is False.")
  parser.add_argument("--vits_vocab_path", type=Path, default=None, help="Path to the TTS vocabulary.")

  # conversation args
  parser.add_argument("--phrase_timeout", type=int, default=5, help="Specify how long phrases should be recorded")

  args = parser.parse_args()

  # Init models
  model, enc = init_whisper(args.whisper_model_name)
  synth, emotion_embedding, text_mapper, hps, model_has_multiple_speakers = init_vits(args.vits_model_to_use, args.vits_emotion_path, args.vits_speaker_id, args.vits_seed)

  # Prepare personality
  llama = LLaMa.build(args.llama_model, args.llama_tokenizer or args.llama_model / "tokenizer.model", args.llama_gen, args.llama_size, args.llama_quantize)
  toks, user_delim, resp_delim, end_delim = llama_prepare(llama, args.llama_temperature, args.llama_pre_prompt_path)
  start_pos = len(toks)
  outputted = llama.tokenizer.decode(toks)

  # Start child process for mic input
  q = mp.Queue()
  is_listening_event = mp.Event()
  p = mp.Process(target=listener, args=(q, is_listening_event,))
  p.daemon = True
  p.start()

  # JIT tts
  for i in ["Hello, I'm a chat bot", "I am capable of doing a lot of things"]:
    tts(
      i, synth, hps, emotion_embedding,
      args.vits_speaker_id, args.vits_model_to_use, args.vits_noise_scale,
      args.vits_noise_scale_w, args.vits_length_scale,
      args.vits_estimate_max_y_length, text_mapper, model_has_multiple_speakers
    )

  # Start the pipeline
  with output_stream(args.vits_num_channels, hps.data.sampling_rate) as stream, log_writer() as log:
    while True:
      tokens = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
      total = np.array([])

      # Listen to mic input
      is_listening_event.set()
      start = time.time()
      while time.time() - start < args.phrase_timeout: total = np.concatenate([total, q.get()])
      is_listening_event.clear()

      # Transcribe text
      s = time.perf_counter()
      with Timing("transcription: "):
        encoded_audio = model.encoder.encode(Tensor(prep_audio(total.reshape(1, -1), 1)))
        while not (txt := voice2text(model, enc, encoded_audio, tokens)).endswith("<|endoftext|>"): print(txt)
        txt = clean_text(enc, txt)
        log.append(user_delim + txt)

      # Generate with llama
      with Timing("llama generation: "):
        outputted, start_pos = llama_generate(llama, txt, start_pos, outputted)
        response = outputted.splitlines()[-1].replace(resp_delim.strip(), "").replace(end_delim.strip(), "")
        log.append(resp_delim + response)

      # Convert to voice
      with Timing("tts: "):
        audio_data = tts(
          response, synth, hps, emotion_embedding,
          args.vits_speaker_id, args.vits_model_to_use, args.vits_noise_scale,
          args.vits_noise_scale_w, args.vits_length_scale,
          args.vits_estimate_max_y_length, text_mapper, model_has_multiple_speakers
        )
      with Timing("audio play: "): stream.write(audio_data.tobytes())
      log.append(f"Total: {time.perf_counter() - s}")
