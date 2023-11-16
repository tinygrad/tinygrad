import argparse
import wave
import multiprocessing as mp
import sys
import string
from pathlib import Path

import numpy as np
import pyaudio
from llama import LLaMa
from tiktoken import Encoding
from whisper import Whisper, init_whisper, prep_audio
from vits import download_if_not_present, load_checkpoint, TextMapper, load_model, get_hparams_from_file, Synthesizer

from tinygrad.tensor import Tensor
from pathlib import Path
from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor

# Whisper constants
RATE = 16000
CHUNK = 1600

# vits constants
VITS_PATH = Path(__file__).parents[1] / "weights/VITS/"
VITS_MODELS = { # config_path, weights_path, config_url, weights_url
  "ljs": (VITS_PATH / "config_ljs.json", VITS_PATH / "pretrained_ljs.pth", "https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/ljs_base.json", "https://drive.google.com/uc?export=download&id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT&confirm=t"),
  "vctk": (VITS_PATH / "config_vctk.json", VITS_PATH / "pretrained_vctk.pth", "https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/vctk_base.json", "https://drive.google.com/uc?export=download&id=11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru&confirm=t"),
  "mmts-tts": (VITS_PATH / "config_mmts-tts.json", VITS_PATH / "pretrained_mmts-tts.pth", "https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/config.json", "https://huggingface.co/facebook/mms-tts/resolve/main/full_models/eng/G_100000.pth"),
  "uma_trilingual": (VITS_PATH / "config_uma_trilingual.json", VITS_PATH / "pretrained_uma_trilingual.pth", "https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/raw/main/configs/uma_trilingual.json", "https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth"),
  "cjks": (VITS_PATH / "config_cjks.json", VITS_PATH / "pretrained_cjks.pth", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/14/config.json", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/14/model.pth"),
  "voistock": (VITS_PATH / "config_voistock.json", VITS_PATH / "pretrained_voistock.pth", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/15/config.json", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/15/model.pth"),
}
VITS_Y_LENGTH_ESTIMATE_SCALARS = {"ljs": 2.8, "vctk": 1.74, "mmts-tts": 1.9, "uma_trilingual": 2.3, "cjks": 3.3, "voistock": 3.1}

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


def tts():
  pass


def download_vits_model(
  model_to_use: str
):
  if model_to_use not in VITS_MODELS: raise ValueError("Such model was not found")
  cfg_path, hp_path, cfg_url, hp_url = VITS_MODELS[model_to_use]
  return download_if_not_present(cfg_path, cfg_url), download_if_not_present(hp_path, hp_url)

def init_vits(
  config_path: Path,
  weights_path: Path,
  emotion_path: Path,
  vocab_path: Path,
  speaker_id: int,
  seed: int,
):
  # Load the hyperparameters from the config file.
  hps = get_hparams_from_file(config_path)

  # If model has multiple speakers, validate speaker id and retrieve name if available.
  model_has_multiple_speakers = hps.data.n_speakers > 0
  if model_has_multiple_speakers:
    if speaker_id >= hps.data.n_speakers: raise ValueError(f"Speaker ID {speaker_id} is invalid for this model.")
    if "speakers" in hps: # maps speaker ids to names
      speakers = hps.speakers
      if isinstance(speakers, list): speakers = {speaker: i for i, speaker in enumerate(speakers)}

  # Load emotions if any. TODO: find an english model with emotions, this is untested atm.
  emotion_embedding = None
  if emotion_path is not None:
    if emotion_path.endswith(".npy"): emotion_embedding = Tensor(np.load(emotion_path), dtype=dtypes.int64).unsqueeze(0)
    else: raise ValueError("Emotion path must be a .npy file.")

  # Load symbols, instantiate TextMapper and clean the text.
  if "symbols" in hps: symbols = hps.symbols
  elif vocab_path is not None: symbols = list(filter(lambda x: x != "\n", open(vocab_path).read()))
  else: symbols = ['_', ' '] + list(string.punctuation) + list(string.ascii_letters) + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
  text_mapper = TextMapper(apply_cleaners=True, symbols=symbols)

  # Load the model.
  if seed is not None:
    Tensor.manual_seed(seed)
    np.random.seed(seed)
  
  synth = Synthesizer(len(symbols), hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, n_speakers = hps.data.n_speakers, **hps.model)
  load_checkpoint(weights_path, synth, None)
  return synth, emotion_embedding, text_mapper, hps, model_has_multiple_speakers


def listener(q: mp.Queue):
  try:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Talk")
    while True: q.put(((np.frombuffer(stream.read(CHUNK), np.int16)/32768).astype(np.float32)*3))
  except: q.put(None)
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
  parser.add_argument("--llama_prompt", type=str, default=None, help="Phrase to start with. Without this, it goes into chatbot mode")
  parser.add_argument("--llama_count", type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument("--llama_personality", type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")
  parser.add_argument("--llama_temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--llama_size", type=str, default="7B", help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B, 70B] for Gen 2, [7B, 13B, 34B] for Code LLaMA")
  parser.add_argument("--llama_gen", default="1", help="Generation of the model to use ['1', '2', 'code']")
  parser.add_argument("--llama_quantize", action="store_true", help="Quantize the weights to int8 in memory")
  parser.add_argument("--llama_model", type=Path, default=None, required=True, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")

  # vits args
  parser.add_argument("--vits_model_to_use", default="vctk", help="Specify the model to use. Default is 'vctk'.")
  parser.add_argument("--vits_speaker_id", type=int, default=6, help="Specify the speaker ID. Default is 6.")
  parser.add_argument("--vits_out_path", default=None, required=True, help="Specify the full output path.")
  parser.add_argument("--vits_noise_scale", type=float, default=0.667, help="Specify the noise scale. Default is 0.667.")
  parser.add_argument("--vits_noise_scale_w", type=float, default=0.8, help="Specify the noise scale w. Default is 0.8.")
  parser.add_argument("--vits_length_scale", type=float, default=1, help="Specify the length scale. Default is 1.")
  parser.add_argument("--vits_seed", type=int, default=None, help="Specify the seed (set to None if no seed). Default is 1337.")
  parser.add_argument("--vits_num_channels", type=int, default=1, help="Specify the number of audio output channels. Default is 1.")
  parser.add_argument("--vits_sample_width", type=int, default=2, help="Specify the number of bytes per sample, adjust if necessary. Default is 2.")
  parser.add_argument("--vits_emotion_path", type=Path, default=None, help="Specify the path to emotion reference.")
  parser.add_argument("--vits_estimate_max_y_length", type=str, default=False, help="If true, overestimate the output length and then trim it to the correct length, to prevent premature realization, much more performant for larger inputs, for smaller inputs not so much. Default is False.")
  parser.add_argument("--vits_vocab_path", type=Path, default=None, help="Path to the TTS vocabulary.")

  arguments = parser.parse_args()

  # Init models
  model, enc = init_whisper(arguments.whisper_model_name)
  llama = LLaMa.build(arguments.llama_model, arguments.llama_model / "tokenizer.model", arguments.llama_gen, arguments.llama_size, arguments.llama_quantize)
  cfg_path, hp_path = download_vits_model(arguments.vits_model_to_use)
  synth, emotion_embedding, text_mapper, hps, model_has_multiple_speakers = init_vits(cfg_path, hp_path, arguments.vits_emotion_path, arguments.vits_vocab_path, arguments.vits_speaker_id, arguments.vits_seed)

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
  lock = mp.Lock()

  # Start the pipeline
  tokens = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  total = np.array([])
  prev_length = 0
  while (data := q.get()) is not None:
    total = np.concatenate([total, data])

    # convert voice to text
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

    # TODO: convert llama output to voice

    prev_length = len(txt)
    tokens.pop()
