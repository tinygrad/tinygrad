import argparse
import pyaudio
import wave
import time
from pathlib import Path
import socket
import numpy as np 
from whisper import prep_audio, listener, init_whisper
import multiprocessing
from os import getenv
from vits import download_if_not_present, get_hparams_from_file, TextMapper, load_model
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes

RATE = 16000
CHUNK = 1600
RECORD_SECONDS = 5
VITS_PATH = Path(__file__).parents[1] / "weights/VITS/"
MODELS = { # config_path, weights_path, config_url, weights_url
  "ljs": (VITS_PATH / "config_ljs.json", VITS_PATH / "pretrained_ljs.pth", "https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/ljs_base.json", "https://drive.google.com/uc?export=download&id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT&confirm=t"),
  "vctk": (VITS_PATH / "config_vctk.json", VITS_PATH / "pretrained_vctk.pth", "https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/vctk_base.json", "https://drive.google.com/uc?export=download&id=11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru&confirm=t"),
  "mmts-tts": (VITS_PATH / "config_mmts-tts.json", VITS_PATH / "pretrained_mmts-tts.pth", "https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/config.json", "https://huggingface.co/facebook/mms-tts/resolve/main/full_models/eng/G_100000.pth"),
  "uma_trilingual": (VITS_PATH / "config_uma_trilingual.json", VITS_PATH / "pretrained_uma_trilingual.pth", "https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/raw/main/configs/uma_trilingual.json", "https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth"),
  "cjks": (VITS_PATH / "config_cjks.json", VITS_PATH / "pretrained_cjks.pth", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/14/config.json", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/14/model.pth"),
  "voistock": (VITS_PATH / "config_voistock.json", VITS_PATH / "pretrained_voistock.pth", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/15/config.json", "https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/15/model.pth"),
}
Y_LENGTH_ESTIMATE_SCALARS = {"ljs": 2.8, "vctk": 1.74, "mmts-tts": 1.9, "uma_trilingual": 2.3, "cjks": 3.3, "voistock": 3.1}


def llama_generate(prompt: str) -> str:
  s.send(prompt.encode()) 
  result = s.recv(1024).decode()
  result = result.replace("[EOS]", "")
  return result[next(n for n, i in enumerate(result) if i == ":"):]


def txt2speech(
    text_to_synthesize: str, 
    model_to_use="vctk", 
    speaker_id=6, 
    emotion_path=None, 
    seed=1337,
    noise_scale=0.667,
    length_scale=1,
    noise_scale_w=0.8,
    estimate_max_y_length=False,
    out_path="result.wav",
    base_name="test",
    num_channels=1,
    sample_width=2
) -> str:
  model_config = MODELS[model_to_use]

  # Load the hyperparameters from the config file.
  config_path = model_config[0]
  download_if_not_present(config_path, model_config[2])
  hps = get_hparams_from_file(config_path)

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
  elif model_to_use == "mmts-tts": symbols = [x.replace("\n", "") for x in open(download_if_not_present(VITS_PATH / "vocab_mmts-tts.txt", "https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/vocab.txt"), encoding="utf-8").readlines()]
  else: symbols = ['_'] + list(';:,.!?¡¿—…"«»“” ') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
  text_mapper = TextMapper(apply_cleaners=True, symbols=symbols)

  # Load the model.
  Tensor.no_grad = True
  if seed is not None:
    Tensor.manual_seed(seed)
    np.random.seed(seed)
  net_g = load_model(text_mapper.symbols, hps, model_config)

  # Convert the input text to a tensor.
  if model_to_use == "mmts-tts": text_to_synthesize = text_mapper.filter_oov(text_to_synthesize.lower())
  stn_tst = text_mapper.get_text(text_to_synthesize, hps.data.add_blank, hps.data.text_cleaners)
  x_tst, x_tst_lengths = stn_tst.unsqueeze(0), Tensor([stn_tst.shape[0]], dtype=dtypes.int64)
  sid = Tensor([speaker_id], dtype=dtypes.int64) if model_has_multiple_speakers else None

  # Perform inference.
  audio_tensor = net_g.infer(x_tst, x_tst_lengths, sid, noise_scale, length_scale, noise_scale_w, emotion_embedding=emotion_embedding,
                             max_y_length_estimate_scale=Y_LENGTH_ESTIMATE_SCALARS[model_to_use] if estimate_max_y_length else None)[0, 0].realize()

  # Save the audio output.
  audio_data = (np.clip(audio_tensor.numpy(), -1.0, 1.0) * 32767).astype(np.int16)
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with wave.open(str(out_path), 'wb') as wav_file:
    wav_file.setnchannels(num_channels)
    wav_file.setsampwidth(sample_width)
    wav_file.setframerate(hps.data.sampling_rate)
    wav_file.setnframes(len(audio_data))
    wav_file.writeframes(audio_data.tobytes())
  return out_path


def play_audio(wave_path: str):
  f = wave.open(str(wave_path), "rb")  
  #instantiate PyAudio  
  p = pyaudio.PyAudio()  
  #open stream  
  stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                  channels = f.getnchannels(),  
                  rate = f.getframerate(),  
                  output = True)  
  #read data  
  data = f.readframes(CHUNK)
  while data:
    stream.write(data)
    data = f.readframes(CHUNK)
  stream.stop_stream()
  stream.close()
  p.terminate()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", type=str, default="localhost")
  parser.add_argument("--port", type=int, default=5000)
  args = parser.parse_args()
  model, enc  = init_whisper("small.en" if getenv("SMALL") else "tiny.en")
  q = multiprocessing.Queue()
  p = multiprocessing.Process(target=listener, args=(q,))
  p.daemon = True
  p.start()

  lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
  total = None
  did_read = False

  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect((args.host, args.port))
  prev_length = 0
  while True:
    while not q.empty() or total is None:
      waveform = q.get()
      if total is None: total = waveform
      else: total = np.concatenate([total, waveform])
      did_read = True
    if did_read:
      log_spec = prep_audio(total)
      encoded_audio = model.encoder(Tensor(log_spec)).realize() # it sometimes laggs and passes longer audio
    out = model.decoder(Tensor([lst]), encoded_audio).realize()
    lst.append(int(out[0,-1].argmax().numpy().item()))
    dec = enc.decode(lst)
    print(dec)
    if dec.endswith("<|endoftext|>"):
      user_input = dec.split("[BLANK_AUDIO]")[1:-1]
      print(user_input)
      if len(user_input) > 0 and len(user_input) != prev_length:
        text = llama_generate(user_input[-1])
        wave_path = txt2speech(text, out_path="res.wav")
        play_audio(wave_path)
        prev_length = len(user_input)
      lst.pop()
