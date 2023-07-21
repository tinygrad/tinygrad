import argparse, multiprocessing, numpy as np, torch, re, sys, simpleaudio as sa
import whisper, llama, vits
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes

# TODO: too much copy paste. tidy up

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Talk with tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--personality', type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")
  parser.add_argument('--speaker_id', type=int, default=6, help="Speaker ID")
  parser.add_argument('--whisper_size', type=str, default="tiny", help="Size of whisper model to use [tiny, small]")
  parser.add_argument('--llama_size', type=str, default="7B", help="Size of llama model to use [7B, 13B, 30B, 65B]")
  parser.add_argument('--vits_model', type=str, default="vctk", help="Vits model to use [ljs, vctk, mmts-tts, uma_trilingual, cjks, voistock]")
  parser.add_argument("--noise_scale", type=float, default=0.667, help="Vits noise scale")
  parser.add_argument("--noise_scale_w", type=float, default=0.8, help="Vits noise scale w")
  parser.add_argument("--length_scale", type=float, default=1, help="Vits length scale")
  args = parser.parse_args()

  # load models
  whisper_model, whisper_enc = whisper.load_model_and_enc(args.whisper_size == "small")

  llama_sp_model = llama.sp_model()
  llama_model = llama.load_model(args.llama_size)
  toks, start_pos, user_delim, end_delim = llama.encode_chatbot_preprompt(llama_model, llama_sp_model, args.personality)
  outputted = llama_sp_model.decode(toks) 

  # VITS SETUP START. TODO: pretty much none of this code should exist in this file
  vits_model_config = vits.MODELS[args.vits_model]
  # Load the hyperparameters from the config file.
  config_path = vits_model_config[0]
  vits.download_if_not_present(config_path, vits_model_config[2])
  hps = vits.get_hparams_from_file(config_path)
  # Load symbols, instantiate TextMapper and clean the text.
  if hps.__contains__("symbols"): symbols = hps.symbols
  elif args.vits_model == "mmts-tts": symbols = [x.replace("\n", "") for x in open(vits.download_if_not_present(vits.VITS_PATH / "vocab_mmts-tts.txt", "https://huggingface.co/facebook/mms-tts/raw/main/full_models/eng/vocab.txt"), encoding="utf-8").readlines()]
  else: symbols = ['_'] + list(';:,.!?¡¿—…"«»“” ') + list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') + list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
  vits_text_mapper = vits.TextMapper(apply_cleaners=True, symbols=symbols)
  vits_net_g = vits.load_model(vits_text_mapper.symbols, hps, vits_model_config)
  # VITS SETUP END

  while 1:
    # whisper -> llama -> vits

    # NOTE: this is pretty much copied from whisper.py. TODO: address code reuse
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=whisper.listener, args=(q,))
    p.daemon = True
    p.start()

    lst = [whisper_enc._special_tokens["<|startoftranscript|>"]]
    total = None
    did_read = False
    for i in range(0, int(whisper.RATE / whisper.CHUNK * whisper.RECORD_SECONDS)):
      while not q.empty() or total is None:
        waveform = q.get()
        if total is None: total = waveform
        else: total = np.concatenate([total, waveform], axis=1)
        did_read = True
      if did_read:
        last_total = total.shape[1]
        log_spec = whisper.prep_audio(torch.Tensor(total), whisper.RATE)
        encoded_audio = whisper_model.encoder(Tensor(log_spec)).realize()
      out = whisper_model.decoder(Tensor([lst]), encoded_audio).realize()
      idx = out[0,-1].numpy().argmax()
      lst.append(idx)
      dec = whisper_enc.decode(lst)
      if dec.endswith("<|endoftext|>"):
        #total = total[:, 320*(len(lst)-1):]
        lst = [whisper_enc._special_tokens["<|startoftranscript|>"]]
        if len(user_speech := re.sub('<\|.*?\|>', '', dec)) > 0:
          break
  
    print(user_speech)

    # throw into llama

    # add tokens from user in chatbot mode
    outputted += user_delim + user_speech + "\n"

    new_toks = [llama_sp_model.bos_id()] + llama_sp_model.encode(outputted)
    assert toks == new_toks[:len(toks)]
    toks = new_toks
    assert outputted == llama_sp_model.decode(toks)

    text_to_synthesize = ""

    while 1:
      logits = llama_model(Tensor([toks[start_pos:]]), start_pos).realize()
      tok = llama.sample(logits, 0.7) # args.temperature in llama.py
      # use the kv cache
      start_pos = len(toks)
      # add the new token
      toks.append(tok)
      # TODO: this is a hack to deal with spaces. i think the decode is fast though, so who cares?
      cur = llama_sp_model.decode(toks)
      out = cur[len(outputted):]
      text_to_synthesize += out
      sys.stdout.write(out)
      sys.stdout.flush()
      outputted = cur
      if outputted.endswith(end_delim): break

    text_to_synthesize = text_to_synthesize.split(": ", 1)[1].replace(" [EOS]", "")

    # text-to-speech
    if args.vits_model == "mmts-tts": text_to_synthesize = vits_text_mapper.filter_oov(text_to_synthesize.lower())
    stn_tst = vits_text_mapper.get_text(text_to_synthesize, hps.data.add_blank, hps.data.text_cleaners)
    x_tst, x_tst_lengths = stn_tst.unsqueeze(0), Tensor([stn_tst.shape[0]], dtype=dtypes.int64)
    sid = Tensor([args.speaker_id], dtype=dtypes.int64)

    audio_tensor = vits_net_g.infer(x_tst, x_tst_lengths, sid, args.noise_scale, args.length_scale, args.noise_scale_w, emotion_embedding=None,max_y_length_estimate_scale=None)[0, 0].realize()

    # Save the audio output.
    audio_data = (np.clip(audio_tensor.numpy(), -1.0, 1.0) * 32767).astype(np.int16)

    play = sa.play_buffer(audio_data, 1, 2, hps.data.sampling_rate)
    play.wait_done()
