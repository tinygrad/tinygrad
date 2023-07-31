import numpy as np
import pyaudio

import examples.llama as llama
import examples.vits as vits
import examples.whisper as whisper
from tinygrad.tensor import Tensor

class TokenProcessor:
  def __init__(self, audio_generator):
    self.buffer, self.audio_generator = "", audio_generator
  def process_tokens(self, tokens):
    print("received tokens: ", tokens)
    for token in tokens:
      if token in '!?.,;':                            # If: token is a punctuation mark
        punct_index = self.buffer.find(token)         # Find index of punctuation mark in the buffer
        if punct_index != -1:
          substring = self.buffer[:punct_index]
          self.audio_generator(substring)             # Generate audio sample with substring before the punctuation mark
          self.buffer = self.buffer[punct_index + 1:] # Update buffer with remainder of the string
      else: self.buffer += " " + token                # Else: Concatenate token to the buffer
  def clear_buffer(self): self.buffer = ""

class Bot:

  def __init__(self, streaming=False):
    self.streaming = streaming
    self.language_model = llama.LLAMAModel()
    self.llm_output, llm_tokens = self.language_model.prepare_personality_kv_cache()
    self.llm_start_pos = len(llm_tokens)
    self.speech_model = vits.VITSModel()
    self.listening_model = whisper.WhisperModel()
    self.token_processor = TokenProcessor(self.say)
    self.pya = pyaudio.PyAudio()
    self.audio_stream = self.pya.open(format=self.pya.get_format_from_width(width=2), channels=1, rate=self.speech_model.hps.data.sampling_rate, output=True)

  def start_listening(self):
    self.listening_model.start_listening(self.respond_to)
    if len(self.token_processor.buffer) > 0: # if there is any remaining text, synthesize it
      self.say(self.token_processor.buffer)
      self.token_processor.clear_buffer()

  def respond_to(self, incoming_text: str):
    print("heard: ", incoming_text)
    self.llm_output = self.llm_output + self.language_model.user_delim + incoming_text + "\n"
    previous_output_len = len(self.llm_output)
    self.llm_start_pos, self.llm_output = self.language_model.complete(
      prompt=self.llm_output,
      start_pos=self.llm_start_pos,
      text_gen_callback=self.say if self.streaming else lambda x: llama.print_text(x),
      end_condition=lambda x: x.endswith(self.language_model.end_delim)
    )
    if not self.streaming:
      new_output = self.llm_output[previous_output_len:]
      if (start_index := new_output.find(":")) != -1:
        response = new_output[start_index + 1:].replace(self.language_model.end_delim, "")
        print("response: ", response)
        self.say(response)
      else:
        print("Malformed output!", new_output)

  def say(self, text):
    audio = self.speech_model.tts(text)
    self.audio_stream.write(audio.tobytes())

  def kill(self):
    self.audio_stream.stop_stream()
    self.audio_stream.close()
    self.pya.terminate()

if __name__ == '__main__':
  Tensor.no_grad = True
  seed = 1337
  Tensor.manual_seed(seed)
  np.random.seed(seed)
  bot = Bot()
  bot.respond_to("Hello, my name is Stan. What is your name?")
  bot.respond_to("Cool! what is your favorite color?")
  bot.kill()