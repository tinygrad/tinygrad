from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from examples.whisper import init_whisper, prep_audio, Whisper

import multiprocessing
import numpy as np
import pyaudio
import tiktoken

SAMPLE_RATE = 16000
FRAME_CHUNK = 1600
RECORD_SECONDS = 10

def stream_audio(queue: multiprocessing.Queue):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAME_CHUNK)

    print("I'm listening...")
    # for _ in range(0, int(SAMPLE_RATE / FRAME_CHUNK * RECORD_SECONDS)):
    while True:
        au_data = stream.read(FRAME_CHUNK)
        waveform = ((np.frombuffer(au_data, np.int16)/32768).astype(np.float32)*3)

        queue.put(waveform)

def stream_audio_v2(p: pyaudio.PyAudio, stream: pyaudio.Stream):
    print("I'm listening...")
    au_data = stream.read(FRAME_CHUNK)
    waveform = ((np.frombuffer(au_data, np.int16)/32768).astype(np.float32)*3)
    print("Done listening")

    return waveform

def transcribe_audio(audio: np.ndarray, model: Whisper, encoder: tiktoken.Encoding):
    lst = [encoder._special_tokens["<|startoftranscript|>"], encoder._special_tokens["<|notimestamps|>"]]
    log_spec = prep_audio(audio)

    encoded_audio = model.encoder(Tensor(log_spec)).realize()
    out = model.decoder(Tensor([lst]), encoded_audio).realize()
    idx = int(out[0,-1].argmax().numpy().item())
    lst.append(idx)
    dec = encoder.decode(lst)

    print(dec)
    if dec.endswith("<|endoftext|>"):
        print("Reached end of teext token!")
        lst.pop()


class TinyConv:
    ...


if __name__ == "__main__":
    model, enc = init_whisper("small.en" if getenv("SMALL") else "tiny.en")

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=stream_audio, args=(queue,))

    process.daemon = True
    process.start()

    lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
    while True:
        audio = queue.get()
        print(f"audio:\n{audio}")
        log_spec = prep_audio(audio)
        encoded_audio = model.encoder(Tensor(log_spec)).realize()
        out = model.decoder(Tensor([lst]), encoded_audio).realize()
        idx = int(out[0,-1].argmax().numpy().item())
        lst.append(idx)
        dec = enc.decode(lst)

        print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
        if dec.endswith("<|endoftext|>"):
            lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
            print("popping")

    # lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
    # total = None
    # did_read = False

    # while True:
    #     for _ in range(100):
    #         audio = queue.get()

    #         print(f"audio: {audio}")

    # while True:
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAME_CHUNK)
    #     audio = stream_audio_v2(p, stream)
    #     stream.close()
    #     p.terminate()

    #     lst = [enc._special_tokens["<|startoftranscript|>"], enc._special_tokens["<|notimestamps|>"]]
    #     print(f"---audio:\n{audio}")
    #     log_spec = prep_audio(audio)
    #     encoded_audio = model.encoder(Tensor(log_spec)).realize()
    #     out = model.decoder(Tensor([lst]), encoded_audio).realize()
    #     idx = int(out[0,-1].argmax().numpy().item())
    #     lst.append(idx)
    #     dec = enc.decode(lst)

    #     print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
    #     if dec.endswith("<|endoftext|>"):
    #         lst.pop()
    #         break


    # for _ in range(0, int(SAMPLE_RATE / FRAME_CHUNK * RECORD_SECONDS)):
    #     while not queue.empty() or total is None:
    #         waveform = queue.get()
    #         if total is None: total = waveform
    #         else: total = np.concatenate([total, waveform])
    #         did_read = True
    #     if did_read:
    #         log_spec = prep_audio(total)
    #         encoded_audio = model.encoder(Tensor(log_spec)).realize()

    #     out = model.decoder(Tensor([lst]), encoded_audio).realize()
    #     idx = int(out[0,-1].argmax().numpy().item())
    #     lst.append(idx)
    #     dec = enc.decode(lst)
    #     print(dec) # DO NOT REMOVE PRINT. IT'S VERY IMPORTANT
    #     if dec.endswith("<|endoftext|>"):
    #         lst.pop()
