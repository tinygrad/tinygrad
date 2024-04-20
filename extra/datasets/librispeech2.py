import subprocess
import os
import glob
import multiprocessing
import functools
import tqdm
import json
import sentencepiece
import pickle
from nvidia import dali
import nvidia.dali.plugin.pytorch
import math
import time
import torch
import tinygrad.runtime.autogen.cuda as cuda
import ctypes
from tinygrad import Device
import random
import hashlib

basedir = "/datasets"

globalt1 = time.time()

def run(cmd):
  with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True,shell=True) as p:
    for line in p.stdout:
      print(line, end='')
  if p.returncode != 0:
    raise Exception(f"FAILED CMD: {cmd}")

def build_input_arr(input_dir):
  txt_files = glob.glob(os.path.join(input_dir, '**', '*.trans.txt'),
                        recursive=True)
  input_data = []
  for txt_file in txt_files:
    rel_path = os.path.relpath(txt_file, input_dir)
    with open(txt_file) as fp:
      for line in fp:
        fname, _, transcript = line.partition(' ')
        rel_path_dir = os.path.dirname(rel_path)
        input_data.append(dict(input_relpath=rel_path_dir,
                                input_fname=fname+'.flac',
                                transcript=transcript))
  return input_data

def _sox(cmd):
  x = subprocess.run(cmd.split(), capture_output=True)
  out = x.stdout.decode().strip()
  err = x.stderr.decode().strip()
  if x.returncode !=0:
     raise Exception(f"Returncode {x.returncode}, err:\n{err}")
  return out

def convert(data, folder, dataset, remove=True, overwrite=False):
  file1 = folder+"/"+data["input_relpath"]+"/"+data["input_fname"]
  folder2 = folder+"-wav/"+data["input_relpath"]
  os.makedirs(folder2, exist_ok=True)
  file2 = folder2+"/"+data["input_fname"].replace(".flac",".wav")
  if overwrite or not os.path.exists(file2):
    _sox(f"sox -D -V2 -c 1 {file1} {file2} speed 1.000000 rate -h 16000.000000")
  dur = float(_sox(f"sox --i -D {file2}"))
  if remove:
    os.remove(file1)
  return {"transcript": data["transcript"], "original_duration": dur, "files": [{"fname": dataset+"-wav/"+data["input_relpath"]+"/"+data["input_fname"].replace(".flac",".wav")}]}

def get_data(dataset, remove=True, overwrite=False, hash=True):
  site = (
  # "https://www.openslr.org/resources/12/" #us
  # "https://us.openslr.org/resources/12/" #us
  # "https://openslr.elda.org/resources/12/" #eu
  "https://huggingface.co/datasets/k2-fsa/LibriSpeech/resolve/main/"
  )
  if not overwrite and os.path.exists(f"{basedir}/LibriSpeech/{dataset}/"):
    return
  if overwrite or (not os.path.exists(f"{basedir}/{dataset}.tar.gz") and not os.path.exists(f"{basedir}/LibriSpeech/{dataset}/")):
    run(f"wget {site}{dataset}.tar.gz -P {basedir}/")
    if hash:
      print(f"Checking hash")
      fpath = f"{basedir}/{dataset}.tar.gz"
      file_hash = hashlib.md5()
      chunksize = 1024*1024
      total_chunks = math.ceil(os.path.getsize(fpath)/chunksize)
      with open(fpath, "rb") as fp:
        for chunk in tqdm.tqdm(iter(lambda: fp.read(chunksize), b""), total=total_chunks):
          file_hash.update(chunk)
        if file_hash.hexdigest() != hashes[dataset]:
          raise Exception("Hash wrong")
        else:
          print(f"Hash matched")
  if overwrite or not os.path.exists(f"{basedir}/LibriSpeech/{dataset}/"):
    print(f"untarring")
    run(f"pv {basedir}/{dataset}.tar.gz | tar -xz -C {basedir}/")
  if remove:
    os.remove(f"{basedir}/{dataset}.tar.gz")

def convert_data(dataset,remove=True, overwrite=False):
  jsonfile=f"{basedir}/LibriSpeech/librispeech-{dataset}-wav.json"
  if overwrite or not os.path.exists(jsonfile):
    print(f"Running sox to convert to wav")
    with multiprocessing.Pool(os.cpu_count()) as p:
      filedata = build_input_arr(f"{basedir}/LibriSpeech/{dataset}")
      converter = functools.partial(convert, folder=f"{basedir}/LibriSpeech/{dataset}", dataset=dataset, remove=remove, overwrite=overwrite)
      data = list(tqdm.tqdm(p.imap(converter, filedata), total=len(filedata)))

      with open(jsonfile, "w") as fp:
        json.dump(data, fp, indent=2)

def sentencepieces(overwrite=False):
  if not overwrite and os.path.exists(f"{basedir}/LibriSpeech/librispeech1023.model"):
    return
  transcripts = []
  for dataset in trainingsets:
    filename = f"{basedir}/LibriSpeech/librispeech-{dataset}-wav.json"
    with open(filename) as fp:
      data = json.load(fp)
      transcripts.extend([el["transcript"] for el in data])
  sentencepiece.SentencePieceTrainer.train(sentence_iterator=iter(transcripts), model_prefix='librispeech1023', vocab_size=1023, character_coverage=1.0, bos_id=-1, eos_id=-1, model_type='unigram')
  for filename in ["librispeech1023.vocab","librispeech1023.model"]:
    os.rename(f"{os.getcwd()}/{filename}",f"{basedir}/LibriSpeech/{filename}")

def pickled_data(overwrite=False):
  for dataset in trainingsets+valsets:
    filename = f"{basedir}/LibriSpeech/librispeech-{dataset}-wav.json"
    output_file = f"{basedir}/LibriSpeech/librispeech-{dataset}-wav-tokenized.pkl"
    if not overwrite and os.path.exists(output_file):
      continue
    with open(filename) as fp:
      data = json.load(fp)
    spm = sentencepiece.SentencePieceProcessor(model_file=f"{basedir}/LibriSpeech/librispeech1023.model")
    data2 = [dict(tokenized_transcript=spm.encode(el["transcript"]), original_duration=el["original_duration"], fname=el["files"][-1]["fname"]) for el in data]
    with open(output_file, 'wb') as f:
      pickle.dump(data2, f)

@functools.cache
def get_datalist(eval=False):
  sets = valsets if eval else trainingsets
  data = []
  for dataset in sets:
    filename = f"{basedir}/LibriSpeech/librispeech-{dataset}-wav-tokenized.pkl"
    with open(filename,"rb") as fp:
      data2 = pickle.load(fp)
    data.extend(data2)
  return data

def download_and_process_alldata(remove_notfinal_files=True, overwrite=False):
  datasets = trainingsets + valsets
  for datasetname in datasets:
    print(f"{datasetname}")
    get_data(datasetname,remove=remove_notfinal_files,overwrite=overwrite)
    convert_data(datasetname,remove=remove_notfinal_files,overwrite=overwrite)
  sentencepieces(overwrite=overwrite)
  pickled_data(overwrite=overwrite)

hashes = {
  "train-other-500": "d1a0fd59409feb2c614ce4d30c387708",
  "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
  "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
  "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
}

trainingsets = [ # all three train together need about 108GB space if remove_notfinal_files=True
  "train-other-500",
  "train-clean-360",
  "train-clean-100",
]
valsets = [
  "dev-clean",
]
# trainingsets = valsets

if __name__=="__main__":
  download_and_process_alldata(remove_notfinal_files=True)

@dali.pipeline_def
def data_pipeline(files, eval=False, shuffle=True):
  sample_rate = 16000
  if eval:
    speed_perturbation = 1.0
  else:
    speed_perturbation = [0.85,1.15]
  SILENCE_THRESHOLD = -60
  dither = 0.00001
  nfft = 512
  window_size = 0.02
  window_stride = 0.01
  spect_wind_len = sample_rate*window_size
  spect_wind_step = sample_rate*window_stride
  nfilter = 80

  audio, label = dali.fn.readers.file(
    files=files,
    file_root="/datasets/LibriSpeech",
    name="Reader",
    pad_last_batch=True,
    shuffle_after_epoch=shuffle,
  )

  resample = speed_perturbation if isinstance(speed_perturbation,float) else dali.fn.random.uniform(range=speed_perturbation)

  audio, _ = dali.fn.decoders.audio(audio, downmix=True, sample_rate=sample_rate*resample)
  begin, length = dali.fn.nonsilent_region(audio, cutoff_db=SILENCE_THRESHOLD)

  # audio = audio.gpu()

  audio = dali.fn.slice(
    audio,
    begin,
    length,
    normalized_anchor=False,
    normalized_shape=False,
    axes=[0],
  )

  distribution = dali.fn.random.normal(device=audio.device)
  audio = audio + distribution * dither
  audio = dali.fn.preemphasis_filter(audio)

  audio = dali.fn.spectrogram(
    audio,
    nfft=nfft,
    window_length=spect_wind_len,
    window_step=spect_wind_step,
  )

  audio = dali.fn.mel_filter_bank(
    audio,
    sample_rate=sample_rate,
    nfilter=nfilter,
  )

  audio = dali.fn.to_decibels(
    audio,
    multiplier=math.log(10),
    reference=1.0,
    cutoff_db=math.log(1e-20),
  )

  audio_len = dali.fn.shapes(audio)
  audio = dali.fn.normalize(audio, axes=[1])
  audio = dali.fn.pad(audio)

  return audio, label, audio_len

torch_context_initialized = False
def use_torch_context(gpus):
  # for dali to work at the same time as tinygrad
  global torch_context_initialized
  if not torch_context_initialized:
    import torch
    from tinygrad.helpers import init_c_var
    from tinygrad.runtime.ops_cuda import check as check2

    torch.cuda.init()
    for gpu in gpus:
      device_id = int(gpu.split(":")[1]) if ":" in gpu else 0
      check2(cuda.cuDeviceGet(ctypes.byref(cu_device := cuda.CUdevice()), device_id))
      context = init_c_var(cuda.CUcontext(), lambda x: check2(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(x), cu_device)))
      d = Device[gpu]
      check2(cuda.cuCtxDestroy_v2(d.context))
      d.context = context
    torch_context_initialized = True

class DataLoader:
  def __init__(self, batch_size, gpus=None, eval=False, prefetch_queue_depth=8, shuffle=True, seed=0):
    use_torch_context(gpus)
    if seed is not None:
      torch.manual_seed(seed)
    max_duration = None if eval else 16.7
    filedata = get_datalist(eval)
    self.filedata = filedata
    self.active_data = [el for el in filedata if el["original_duration"]<=max_duration] if max_duration is not None else filedata
    if shuffle:
      if seed is not None:
        random.seed(seed)
      random.shuffle(self.active_data)
    self.files = [f"{el['fname']}" for el in self.active_data]
    self.transcripts = [el["tokenized_transcript"] for el in self.active_data]
    self.eval = eval
    self.batch_size = batch_size

    self.pipe = data_pipeline(self.files,shuffle=shuffle,eval=self.eval,batch_size=batch_size, num_threads=8, device_id=0, prefetch_queue_depth=prefetch_queue_depth,seed=seed)
    self.iter = nvidia.dali.plugin.pytorch.DALIGenericIterator(pipelines=[self.pipe],output_map=["audio","label","audio_len"],reader_name="Reader",
                                                               last_batch_policy = nvidia.dali.plugin.base_iterator.LastBatchPolicy.DROP if not eval else
                                                               nvidia.dali.plugin.base_iterator.LastBatchPolicy.PARTIAL)
    self.freq_masks=2
    self.min_freq=0
    self.max_freq=20
    self.time_masks=10
    self.min_time=0
    self.max_time=0.03
    self.stacking=3
    self.subsampling=3
    self.pad_align_time = 1
    self.pad_align_freq = 32
    self.padT = 642 if not eval else 1092
    self.padU = 125 if not eval else 188

  def __iter__(self):
    return self

  def __next__(self):
    res = next(self.iter)[0]
    audio, label, audio_len = res["audio"].cuda(), res["label"], res["audio_len"].cuda()
    with torch.no_grad():
      audio_len = audio_len[:, 1]
      audio, audio_len = self.preproc(audio,audio_len)
      audio, audio_len = audio.permute(2, 0, 1), audio_len
      txt = [self.transcripts[x] for x in label]
      txt_len = [len(el) for el in txt]
      txt = torch.nn.utils.rnn.pad_sequence([torch.tensor(txt,dtype=torch.int32) for txt in txt], batch_first=True)
      txt_len = torch.tensor(txt_len, dtype=torch.int32)
      audio = torch.nn.functional.pad(audio,(0,0,0,0,0,self.padT-audio.shape[0]))
      txt = torch.nn.functional.pad(txt,(0,self.padU-txt.shape[1]))
    return audio, audio_len, txt, txt_len

  def __del__(self):
    torch.cuda.empty_cache()

  def preproc(self, audio, audio_len):
    with torch.no_grad():
      if not self.eval:
        audio, audio_len = self.specaug(audio,audio_len)
      audio, audio_len = self.stack_subsample(audio,audio_len)
      audio, audio_len = self.fillpad(audio,audio_len)
      audio, audio_len = self.padalign(audio,audio_len)
    return audio, audio_len

  def specaug(self,x,x_lens):
    b, h, w = x.shape
    xlen = x_lens.view(-1, 1, 1)

    time_shape   = (torch.rand([b, self.time_masks, 1], device=x.device) * (xlen * self.max_time + 1)).int()
    time_anchors = (torch.rand([b, self.time_masks, 1], device=x.device) * (w - time_shape + 1)).int()
    time_idx     = torch.linspace(0, w-1, w, dtype=int, device=x.device)
    time_mask   = (
        (time_idx >= time_anchors) * (time_idx <= time_anchors + time_shape)
    ).any(dim=1)

    freq_shape   = torch.randint(self.min_freq, self.max_freq + 1, [b, self.freq_masks, 1], device=x.device)
    freq_anchors = (torch.rand([b, self.freq_masks, 1], device=x.device) * (h - freq_shape)).round().int()
    freq_idx     = torch.linspace(0, h-1, h, dtype=int, device=x.device)
    freq_mask   = (
        (freq_idx >= freq_anchors) * (freq_idx <= freq_anchors + freq_shape)
    ).any(dim=1)

    return x.masked_fill(time_mask.view(b,1,-1) + freq_mask.view(b,-1,1), 0), x_lens

  def stack_subsample(self, x, x_lens):
    x = x.transpose(1, 2)
    T = x.size(1)
    padded = torch.nn.functional.pad(x, (0, 0, 0, (self.stacking - (T % self.stacking)) % self.stacking))
    B, T, H = padded.size()
    x = padded.reshape(B, T // self.stacking, -1)
    x = x.transpose(1, 2)
    x_lens = torch.div(x_lens.int() + self.stacking - 1,
                       self.stacking, rounding_mode='trunc')
    return x, x_lens

  def fillpad(self, x, x_lens):
    max_len = x.size(-1)
    mask = torch.arange(max_len, dtype=x_lens.dtype, device=x.device)
    mask = mask.expand(x.size(0), max_len) >= x_lens.unsqueeze(1)
    x = x.masked_fill(mask.unsqueeze(1), 0)

    return x, x_lens

  def padalign(self, x, x_lens):
    pad_time = 0
    pad_freq = 0

    if self.pad_align_time > 0:
      pad_amt = x.size(2) % self.pad_align_time
      pad_time = (self.pad_align_time - pad_amt if pad_amt > 0 else 0)

    if self.pad_align_freq > 0:
      pad_amt = x.size(1) % self.pad_align_freq
      pad_freq = (self.pad_align_freq - pad_amt if pad_amt > 0 else 0)

    x = torch.nn.functional.pad(x, (0, pad_time, 0, pad_freq))
    return x, x_lens
