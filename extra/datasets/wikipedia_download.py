# pip install nltk, gdown, wikiextractor (currently does not work with python 3.11)
# Mostly copied from https://github.com/mlcommons/training/blob/master/language_model/tensorflow/bert/cleanup_scripts/
import os, bz2, hashlib, subprocess
from pathlib import Path
from typing import List
import gdown
from tqdm import tqdm
from tinygrad.helpers import getenv

def verify_checksum(file_path:str, checksum_path:str):
  with open(checksum_path, 'r') as f:
    expected_checksum = f.read().split()[0] # only first string is checksum
  hasher = hashlib.md5()
  with open(file_path, 'rb') as f:
    for buf in iter(lambda: f.read(4096), b''): hasher.update(buf)
  return hasher.hexdigest() == expected_checksum

def gdrive_download(url:str, path:str): 
  if not os.path.exists(path): gdown.download(url, path)

def wikipedia_extract(file, path):
  chunk_size = 1024 * 1024 * 1024  # 1GiB
  if not os.path.exists(path):
    print("Uncompressing train file...")
    with bz2.BZ2File(file, 'rb') as f_in:
      with open(path, 'wb') as f_out:
        for data in tqdm(iter(lambda : f_in.read(chunk_size), b'')):
          f_out.write(data)
    os.remove(file)
  subprocess.run(["python", "-m", "wikiextractor.WikiExtractor", path, "-o", os.path.join(os.path.dirname(path), "wiki"), "--processes", str(getenv("NUM_WORKERS", os.cpu_count()))], check=True) # extract xml to .txt files
  os.remove(path)

def download_uncompress_wikipedia(path:str):
  # TODO: Download tf2 checkpoints, eval?
  os.makedirs(path, exist_ok=True)
  gdrive_download("https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW", os.path.join(path, "bert_config.json"))
  gdrive_download("https://drive.google.com/uc?id=1USK108J6hMM_d27xCHi738qBL8_BT1u1", os.path.join(path, "vocab.txt"))
  if getenv("TRAIN_DOWNLOAD", 0):
    gdrive_download("https://drive.google.com/uc?id=14_A6gQ0NJ7Pay1X0xFq9rCKUuFJcKLF-", os.path.join(path, "enwiki-20200101-pages-articles-multistream.xml.bz2.md5sum"))
    gdrive_download("https://drive.google.com/uc?id=18K1rrNJ_0lSR9bsLaoP3PkQeSFO-9LE7", os.path.join(path, "enwiki-20200101-pages-articles-multistream.xml.bz2"))
    if getenv("VERIFY", 0):
      if not verify_checksum(os.path.join(path, "enwiki-20200101-pages-articles-multistream.xml.bz2"), os.path.join(path, "enwiki-20200101-pages-articles-multistream.xml.bz2.md5sum")):
        raise Exception("Checksum does not match downloaded dataset.")
      else: 
        print(f"Checksum matches.")
    wikipedia_extract(os.path.join(path, "enwiki-20200101-pages-articles-multistream.xml.bz2"), os.path.join(path, "enwiki-20200101-pages-articles-multistream.xml"))


if __name__ == "__main__": 
  download_uncompress_wikipedia(os.path.join(Path(__file__).parent / "wiki"))