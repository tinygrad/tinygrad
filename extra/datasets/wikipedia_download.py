import os, hashlib
from pathlib import Path
import tarfile
import gdown
from tqdm import tqdm
from tinygrad.helpers import getenv

def gdrive_download(url:str, path:str): 
  if not os.path.exists(path): gdown.download(url, path)

def wikipedia_uncompress_and_extract(file, path, small=False):
  if not os.path.exists(os.path.join(path, "results4")):
    print("Uncompressing and extracting file...")
    with tarfile.open(file, 'r:gz') as tar:
      tar.extractall(path=path)
      os.remove(file)
      if small: # Show progressbar only for big files
        for member in tar.getmembers(): tar.extract(path=path, member=member)
      else:
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())): tar.extract(path=path, member=member)

def verify_checksum(folder_path:str, checksum_path:str):
  with open(checksum_path, 'r') as f:
    for line in f:
      expected_checksum, folder_name = line.split()
      file_path = os.path.join(folder_path, folder_name[2:]) # remove './' from the start of the folder name
      hasher = hashlib.md5()
      with open(file_path, 'rb') as f:
        for buf in iter(lambda: f.read(4096), b''): hasher.update(buf)
      if hasher.hexdigest() != expected_checksum:
        print(f"Checksum does not match for file: {folder_path}")
        return False
  return True

def download_wikipedia(path:str):
  # TODO: Download tf2 checkpoints, eval?
  os.makedirs(path, exist_ok=True)
  gdrive_download("https://drive.google.com/uc?id=1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW", os.path.join(path, "bert_config.json"))
  gdrive_download("https://drive.google.com/uc?id=1USK108J6hMM_d27xCHi738qBL8_BT1u1", os.path.join(path, "vocab.txt"))
  if getenv("TRAIN_DOWNLOAD", 0):
    gdrive_download("https://drive.google.com/uc?id=1tmMgLwoBvbEJEHXh77sqrXYw5RpqT8R_", os.path.join(path, "bert_reference_results_text_md5.txt"))
    gdrive_download("https://drive.google.com/uc?id=14xV2OUGSQDG_yDBrmbSdcDC-QGeqpfs_", os.path.join(path, "results_text.tar.gz"))
    wikipedia_uncompress_and_extract(os.path.join(path, "results_text.tar.gz"), path)
    if getenv("VERIFY", 0):
      if not verify_checksum(os.path.join(path, "results4"), os.path.join(path, "bert_reference_results_text_md5.txt")):
        raise Exception("Checksum does not match downloaded dataset.")
      else:
        print(f"Checksums match.")

if __name__ == "__main__": 
  download_wikipedia(os.path.join(Path(__file__).parent / "wiki"))