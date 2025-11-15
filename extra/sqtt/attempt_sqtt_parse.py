import pickle
from hexdump import hexdump
from tinygrad.helpers import colored
from extra.sqtt.roc import decode, ProfileSQTTEvent

def hexdiff(b0, b1, len=-1):
  for i, (c0, c1) in enumerate(zip(b0, b1)):
    if i == len: break
    if i % 0x10 == 0: print(f"{i:8X}: ", end="")
    print(colored(f"{c0:02X} ", "green" if c0 == c1 else "red"), end="\n" if i%0x10 == 0xf else "")

def parse(fn:str):
  dat = pickle.load(open(fn, "rb"))
  ctx = decode(dat)
  dat_sqtt = [x for x in dat if isinstance(x, ProfileSQTTEvent)]
  print(f"got {len(dat_sqtt)} SQTT events in {fn}")
  return dat_sqtt

if __name__ == "__main__":
  print("parse plus")
  dat_plus_sqtt = parse("extra/sqtt/examples/profile_plus_run_0.pkl")
  print("parse gemm")
  dat_gemm_sqtt = parse("extra/sqtt/examples/profile_plus_run_1.pkl")

  blob_0 = dat_plus_sqtt[0].blob
  blob_1 = dat_gemm_sqtt[0].blob
  hexdiff(blob_0, blob_1, 0x200)
  print("")
  hexdiff(blob_1, blob_0, 0x200)
