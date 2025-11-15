import pickle, binascii
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

pkt_lengths = {0x51: 3, 0xd1: 3,
               0x1: 6, 0x71: 8,
               # all 0x?9 = 8
               0x9: 8, 0x19: 8, 0x29: 8, 0x39: 8, 0x49: 8, 0x59: 8,
               # all 0x?8 = 1
               0x58: 1, 0x98: 1, 0xc8: 1, 0xe8: 1}

def pktparse(s:bytes):
  # start at 8, skip the header
  # assuming the second byte of each packet is the packet type
  ptr = 8
  while ptr < len(s):
    if s[ptr] == 0:
      #print(f"skip {ptr:2x}")
      ptr += 1
      continue
    if s[ptr] not in pkt_lengths:
      print(f"can't parse {s[ptr]:2x} @ {ptr:2x}", "  --  ", binascii.hexlify(s[ptr:ptr+0x20]).decode())
      break
    print(f"got packet {s[ptr]:2x} @ {ptr:2x}", "  --  ", binascii.hexlify(s[ptr:ptr+pkt_lengths[s[ptr]]]).decode())
    ptr += pkt_lengths[s[ptr]]


if __name__ == "__main__":
  dat_gemm_0_sqtt = parse("extra/sqtt/examples/profile_gemm_run_0.pkl")
  dat_gemm_1_sqtt = parse("extra/sqtt/examples/profile_gemm_run_1.pkl")
  dat_plus_0_sqtt = parse("extra/sqtt/examples/profile_plus_run_0.pkl")
  dat_plus_1_sqtt = parse("extra/sqtt/examples/profile_plus_run_1.pkl")

  blob_0 = dat_plus_0_sqtt[0].blob
  blob_1 = dat_plus_1_sqtt[0].blob
  blob_2 = dat_gemm_0_sqtt[0].blob
  blob_3 = dat_gemm_1_sqtt[0].blob

  hexdiff(blob_0, blob_1, 0x300)
  print("")
  hexdiff(blob_1, blob_0, 0x300)
  print("")
  hexdiff(blob_2, blob_0, 0x200)
  print("")
  hexdiff(blob_3, blob_2, 0x200)

  print("parse blob 0")
  pktparse(blob_0)
  print("parse blob 1")
  pktparse(blob_1)
  print("parse blob 2")
  pktparse(blob_2)
  print("parse blob 3")
  pktparse(blob_3)
