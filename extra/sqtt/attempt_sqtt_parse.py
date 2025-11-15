import pickle, binascii, struct
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

# fmt -> header length (bytes)
FMT_TO_LEN = {
  0x0: 2,  # class 0x10
  0x1: 8,  # class 0x40
  0x2: 8,  # class 0x40
  0x3: 4,  # class 0x20
  0x4: 2,  # class 0x10
  0x5: 6,  # class 0x30
  0x6: 2,  # class 0x10
  0x7: 2,  # class 0x10
  0x8: 2,  # class 0x10
  0x9: 2,  # class 0x10
  0xA: 2,  # class 0x10
  0xB: 8,  # class 0x40
  0xC: 6,  # class 0x30
  0xD: 4,  # class 0x20
  0xE: 8,  # class 0x40
  0xF: 6,  # class 0x30
}

def pktparse(s:bytes):
  # start at 8, skip the header
  ptr = 8
  while ptr < len(s):
    # the header is the first two bytes
    pkt_hdr = struct.unpack("H", s[ptr:ptr+2])[0]
    # this is the packet format
    pkt_fmt = pkt_hdr & 0xF
    event_id   = (pkt_hdr >> 4) & 0xFF   # matches `probably_packet_type = (uVar27 >> 4) & 0xff` etc.
    se_id      = (pkt_hdr >> 12) & 0x3   # often top bits are shader engine / instance

    raw = s[ptr:ptr + FMT_TO_LEN[pkt_fmt]]
    print(f"{ptr:#06x}  {pkt_hdr=:#06x}  {pkt_fmt=:x}  {event_id=:02x}  {se_id=}  {raw.hex(' ')}")
    ptr += len(raw)


"""
pkt_lengths = {
               #0x71: 8,
               #0xc: 4, 0xd: 0xe,
               #0x11: 3, 0x31: 3, 0x61: 3,
               #0xd: 17,
               # wrong?
               0xc: 4,
               0x31: 7,
               # section header
               0x1: 6,
               # some 0x?1 == 3
               0x51: 3, 0x61: 3, 0xd1: 3,
               # 0x71
               0x71: 8,
               # all 0x?9 = 8
               0x9: 8, 0x19: 8, 0x29: 8, 0x39: 8, 0x49: 8, 0x59: 8,
               # all 0x?8 = 1, prefixes
               0x28: 1, 0x58: 1, 0x98: 1, 0xa8: 1, 0xc8: 1, 0xe8: 1}
"""

if __name__ == "__main__":
  dat_gemm_0_sqtt = parse("extra/sqtt/examples/profile_gemm_run_0.pkl")
  dat_gemm_1_sqtt = parse("extra/sqtt/examples/profile_gemm_run_1.pkl")
  dat_plus_0_sqtt = parse("extra/sqtt/examples/profile_plus_run_0.pkl")
  dat_plus_1_sqtt = parse("extra/sqtt/examples/profile_plus_run_1.pkl")

  blob_0 = dat_plus_0_sqtt[0].blob
  blob_1 = dat_plus_1_sqtt[0].blob
  blob_2 = dat_gemm_0_sqtt[0].blob
  blob_3 = dat_gemm_1_sqtt[0].blob

  """
  hexdiff(blob_0, blob_1, 0x300)
  print("")
  hexdiff(blob_1, blob_0, 0x300)
  print("")
  hexdiff(blob_2, blob_0, 0x200)
  print("")
  hexdiff(blob_3, blob_2, 0x200)
  """
  #hexdump(blob_1)

  print("parse blob 0")
  pktparse(blob_0)
  #print("parse blob 1")
  #pktparse(blob_1)
  #hexdump(blob_2)
  #print("parse blob 2")
  #pktparse(blob_2)
  """
  print("parse blob 3")
  pktparse(blob_3)
  """
