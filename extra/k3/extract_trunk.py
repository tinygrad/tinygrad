#!/usr/bin/env python3
# the raw trunk gguf copied off store disk 0 (a raid0 of 4 drives, 1 MiB stripes) through the kernel nvme driver, read only. then
# trunk_names.py makes the renamed gguf the engine loads (its size must match the trunk snapshot's index for the snapshot to be used).
# python3 extract_trunk.py out.gguf /dev/nvmeAn1 /dev/nvmeBn1 /dev/nvmeCn1 /dev/nvmeDn1 (disk 0's drives in stripe order)
import json, os, sys

M = 1 << 20
fds = [os.open(d, os.O_RDONLY) for d in sys.argv[2:6]]

def read(off:int, size:int) -> bytes:
  out = bytearray()
  while size:
    s, n = off // M, min(size, M - off % M)
    out += os.pread(fds[s % len(fds)], n, (s // len(fds)) * M + off % M)
    off, size = off + n, size - n
  return bytes(out)

raw = read(0, 16 * M)
m = json.loads(raw[:raw.index(b"\0")])
with open(sys.argv[1], "wb") as f:
  for off in range(0, m["trunk_bytes"], 64 * M): f.write(read(m["trunk_off"] + off, min(64 * M, m["trunk_bytes"] - off)))
print(f"{m['trunk_name']}: {os.path.getsize(sys.argv[1])} bytes")
