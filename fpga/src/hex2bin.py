#!/usr/bin/env python3
import binascii

out = []
for d in open("firmware.hex").read().strip().split("\n"):
  out.append(binascii.unhexlify(d)[::-1])

with open("firmware.bin", "wb") as f:
  f.write(b''.join(out))
  
