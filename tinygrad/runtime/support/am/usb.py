
# import argparse
# import os
# import struct
# import sys
# import time
# from pathlib import Path

# try:
#   import sgio
# except ModuleNotFoundError:
#   sys.stderr.write("Error: Failed to import \"sgio\". Please install \"cython-sgio\", then try running this script again.\n")
#   sys.exit(1)

# class USBIface:
#   def __init__(self):
#     self.dev = Asm236x(device)

#   def read(self, start_addr, read_len, stride=255):
#     data = bytearray(read_len)

#     for i in range(0, read_len, stride):
#       remaining = read_len - i
#       buf_len = min(stride, remaining)

#       cdb = struct.pack('>BBBHB', 0xe4, buf_len, 0x00, start_addr + i, 0x00)

#       buf = bytearray(buf_len)
#       ret = sgio.execute(self._file, cdb, None, buf)
#       assert ret == 0

#       data[i:i+buf_len] = buf

#     return bytes(data)

#   def write(self, start_addr, data):
#       for offset, value in enumerate(data):
#         cdb = struct.pack('>BBBHB', 0xe5, value, 0x00, start_addr + offset, 0x00)
#         ret = sgio.execute(self._file, cdb, None, None)
#         assert ret == 0
