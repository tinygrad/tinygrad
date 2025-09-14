#!/usr/bin/env python3
import sys, os, zlib, struct, hashlib
from hexdump import hexdump
from tinygrad.helpers import DEBUG, getenv, fetch
from tinygrad.runtime.support.usb import USB3

def patch(input_filepath, file_hash, patches):
  with open(input_filepath, 'rb') as infile: data = bytearray(infile.read())

  if_hash = hashlib.md5(data).hexdigest()
  if if_hash != file_hash:
    raise ValueError(f"File hash mismatch: expected {file_hash}, got {if_hash}")

  # find 90 b2 51 in data
  for i in range(len(data) - 2):
    if data[i] == 0xb2 and data[i+1] == 0x51:
      print(f"Found {i:x} {[hex(int(x)) for x in data[i:i+3]]}")
  # exit(0)

  for offset, expected_bytes, new_bytes in patches:
    if len(expected_bytes) != len(new_bytes):
      raise ValueError(f"Expected bytes and new bytes must be the same length {len(expected_bytes)} {len(new_bytes)}")

    if offset + len(new_bytes) > len(data): return False
    current_bytes = data[offset:offset + len(expected_bytes)]
    assert bytes(current_bytes) == expected_bytes, f"Expected {expected_bytes} at offset {offset:x}, but got {current_bytes}"
    data[offset:offset + len(new_bytes)] = new_bytes

  checksum = sum(data[4:-6]) & 0xff
  crc32 = zlib.crc32(data[4:-6]).to_bytes(4, 'little')
  data[-5] = checksum
  data[-4] = crc32[0]
  data[-3] = crc32[1]
  data[-2] = crc32[2]
  data[-1] = crc32[3]
  return data

path = os.path.dirname(os.path.abspath(__file__))
file_hash = "5284e618d96ef804c06f47f3b73656b7"
file_path = os.path.join(path, "Software/AS_USB4_240417_85_00_00.bin")

if not os.path.exists(file_path):
  url = "https://web.archive.org/web/20250430124720/https://www.station-drivers.com/index.php/en/component/remository/func-download/6341/chk,3ef8b04704a18eb2fc57ff60382379ad/no_html,1/lang,en-gb/"
  os.system(f'curl -o "{path}/fw.zip" "{url}"')
  os.system(f'unzip -o "{path}/fw.zip" "Software/AS_USB4_240417_85_00_00.bin" -d "{path}"')

patches = [(0x2a0d + 1 + 4, b'\x0a', b'\x05')]
# patches += [(0x58a1 + 4, b'\x08\x06\x62', b'\xff\xff\xff')]
# patches += [(0x590e + 4, b'\x08\x06\x62', b'\xff\xff\xff')]
patches += [(0x40e1 + 4, b'\x90\x06\xe6\x04\xf0\x78\x0d\xe6\xfe\x24\x71\x12\x1b\x0b\x60\x0b\x74\x08\x2e\x12',
                         b'\x90\x90\x07\x74\x04\xf0\xa3\x74\x00\xf0\x74\x01\xa6\xe0\x12\x60\x00\x02\x41\x7a')] # \x12\x60\x00

asm = bytes([
    0x90, 0xF0, 0x00,  # 6000: MOV  DPTR,#0xF000   ; source = 0xF000
    0x75, 0xA0, 0x80,  # 6003: MOV  P2,#0x80       ; dest-hi = 0x80 (→ 0x8000)
    0x78, 0x00,        # 6006: MOV  R0,#0x00       ; dest-lo = 0x00
    0x7E, 0x04,        # 6008: MOV  R6,#0x04       ; 4 × 256-byte pages
    0x7F, 0x00,        # 600A: MOV  R7,#0x00       ; inner-loop counter (256)
                       # ----- inner loop (256 bytes) -----
    0xE0,              # 600C: MOVX A,@DPTR        ; read  byte  [DPTR] 2
    0xA3,              # 600D: INC  DPTR           ; ++source 1
    0xF2,              # 600E: MOVX @R0,A          ; write byte  [(P2:R0)] 2
    0x08,              # 600F: INC  R0             ; ++dest-lo 1
    0xDF, 0xFA,        # 6010: DJNZ R7,600C        ; 256-byte loop 2
                       # ----- end inner loop -----
    0x78, 0x00,        # 6012: MOV  R0,#0x00       ; reset dest-lo
    0x05, 0xA0,        # 6014: INC  P2             ; next 256-byte page
    0xDE, 0xF2,        # 6016: DJNZ R6,600A        ; outer loop (16 pages)
    0x22               # 6018: RET
])
asm2 = bytes([
    0x90, 0x3F, 0x00,  # 6000: MOV  DPTR,#0x3F00   ; scratch holds DPH
    0xE0,              # 6003: MOVX A,@DPTR        ; A ← saved DPH
    0xF5, 0x83,        # 6004: MOV  DPH,A
    0x75, 0x82, 0x00,  # 6006: MOV  DPL,#0x00

    0x75, 0xA0, 0x80,  # 6009: MOV  P2,#0x80       ; dest-hi = 0x8000
    0x78, 0x00,        # 600C: MOV  R0,#0x00
    0x7E, 0x04,        # 600E: MOV  R6,#0x04       ; 4 × 256-byte pages
    0x7F, 0x00,        # 6010: MOV  R7,#0x00       ; inner loop (256)

    # ----- inner loop -----
    0xE0,              # 6012: MOVX A,@DPTR
    0xA3,              # 6013: INC  DPTR
    0xF2,              # 6014: MOVX @R0,A
    0x08,              # 6015: INC  R0
    0xDF, 0xFA,        # 6016: DJNZ R7,6012
    # ----------------------

    0x78, 0x00,        # 6018: MOV  R0,#0x00
    0x05, 0xA0,        # 601A: INC  P2
    0xDE, 0xF2,        # 601C: DJNZ R6,6010

    # ----- save next source page -----
    0xE5, 0x83,        # 601E: MOV  A,DPH          ; A = new high byte
    0x60, 0x05,        # 6020: JZ   6027           ; if 0, wrap to F0
    0x90, 0x3F, 0x00,  # 6022: MOV  DPTR,#0x3F00
    0xF0,              # 6025: MOVX @DPTR,A        ; save A (F4/F8/FC)
    0x22,              # 6026: RET

    0x74, 0xF0,        # 6027: MOV  A,#0xF0        ; wrap value
    0x90, 0x3F, 0x00,  # 6029: MOV  DPTR,#0x3F00
    0xF0,              # 602C: MOVX @DPTR,A
    0x22               # 602D: RET
])
patches += [(0x6000 + 4, b'\x00' * len(asm2), asm2)]

next_traphandler = 0
next_iftrap = 0
def add_traphandler(addr, sec):
  global next_traphandler, patches

  trap_addr = 0x6200 + next_traphandler * 0x40
  assert trap_addr < 0x6c00

  return_addr = addr + len(sec)
  cntr_addr = 0x3000 + next_traphandler
  patches += [
    (addr + 4, sec, b'\x02' + trap_addr.to_bytes(2, 'big') + b'\x22'*(len(sec)-3)),
    (trap_addr + 4, b'\x00' * (21 + len(sec)),
      b'\xc0\xe0\xc0\x82\xc0\x83\x90' + cntr_addr.to_bytes(2, 'big') + b'\xe0\x04\xf0\xd0\x83\xd0\x82\xd0\xe0' + sec + b'\x02' + return_addr.to_bytes(2, 'big')),
  ]
  next_traphandler += 1

def add_if(addr, sec, true_addr2):
  global next_iftrap, patches

  trap_addr = 0x6c00 + next_iftrap * 0x40
  assert trap_addr < 0x7000

  return_addr = addr + len(sec)
  cntr_addr = 0x3800 + next_iftrap
  jump_over = 7 + len(sec) + 1 + 2
  patches += [
    (addr + 4, sec, b'\x02' + trap_addr.to_bytes(2, 'big') + b'\x22'*(len(sec)-3)),
    (trap_addr + 4, b'\x00' * (33 + len(sec) * 2),
      b'\xc0\xe0\xc0\x82\xc0\x83\x90' + cntr_addr.to_bytes(2, 'big') + b'\xe0\x30\xe0' + jump_over.to_bytes(1, 'big') +
      b'\xf0\xd0\x83\xd0\x82\xd0\xe0' + sec + b'\x02' + true_addr2.to_bytes(2, 'big') +
      b'\xf0\xd0\x83\xd0\x82\xd0\xe0' + sec + b'\x02' + return_addr.to_bytes(2, 'big'))
  ]
  next_iftrap += 1

def add_if_ret(addr, sec):
  global next_iftrap, patches

  trap_addr = 0x6800 + next_iftrap * 0x40
  return_addr = addr + len(sec)
  cntr_addr = 0x3800 + next_iftrap
  jump_over = 7 + 1
  patches += [
    (addr + 4, sec, b'\x02' + trap_addr.to_bytes(2, 'big') + b'\x22'*(len(sec)-3)),
    (trap_addr + 4, b'\x00' * (31 + len(sec)),
      b'\xc0\xe0\xc0\x82\xc0\x83\x90' + cntr_addr.to_bytes(2, 'big') + b'\xe0\x30\xe0' + jump_over.to_bytes(1, 'big') +
      b'\xf0\xd0\x83\xd0\x82\xd0\xe0' + b'\x22' +
      b'\xf0\xd0\x83\xd0\x82\xd0\xe0' + sec + b'\x02' + return_addr.to_bytes(2, 'big'))
  ]
  next_iftrap += 1


traps = [
  # (0x10e7, b'\x90\xce\xf3', "LAB_CODE_10e0"),
  # (0x110a, b'\x12\x3a\xdb', "call FUN_CODE_3adb"),
  # (0x2608, b'\x12\x16\x87', "in FUN_CODE_2608"),
  # (0x2641, b'\x90\xc8\xd6', "call FUN_CODE_2608 in rp"),
  # (0x1114, b'\x75\x37\x00', "call (DAT_EXTMEM_c802 >> 2 & 1) != 0"),

  # (0x1148, b'\x12\x3e\x81', "call FUN_CODE_3e81"),
  # (0x1152, b'\x12\x48\x8f', "call FUN_CODE_488f"),
  # (0x1172, b'\x12\x47\x84', "call FUN_CODE_4784"),

  # (0x1045, b'\x12\x11\x96', "scsi call 1 in main loop"),
  # (0x112e, b'\x12\x11\x96', "scsi call 2 in main loop"),

  # (0x0e82, b'\x90\x91\x01', "in (DAT_EXTMEM_c802 & 1) == 1 (not fast path?)"),
  # (0x0e8c, b'\x90\x90\x00', "in (DAT_EXTMEM_9101 >> 5 & 1) == 1"),
  # (0x0f36, b'\x90\x93\x01', "in (DAT_EXTMEM_9101 >> 3 & 1) != 0"),
  # (0x0fc0, b'\x12\x03\x3b', "in (DAT_EXTMEM_9101 >> 1 & 1) != 0"),
  # (0x0fcd, b'\x90\x90\x93', "in (DAT_EXTMEM_9101 >> 2 & 1) == 1"),
  # (0x100f, b'\x90\x91\x01', "in (DAT_EXTMEM_9101 >> 2 & 1) == 0"),
  # (0x0fd7, b'\x12\x32\xa5', "call to FUN_CODE_32a5"),
  # (0x0fe4, b'\x12\x4d\x44', "call to FUN_CODE_4d44"),
  # (0x0ff4, b'\x12\x54\x55', "call to FUN_CODE_5455"),

  # (0x3ede, b'\x90\xce\xf3', "precall FUN_CODE_2608 in FUN_CODE_3e81"),
  # (0x3eb9, b'\x12\x31\x79', "call to FUN_CODE_3179 in FUN_CODE_3e81"),
  # (0x3f36, b'\x12\x45\xd0', "call to FUN_CODE_45d0 in FUN_CODE_3e81"),
  # (0x3ee9, b'\x12\x03\x95', "call to FUN_CODE_0395 in FUN_CODE_3e81"),

  # (0x4013, b'\x12\x32\x98', "in FUN_CODE_4013"),
  # (0x4d78, b'\x12\x31\x2a', "FUN_CODE_4d44: set_af2_to_1_and_9006|=1"),

  # (0x4082, b'\x90\x0a\x7d', "choose path DAT_EXTMEM_0a7d = 0x80"),
  # (0x4089, b'\x90\x0a\x7d', "choose path DAT_EXTMEM_0a7d = 0xf0"),
  # (0x407b, b'\x90\x0a\x7d', "choose path DAT_EXTMEM_0a7d = 0xe8"),

  # Fill scsi resp
  # (0x1aa4, b'\x12\x02\x06', "fill_scsi_resp1"),
  # (0x2777, b'\x12\x02\x06', "fill_scsi_resp2"),
  # (0x2bc7, b'\x12\x02\x06', "fill_scsi_resp3"),
  # (0x33f2, b'\x12\x02\x06', "fill_scsi_resp4"),
  # (0x3f12, b'\x12\x02\x06', "fill_scsi_resp5"),
  # (0x416e, b'\x12\x02\x06', "fill_scsi_resp6"),
  # (0x47e5, b'\x12\x02\x06', "fill_scsi_resp7"),
  # (0x4d6b, b'\x12\x02\x06', "fill_scsi_resp8"),
  # (0x4165, b'\x02\x02\x06', "fill_scsi_resp9"),
  # (0x425c, b'\x02\x02\x06', "fill_scsi_resp10"),
]

if __name__ == "__main__":
  for addr, sec, name in traps:
    print(f"Adding {name} at {addr:x}")
    add_traphandler(addr, sec)
  # add_if(0x4d54, b'\x90\x00\x01', 0x4d78)
  # add_if(0x4d8e, b'\x12\x31\x33', 0x4d68)

  # add_if(0x1141, b'\x90\xc5\x20', 0x1148) # force jump into FUN_CODE_3e81
  # add_if(0x0e78, b'\x90\xc8\x02', 0x10e0) # fast path
  # add_if(0x3eaf, b'\xf5\x83\xe0', 0x3eb5) # skip while in FUN_CODE_3e81
  # add_if(0x3e9c, b'\xf5\x83\xe0', 0x3ea6) # skip if in FUN_CODE_3e81

  # add_if(0x10e0, b'\x90\xc8\x06', 0x110d)
  # add_if(0x0fcd, b'\x90\x90\x93', 0x113a)

  patched_fw = patch(file_path, file_hash, patches)
  # with open('/Users/nimelehin/Develop/ML/tinygrad/demo.fw', 'wb') as outfile: outfile.write(patched_fw)

  # exit(0)

  vendor, device = [int(x, base=16) for x in getenv("USBDEV", "ADD1:0001").split(":")]
  try: dev = USB3(vendor, device, 0x81, 0x83, 0x02, 0x04)
  except RuntimeError as e:
    raise RuntimeError(f'{e}. You can set USBDEV environment variable to your device\'s vendor and device ID (e.g., USBDEV="174C:2464")') from e

  config1 = bytes([
    0xFF, 0xFF, 0xFF, 0xFF, 0x41, 0x41, 0x41, 0x41, 0x42, 0x42, 0x42, 0x42, 0x30, 0x30, 0x36, 0x30,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x74, 0x69, 0x6E, 0x79, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x74, 0x69, 0x6E, 0x79,
    0xFF, 0xFF, 0xFF, 0xFF, 0x55, 0x53, 0x42, 0x20, 0x33, 0x2E, 0x32, 0x20, 0x50, 0x43, 0x49, 0x65,
    0x20, 0x54, 0x69, 0x6E, 0x79, 0x45, 0x6E, 0x63, 0x6C, 0x6F, 0x73, 0x75, 0x72, 0x65, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0x54, 0x69, 0x6E, 0x79, 0x45, 0x6E, 0x63, 0x6C, 0x6F, 0x73, 0x75, 0x72,
    0x65, 0xFF, 0xFF, 0xFF, 0xD1, 0xAD, 0x01, 0x00, 0x00, 0x01, 0xCF, 0xFF, 0x02, 0xFF, 0x5A, 0x94])

  config2 = bytes([
    0xFF, 0xFF, 0xFF, 0xFF, 0x47, 0x6F, 0x70, 0x6F, 0x64, 0x20, 0x47, 0x72, 0x6F, 0x75, 0x70, 0x20,
    0x4C, 0x69, 0x6D, 0x69, 0x74, 0x65, 0x64, 0x2E, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x55, 0x53, 0x42, 0x34,
    0x20, 0x4E, 0x56, 0x4D, 0x65, 0x20, 0x53, 0x53, 0x44, 0x20, 0x50, 0x72, 0x6F, 0x20, 0x45, 0x6E,
    0x63, 0x6C, 0x6F, 0x73, 0x75, 0x72, 0x65, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0x8C, 0xBF, 0xFF, 0x97, 0xC1, 0xF3, 0xFF, 0xFF, 0x01, 0x2D, 0x66, 0xD6,
    0x66, 0x06, 0x00, 0xC0, 0x87, 0x01, 0x5A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xCA, 0x01, 0x66, 0xD6,
    0xE3, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x01, 0x00, 0xA5, 0x67])

  part1 = patched_fw[:0xff00]
  part2 = patched_fw[0xff00:]

  # config patch
  cdb = struct.pack('>BBB12x', 0xe1, 0x50, 0x0)
  dev.send_batch(cdbs=[cdb], odata=[config1])

  cdb = struct.pack('>BBB12x', 0xe1, 0x50, 0x1)
  dev.send_batch(cdbs=[cdb], odata=[config2])

  cdb = struct.pack('>BBI', 0xe3, 0x50, len(part1))
  dev.send_batch(cdbs=[cdb], odata=[part1])

  cdb = struct.pack('>BBI', 0xe3, 0xd0, len(part2))
  dev.send_batch(cdbs=[cdb], odata=[part2])

  cdb = struct.pack('>BB13x', 0xe8, 0x51)
  dev.send_batch(cdbs=[cdb])

  print("done, you can disconnect the controller!")
