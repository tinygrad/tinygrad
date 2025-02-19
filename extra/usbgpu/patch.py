#!/usr/bin/env python3
import sys

def patch(input_filepath, output_filepath, patches):
  with open(input_filepath, 'rb') as infile: data = bytearray(infile.read())

  for offset, expected_bytes, new_bytes in patches:
    # if len(expected_bytes) != len(new_bytes):
    #   raise ValueError("Expected bytes and new bytes must be the same length")

    if offset + len(new_bytes) > len(data): return False
    current_bytes = data[offset:offset + len(expected_bytes)]
    assert bytes(current_bytes) == expected_bytes
    data[offset:offset + len(new_bytes)] = new_bytes

  with open(output_filepath, 'wb') as outfile:
    outfile.write(data)

  return True

# function verifier, old asm
patched_pci_bar_write_old = bytes([
    0x90, 0x00, 0x06,  # MOV DPTR,#0x6
    0xE0,              # MOVX A,@DPTR
    0xFB,              # MOV R3,A
    0xA3,              # INC DPTR
    0xE0,              # MOVX A,@DPTR
    0xFA,              # MOV R2,A
    0xA3,              # INC DPTR
    0xE0,              # MOVX A,@DPTR
    0xF9,              # MOV R1,A
    0x90, 0x00, 0x10,  # MOV DPTR,#0x10
    0x12, 0x71, 0x3F,  # LCALL mb_read_from_addrspace
    0x54, 0x01,        # ANL A,#0x1
    0xF5, 0x4C,        # MOV DAT_INTMEM_4c,A
    0x90, 0x90, 0x00,  # MOV DPTR,#0x9000
    0xE0,              # MOVX A,@DPTR
    0x30])

patched_pci_bar_write = bytes([
    0x90, 0x00, 0x07,  # MOV DPTR,#0x7
    0xE0,              # MOVX A,@DPTR
    0xFA,              # MOV R2,A
    0xA3,              # INC DPTR
    0xE0,              # MOVX A,@DPTR
    0x24, 0x10,        # ADD A,#0x10
    0xF9,              # MOV R1,A
    0xE4,              # CLR A
    0x3A,              # ADDC A,R2
    0xF5, 0x83,        # MOV DPH,A
    0x89, 0x82,        # MOV DPL,R1
    0x12, 0x74, 0xD7,  # LCALL copy4_from_dptr_into_r4_r7
    0x12, 0x75, 0x09,  # LCALL copy4_into_b220
    0x90, 0x00, 0x07,  # MOV DPTR,#0x7
    0xE0,              # MOVX A,@DPTR
    0xFA,              # MOV R2,A
    0xA3,              # INC DPTR
    0xE0,              # MOVX A,@DPTR
    0x24, 0x14,        # ADD A,#0x14
    0xF9,              # MOV R1,A
    0xE4,              # CLR A
    0x3A,              # ADDC A,R2
    0xF5, 0x83,        # MOV DPH,A
    0x89, 0x82,        # MOV DPL,R1
    0x12, 0x74, 0xD7,  # LCALL copy4_from_dptr_into_r4_r7
    0x90, 0x05, 0xF9,  # MOV DPTR,#0x5f9
    0x12, 0x72, 0x8D,  # LCALL copy_4_into_ptr
    0x12, 0xA3, 0x99,  # LCALL mb_tlp_mem_req
    0x12, 0xB6, 0xE3,  # LCALL SUB_CODE_b6e3
    0x22               # RET
])

patches = [
  (0x7a63, b'\xC6\x4B', b'\xB6\xB3'),
  (0x9f0c, patched_pci_bar_write_old, patched_pci_bar_write),
]

assert patch(sys.argv[1], sys.argv[2], patches) is True
