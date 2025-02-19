#!/usr/bin/env python3
import sys

def patch(input_filepath, output_filepath, patches):
  with open(input_filepath, 'rb') as infile: data = bytearray(infile.read())

  # Apply all patches
  for offset, expected_bytes, new_bytes in patches:
    if len(expected_bytes) != len(new_bytes):
      raise ValueError("Expected bytes and new bytes must be the same length")

    if offset + len(expected_bytes) > len(data):
      return False

    current_bytes = data[offset:offset + len(expected_bytes)]
    if bytes(current_bytes) != expected_bytes:
      return False

    data[offset:offset + len(new_bytes)] = new_bytes

  with open(output_filepath, 'wb') as outfile:
    outfile.write(data)

  return True

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

patched_pci_bar_write = (
    # MOV DPTR, #0x7
    b'\x90\x00\x07'     
    # MOVX A, @DPTR
    b'\xE0'             
    # MOV R2, A
    b'\xFA'             
    # INC DPTR
    b'\xA3'             
    # MOVX A, @DPTR
    b'\xE0'             
    # ADD A, #0x10
    b'\x24\x10'         
    # MOV R1, A
    b'\xF9'             
    # CLR A
    b'\xE4'             
    # ADDC A, R2
    b'\x3A'             
    # MOV DPH, A
    b'\xF5\x83'         
    # MOV DPL, R1
    b'\x89\x82'         
    # LCALL copy4_from_dptr_into_r4_r7
    b'\x12\x74\xD7'     
    # LCALL copy4_into_b220
    b'\x12\x75\x09'     
    # LCALL SUB_CODE_b6e3 # call into return, not function
    b'\x12\xB6\xE3'     
    # RET
    b'\x22'             
)

patches = [
  (0x7a63, b'\xC6\x4B', b'\xB6\xB3'),
  (0x9f0c, patched_pci_bar_write_old, patched_pci_bar_write),
]

assert patch(sys.argv[1], sys.argv[2], patches) is True
