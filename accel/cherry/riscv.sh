#!/bin/bash -e
cd out
riscv64-unknown-elf-gcc -Os -march=rv32i -mabi=ilp32 -nostdlib ../src/main.c
#riscv64-unknown-elf-as ../src/riscv.asm
riscv64-unknown-elf-objdump -d a.out
riscv64-unknown-elf-objcopy -O binary a.out a.asm
xxd a.asm
python -c 'import struct; dat = open("a.asm", "rb").read(); print("\n".join(["%08x" % c for c in struct.unpack("I"*(len(dat)//4), dat)]))' > ../src/firmware.hex


