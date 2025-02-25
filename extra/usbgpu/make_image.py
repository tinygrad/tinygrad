#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

# make_image.py - Script to generate a firmware image from a raw binary.
# Copyright (C) 2022-2023  Forest Crossman <cyrozap@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import argparse
import struct
import sys

from datetime import datetime


CHIP_INFO = {
    "ASM2362": (0x5a, 0x2362),
    "ASM2364": (0x4b, 0x2364),
}


def checksum(data : bytes):
    return sum(data) & 0xff

def gen_string(s : str, size : int):
    s = s.encode('ascii')
    if len(s) > size:
        raise ValueError("String of size {} is too large for field of size {}".format(len(s), size))

    padding = b'\xff' * (size - len(s))
    return s + padding

def gen_config(chip : str):
    usb_pid = CHIP_INFO[chip][1]

    # Unknown
    config = b'\xff' * 4

    # Strings
    config += gen_string("0" * 16, 20)  # Serial number
    config += gen_string("ASMedia", 36)  # EP0 Manufacturer String
    config += gen_string("ASMT", 8)  # T10 Manufacturer String
    config += gen_string("ASM236x series", 32)  # EP0 Product String
    config += gen_string("ASM236x NVMe", 16)  # T10 Product String

    # USB VID, PID, and device BCD
    config += struct.pack('<HHH', 0x174c, usb_pid, 0x0100)

    lp_if_u3 = 3
    lp_if_idle = 3
    idle_timer = 3

    unk7b_76 = 3
    pcie_lane = 3
    pcie_speed = 3
    pcie_aspm = 3

    unk7c = 2

    disable_slow_enumeration = 1
    disable_2tb = 1
    disable_low_power_mode = 1
    disable_u1u2 = 1
    disable_wtg = 1
    disable_two_leds = 1
    disable_eup = 1
    disable_usb_removable = 1

    config += bytes([
        (lp_if_u3 << 6) | (lp_if_idle << 4) | idle_timer,
        (unk7b_76 << 6) | (pcie_lane << 4) | (pcie_speed << 2) | (pcie_aspm << 0),
        unk7c,
        (disable_slow_enumeration << 7) | (disable_2tb << 6) | (disable_low_power_mode << 5) | (disable_u1u2 << 4) |
            (disable_wtg << 3) | (disable_two_leds << 2) | (disable_eup << 1) | (disable_usb_removable << 0),
    ])

    config += bytes([0x5a])  # Magic

    config += bytes([checksum(config[4:])])  # Checksum

    return config

def gen_fw(chip : str, code : bytes):
    body_magic = CHIP_INFO[chip][0]

    code = bytearray(code)
    bcd_timestamp = bytes.fromhex(datetime.now().strftime('%y%m%d%H%M%S'))
    struct.pack_into('6s', code, 0x200, bcd_timestamp)

    data = struct.pack('<H', len(code)) + code + bytes([body_magic, checksum(code)])

    padding_len = 16 - (len(data) % 16)
    padding = bytes(padding_len)
    if padding_len == 16:
        padding = b''

    return bytes(data) + padding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input binary.")
    parser.add_argument("-c", "--chip", type=str, choices=CHIP_INFO.keys(), default="ASM2362", help="Chip to target.")
    parser.add_argument("-t", "--type", type=str, choices=["flash", "fw"], default="fw", help="Image type.")
    parser.add_argument("-o", "--output", type=str, default="firmware.bin", help="Output image.")
    args = parser.parse_args()

    binary = open(args.input, 'rb').read()

    if args.type == "fw":
        image = gen_fw(args.chip, binary)
    elif args.type == "flash":
        config = gen_config(args.chip)
        image = config + binary
    else:
        print("Error: Unrecognized image type: {}".format(args.type))
        return 1

    output = open(args.output, 'wb')
    output.write(image)
    output.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
