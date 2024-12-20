#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

# asm2x6x_tool.py - A tool to interact with ASM2x6x devices over USB.
# Copyright (C) 2022-2024  Forest Crossman <cyrozap@gmail.com>
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
import os
import struct
import sys
import time
from pathlib import Path

try:
    import sgio
except ModuleNotFoundError:
    sys.stderr.write("Error: Failed to import \"sgio\". Please install \"cython-sgio\", then try running this script again.\n")
    sys.exit(1)


class Asm2x6x:
    def __init__(self, dev_path):
        self._file = os.fdopen(os.open(dev_path, os.O_RDWR | os.O_NONBLOCK))

    def get_fw_version_data(self):
        return self.read(0x07f0, 6)

class Asm236x(Asm2x6x):
    def __init__(self, dev_path):
        super().__init__(dev_path)

    def flash_dump(self, read_len):
        data = bytearray(read_len)

        cdb = struct.pack('>BBI', 0xe2, 0x00, read_len)

        ret = sgio.execute(self._file, cdb, None, data)
        assert ret == 0

        return bytes(data)

    def config_write(self, config_data):
        assert len(config_data) == 128

        cdb = struct.pack('>B15x', 0xe1)

        ret = sgio.execute(self._file, cdb, config_data, None)
        assert ret == 0

    def fw_write(self, fw_data):
        cdb = struct.pack('>BBI', 0xe3, 0x00, len(fw_data))

        ret = sgio.execute(self._file, cdb, fw_data, None)
        assert ret == 0

    def read(self, start_addr, read_len, stride=255):
        data = bytearray(read_len)

        for i in range(0, read_len, stride):
            remaining = read_len - i
            buf_len = min(stride, remaining)

            cdb = struct.pack('>BBBHB', 0xe4, buf_len, 0x00, start_addr + i, 0x00)

            buf = bytearray(buf_len)
            ret = sgio.execute(self._file, cdb, None, buf)
            assert ret == 0

            data[i:i+buf_len] = buf

        return bytes(data)

    def write(self, start_addr, data):
        for offset, value in enumerate(data):
            cdb = struct.pack('>BBBHB', 0xe5, value, 0x00, start_addr + offset, 0x00)
            ret = sgio.execute(self._file, cdb, None, None)
            assert ret == 0

    def reload(self):
        cdb = bytes.fromhex("e8 00 00 00 00 00 00 00 00 00 00 00")
        ret = sgio.execute(self._file, cdb, None, None)
        assert ret == 0

    def flash_read(self, start_addr, read_len, stride=128):
        data = bytearray(read_len)

        for i in range(0, read_len, stride):
            remaining = read_len - i
            buf_len = min(stride, remaining)

            flash_addr = start_addr + i
            flash_addr_lo = flash_addr & 0xff
            flash_addr_md = (flash_addr >> 8) & 0xff
            flash_addr_hi = (flash_addr >> 16) & 0xff

            # Set FLASH_CON_MODE to read, with normal I/O config.
            self.write(0xC8AD, bytes([0x00]))

            # Set FLASH_CON_BUF_OFFSET to zero.
            self.write(0xC8AE, struct.pack('>H', 0x0000))

            # Set FLASH_CON_ADDR_LEN_MAYBE to 3.
            self.write(0xC8AC, bytes([0x03]))

            # Set the flash address.
            self.write(0xC8A1, bytes([flash_addr_lo]))
            self.write(0xC8A2, bytes([flash_addr_md]))
            self.write(0xC8AB, bytes([flash_addr_hi]))

            # Set FLASH_CON_DATA_LEN to the number of bytes to read.
            self.write(0xC8A3, struct.pack('>H', buf_len))

            # Set FLASH_CON_CSR bit 0 to start the read.
            self.write(0xC8A9, bytes([0x01]))

            # Wait for read to finish.
            while self.read(0xC8A9, 1)[0] & 1:
                continue

            buf = self.read(0x7000, buf_len)

            data[i:i+buf_len] = buf

        return bytes(data)

    def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
        assert byte_addr >> 12 == 0

        assert bus >> 8 == 0
        assert dev >> 5 == 0
        assert fn >> 3 == 0

        cfgreq_type = int(bus > 0)
        assert cfgreq_type >> 1 == 0

        fmt_type = 0x04
        if value is not None:
            fmt_type = 0x44

        fmt_type |= cfgreq_type
        address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)

        return self.pcie_gen_req(fmt_type, address, value, size)

    def pcie_mem_req(self, address, value=None, size=4):
        fmt_type = 0x00
        if value is not None:
            fmt_type = 0x40

        return self.pcie_gen_req(fmt_type, address, value, size)

    def pcie_gen_req(self, fmt_type, address, value=None, size=4):
        assert fmt_type >> 8 == 0
        assert size > 0 and size <= 4

        masked_address = address & 0xfffffffc
        offset = address & 0x00000003

        assert size + offset <= 4

        byte_enable = ((1 << size) - 1) << offset

        if value is not None:
            assert value >> (8 * size) == 0, f"{value}"
            shifted_value = value << (8 * offset)
            self.write(0xB220, struct.pack('>I', shifted_value))

        self.write(0xB210, struct.pack('>III',
            0x00000001 | (fmt_type << 24),
            byte_enable,
            masked_address,
        ))

        # Clear timeout bit.
        self.write(0xB296, bytes([0x01]))

        # Unknown
        self.write(0xB254, bytes([0x0f]))

        # Wait for PCIe to become ready.
        while self.read(0xB296, 1)[0] & 4 == 0:
            continue

        # Write to CSR bit 2 to send the TLP.
        self.write(0xB296, bytes([0x04]))

        if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000):
            # This is a posted transaction, so there's no completion and we can return early.
            return

        # Wait for completion.
        # print(self.read(0xB296, 1)[0])
        # print(self.read(0xB212, 1)[0])
        # print(self.read(0xB213, 1)[0])
        # print(self.read(0xB23e, 1)[0])
        while self.read(0xB296, 1)[0] & 2 == 0:
            # print(self.read(0xB296, 1)[0])
            if self.read(0xB296, 1)[0] & 1:
                # Clear timeout bit.
                self.write(0xB296, bytes([0x01]))

                raise Exception("PCIe timeout!")

        # Clear done bit.
        self.write(0xB296, bytes([0x02]))

        b284 = self.read(0xB284, 1)[0]
        #print("0xB284: 0x{:02x}".format(b284))
        b284_bit_0 = b284 & 0x01

        completion = struct.unpack('>III', self.read(0xB224, 12))
        # print("Completion TLP: 0x{:08x} 0x{:08x} 0x{:08x}".format(*completion))
        if (fmt_type & 0xbe == 0x04):
            # Completion TLPs for configuration requests always have a byte count of 4.
            assert completion[1] & 0xfff == 4
        else:
            assert completion[1] & 0xfff == size

        status_map = {
            0b000: "Successful Completion (SC)",
            0b001: "Unsupported Request (UR)",
            0b010: "Configuration Request Retry Status (CRS)",
            0b100: "Completer Abort (CA)",
        }
        status = (completion[1] >> 13) & 0x7
        if status or ((fmt_type & 0xbe == 0x04) and (((value is None) and (not b284_bit_0)) or ((value is not None) and b284_bit_0))):
            raise Exception("Completion status: {}, 0xB284 bit 0: {}".format(
                status_map.get(status, "Reserved (0b{:03b})".format(status)), b284_bit_0))

        if value is None:
            full_value = struct.unpack('>I', self.read(0xB220, 4))[0]
            shifted_value = full_value >> (8 * offset)
            masked_value = shifted_value & ((1 << (8 * size)) - 1)
            return masked_value

class Asm246x(Asm236x):
    def __init__(self, dev_path):
        super().__init__(dev_path)

    def flash_dump(self, read_len):
        first_read_len = read_len
        second_read_len = 0
        if read_len > 0x10000:
            first_read_len = 0xff00
            second_read_len = read_len - first_read_len

        cdb = struct.pack('>BBI', 0xe2, 0x50, first_read_len)
        first_data = bytearray(first_read_len)
        ret = sgio.execute(self._file, cdb, None, first_data)
        assert ret == 0

        time.sleep(1)

        data = bytes(first_data)
        if second_read_len:
            cdb = struct.pack('>BBI', 0xe2, 0xd0, second_read_len)
            second_data = bytearray(second_read_len)
            ret = sgio.execute(self._file, cdb, None, second_data)
            assert ret == 0

            time.sleep(1)

            data += bytes(second_data)

        return data

    # def reload(self):
    #     cdb = bytes.fromhex("e8 00 00 00 00 00 00 00")
    #     # print("here", cdb, len(cdb))
    #     # cdb = struct.pack('>B8x', 0xe8)
    #     # print("here", cdb, len(cdb))
    #     ret = sgio.execute(self._file, cdb, None, None)
    #     assert ret == 0

    def read(self, start_addr, read_len, stride=255):
        data = bytearray(read_len)

        for i in range(0, read_len, stride):
            remaining = read_len - i
            buf_len = min(stride, remaining)

            current_addr = start_addr + i
            assert current_addr >> 17 == 0
            current_addr &= 0x01ffff
            current_addr |= 0x500000

            cdb = struct.pack('>BBBHB', 0xe4, buf_len, current_addr >> 16, current_addr & 0xffff, 0x00)

            buf = bytearray(buf_len)
            ret = sgio.execute(self._file, cdb, None, buf)
            assert ret == 0

            data[i:i+buf_len] = buf

        return bytes(data)

    def write(self, start_addr, data):
        for offset, value in enumerate(data):
            current_addr = start_addr + offset
            # Ensure the address fits the expected range
            assert current_addr >> 17 == 0

            # Mask and shift address for the ASM246x's addressing scheme
            current_addr &= 0x01ffff
            current_addr |= 0x500000

            # Use command code 0xe5 similarly to ASM236x
            cdb = struct.pack('>BBBHB', 0xe5, value, current_addr >> 16, current_addr & 0xffff, 0x00)
            ret = sgio.execute(self._file, cdb, None, None)
            assert ret == 0


def get_asm2x6x_dev(device="auto"):
    if device == "auto":
        # Search for devices
        for path in Path("/sys/bus/scsi/devices").iterdir():
            try:
                vendor = open(path.joinpath("vendor"), "rb").read()
                if not vendor.startswith(b"ASMT"):
                    continue

                model = open(path.joinpath("model"), "rb").read()
                if not (model.startswith(b"ASM236") or model.startswith(b"ASM246")):
                    continue

                for sg in path.joinpath("scsi_generic").iterdir():
                    device = str(Path("/dev", sg.parts[-1]))
                    sys.stderr.write("Using {} device at \"{}\".\n".format(model.split(b' ')[0].decode('utf-8'), device))
                    break

                if device != "auto":
                    break
            except FileNotFoundError:
                continue

        if device == "auto":
            return None

    # Initialize the device object.
    dev = Asm236x(device)
    try:
        # This will fail if the device is an ASM246x
        dev.get_fw_version_data()
    except sgio.CheckConditionError:
        dev = Asm246x(device)

    return dev

def fw_version_bytes_to_string(version):
    return "{:02X}{:02X}{:02X}_{:02X}_{:02X}_{:02X}".format(*version)

def parse_bdf(bdf):
    bus = 0
    dev = 0
    fn = 0

    # [[[[<domain>]:]<bus>]:][<dev>][.[<fun>]]

    bdf_split_period = bdf.split('.')
    if len(bdf_split_period) > 2:
        raise Exception("Too many periods in BDF string \"{}\".".format(bdf))
    if len(bdf_split_period) > 1:
        fn_str = bdf_split_period[-1]
        if len(fn_str) > 0:
            try:
                fn = int(fn_str, 16)
            except Exception as e:
                raise Exception("Function number \"{}\" is invalid: {}".format(fn_str, e))
            assert fn >> 3 == 0

    bd_str = bdf_split_period[0]
    bd_str_split_colon = bd_str.split(":")
    if len(bd_str_split_colon) > 2:
        raise Exception("Too many colons in BDF string \"{}\".".format(bdf))
    if len(bd_str_split_colon) > 1:
        bus_str = bd_str_split_colon[0]
        if len(bus_str) > 0:
            try:
                bus = int(bus_str, 16)
            except Exception as e:
                raise Exception("Bus number \"{}\" is invalid: {}".format(bus_str, e))

    dev_str = bd_str_split_colon[-1]
    if len(dev_str) > 0:
        try:
            dev = int(dev_str, 16)
        except Exception as e:
            raise Exception("Device number \"{}\" is invalid: {}".format(dev_str, e))
        assert dev >> 5 == 0

    return (bus, dev, fn)

def dump(args, dev):
    start_addr = 0x0000
    read_len = 1 << 16
    stride = 128

    start_ns = time.perf_counter_ns()
    data = dev.read(start_addr, read_len, stride)
    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    print("Read {} bytes in {:.6f} seconds ({} bytes per second).".format(
        len(data), elapsed/1e9, int(len(data)*1e9) // elapsed))

    open(args.dump_file, 'wb').write(data)

    return 0

def flash_dump(args, dev):
    read_len = args.length

    start_ns = time.perf_counter_ns()
    data = dev.flash_dump(read_len)
    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    print("Read {} bytes in {:.6f} seconds ({} bytes per second).".format(
        len(data), elapsed/1e9, int(len(data)*1e9) // elapsed))

    open(args.flash_dump_file, 'wb').write(data)

    return 0

def fw_write(args, dev):
    fw_file = open(args.fw_file, 'rb').read()

    config_data = b''
    fw_data = fw_file
    if not args.raw:
        config_data = b'\xff' * 4 + fw_file[4:0x80]
        fw_data = fw_file[0x80:]

    if config_data:
        config_magic = config_data[-2]
        if config_magic != 0x5a:
            print("Error: Bad config magic. Expected 0x5a, got 0x{:02x}.".format(fw_magic))
            return 1
        config_checksum = config_data[-1]
        config_checksum_calc = sum(config_data[4:-1]) & 0xff
        if config_checksum_calc != config_checksum:
            print("Error: Bad config checksum. Expected 0x{:02x}, calculated 0x{:02x}.".format(config_checksum, config_checksum_calc))
            return 1

    fw_size = struct.unpack_from('<H', fw_data, 0)[0]
    fw_data = fw_data[:fw_size+8]
    fw_magic = fw_data[2+fw_size+0]
    if fw_magic not in (0x4b, 0x5a):
        print("Error: Bad FW magic. Expected 0x4b or 0x5a, got 0x{:02x}.".format(fw_magic))
        return 1
    fw_checksum_calc = sum(fw_data[2:2+fw_size]) & 0xff
    fw_checksum = fw_data[2+fw_size+1]
    if fw_checksum_calc != fw_checksum:
        print("Error: Bad FW checksum. Expected 0x{:02x}, calculated 0x{:02x}.".format(fw_checksum, fw_checksum_calc))
        return 1

    if config_data:
        start_ns = time.perf_counter_ns()
        data = dev.config_write(config_data)
        end_ns = time.perf_counter_ns()
        elapsed = end_ns - start_ns
        print("Wrote {} bytes in {:.6f} seconds ({} bytes per second).".format(
            len(config_data), elapsed/1e9, int(len(config_data)*1e9) // elapsed))

    start_ns = time.perf_counter_ns()
    data = dev.fw_write(fw_data)
    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    print("Wrote {} bytes in {:.6f} seconds ({} bytes per second).".format(
        len(fw_data), elapsed/1e9, int(len(fw_data)*1e9) // elapsed))

    start_ns = time.perf_counter_ns()
    rb_data = dev.flash_dump(0x80 + len(fw_data))
    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    print("Read back {} bytes in {:.6f} seconds ({} bytes per second).".format(
        len(rb_data), elapsed/1e9, int(len(rb_data)*1e9) // elapsed))

    errors = 0
    if config_data:
        for i in range(4, len(config_data)):
            rb_byte = rb_data[i]
            config_byte = config_data[i]
            if rb_byte != config_byte:
                errors += 1
                print("Error: Bad data at flash address 0x{:08x} (config offset 0x{:08x}): Expected 0x{:02x}, read 0x{:02x}.".format(
                    i, i, config_byte, rb_byte))

    for i in range(len(fw_data)):
        rb_byte = rb_data[0x80:][i]
        fw_byte = fw_data[i]
        if rb_byte != fw_byte:
            errors += 1
            print("Error: Bad data at flash address 0x{:08x} (firmware offset 0x{:08x}): Expected 0x{:02x}, read 0x{:02x}.".format(0x80 + i, i, fw_byte, rb_byte))

    if errors > 0:
        print("Error: Verification failed with {} errors!".format(errors))
        return 1

    return 0

def info(args, dev):
    print("Firmware version: {}".format(fw_version_bytes_to_string(dev.get_fw_version_data())))

    return 0

def read(args, dev):
    start_addr = int(args.address, 16)
    read_len = args.length
    stride = args.stride
    assert stride > 0
    assert stride < 256

    start_ns = time.perf_counter_ns()
    data = dev.read(start_addr, read_len, stride)
    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    print("Read {} bytes in {:.6f} seconds ({} bytes per second).".format(
        len(data), elapsed/1e9, int(len(data)*1e9) // elapsed))

    range_end_string = ""
    if read_len > 1:
        range_end_string = ":0x{:04X}".format(start_addr + read_len - 1)
    print("XDATA[0x{:04X}{}]: {} {}".format(start_addr, range_end_string, data.hex(), data))

    return 0

def write(args, dev):
    start_addr = int(args.address, 16)
    data = b"".join([bytes.fromhex(x) for x in args.data])

    if args.read_before:
        read_len = len(data)
        stride = min(read_len, 255)
        start_ns = time.perf_counter_ns()
        read_data = dev.read(start_addr, read_len, stride)
        end_ns = time.perf_counter_ns()
        elapsed = end_ns - start_ns
        print("Read {} bytes in {:.6f} seconds ({} bytes per second).".format(
            len(read_data), elapsed/1e9, int(len(read_data)*1e9) // elapsed))

        range_end_string = ""
        if read_len > 1:
            range_end_string = ":0x{:04X}".format(start_addr + read_len - 1)
        print("XDATA[0x{:04X}{}]: {} {}".format(
            start_addr, range_end_string, read_data.hex(), read_data))

    start_ns = time.perf_counter_ns()
    dev.write(start_addr, data)
    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    print("Wrote {} bytes in {:.6f} seconds ({} bytes per second).".format(
        len(data), elapsed/1e9, int(len(data)*1e9) // elapsed))

    range_end_string = ""
    if len(data) > 1:
        range_end_string = ":0x{:04X}".format(start_addr + len(data) - 1)
    print("XDATA[0x{:04X}{}]: {} {}".format(start_addr, range_end_string, data.hex(), data))

    if args.read_after:
        read_len = len(data)
        stride = min(read_len, 255)
        start_ns = time.perf_counter_ns()
        read_data = dev.read(start_addr, read_len, stride)
        end_ns = time.perf_counter_ns()
        elapsed = end_ns - start_ns
        print("Read {} bytes in {:.6f} seconds ({} bytes per second).".format(
            len(read_data), elapsed/1e9, int(len(read_data)*1e9) // elapsed))

        range_end_string = ""
        if read_len > 1:
            range_end_string = ":0x{:04X}".format(start_addr + read_len - 1)
        print("XDATA[0x{:04X}{}]: {} {}".format(
            start_addr, range_end_string, read_data.hex(), read_data))

    return 0

def reload(args, dev):
    dev.reload()

    return 0

def memtest(args, dev):
    start_addr = int(args.address, 16)
    test_len = args.length
    stride = args.stride

    print("Testing data from 0x{:04x} to 0x{:04x}, inclusive...".format(start_addr, start_addr+test_len-1))

    zeros = bytes(test_len)

    tests = (
        ("Random 1", os.urandom(test_len)),
        ("Zeros 1", zeros),
        ("Random 2", os.urandom(test_len)),
        ("Zeros 2", zeros),
    )

    for test_name, test_data in tests:
        print("Running test \"{}\"...".format(test_name))

        before = dev.read(start_addr, test_len, stride)
        dev.write(start_addr, test_data)
        after = dev.read(start_addr, test_len, stride)

        if after != test_data:
            print("Error: Failed test \"{}\"!".format(test_name))
            for i in range(test_len):
                if after[i] != test_data[i]:
                    print(" + Mismatch at address 0x{:04x}: Expected 0x{:02x}, got 0x{:02x} instead.".format(
                        start_addr + i, test_data[i], after[i]))
                    break

            if after == before:
                print(" + Write had no effect--this is likely read-only memory.")

            return 1

    return 0

def flash_read(args, dev):
    start_addr = int(args.address, 16)
    read_len = args.length
    stride = args.stride
    assert stride > 0
    assert stride <= 4096

    start_ns = time.perf_counter_ns()
    data = dev.flash_read(start_addr, read_len, stride)
    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    print("Read {} bytes in {:.6f} seconds ({} bytes per second).".format(
        len(data), elapsed/1e9, int(len(data)*1e9) // elapsed))

    range_end_string = ""
    if read_len > 1:
        range_end_string = ":0x{:04X}".format(start_addr + read_len - 1)
    print("FLASH[0x{:04X}{}]: {} {}".format(start_addr, range_end_string, data.hex(), data))

    return 0

def pcie(args, dev):
    value = None
    if args.value:
        value = int(args.value, 16)

    size = 4

    addr_parts = args.address.split('.')
    if len(addr_parts) == 2:
        size = {
            'b': 1,
            'w': 2,
            'l': 4,
        }.get(addr_parts[-1].lower())
        if not size:
            raise ValueError("Invalid address specifier \"{}\"".format(args.address))
    elif len(addr_parts) > 2:
        raise ValueError("Invalid address specifier \"{}\"".format(args.address))
    byte_addr = int(addr_parts[0], 16)

    if args.bdf:
        mem_type = "CFG"

        bdf = parse_bdf(args.bdf)

        cfgreq_type = 0
        if bdf[0] != 0:
            cfgreq_type = 1

        addr_str = "0x{:04X}".format(byte_addr)

        # print(byte_addr, cfgreq_type, value, size)
        data = dev.pcie_cfg_req(byte_addr, bus=bdf[0], dev=bdf[1], fn=bdf[2], cfgreq_type=cfgreq_type, value=value, size=size)

    else:
        mem_type = "MEM"

        addr_str = "0x{:08X}".format(byte_addr)

        data = dev.pcie_mem_req(byte_addr, value, size)

    if value is None:
        data_str = {
            1: "0x{:02x}",
            2: "0x{:04x}",
            4: "0x{:08x}",
        }[size].format(data)
        print("{}[{}]: {}".format(mem_type, addr_str, data_str))

    return 0

def pcie_cfg_dump(args, dev):
    bdf = parse_bdf(args.bdf)

    cfgreq_type = 0
    if bdf[0] != 0:
        cfgreq_type = 1

    buf = bytearray(4096)
    start_ns = time.perf_counter_ns()
    for addr in range(0, len(buf), 4):
        struct.pack_into('<I', buf, addr, dev.pcie_cfg_req(addr, bus=bdf[0], dev=bdf[1], fn=bdf[2], cfgreq_type=cfgreq_type))
    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    if args.output:
        print("Read {} bytes in {:.6f} seconds ({} bytes per second).".format(
            len(buf), elapsed/1e9, int(len(buf)*1e9) // elapsed))

    dump = sys.stdout
    if args.output:
        dump = open(args.output, 'w')

    dump.write("{:02x}:{:02x}.{:x} \n".format(*bdf))
    for addr in range(0, len(buf), 16):
        dump.write("{:03x}: {}\n".format(addr, " ".join(["{:02x}".format(b) for b in buf[addr:addr+16]])))
    dump.write("\n")

    if args.output:
        print("Wrote config space dump to \"{}\". Use \"lspci -A dump -O dump.name={}\" to read the file.".format(
            args.output, args.output))

    return 0

def pcie_mem_dump(args, dev):
    start_addr = int(args.address, 16)

    if args.length.startswith("0x"):
        length = int(args.length, 16)
    else:
        length = int(args.length)

    buf = bytearray(length)

    start_ns = time.perf_counter_ns()

    offset = start_addr & 0x3
    unaligned_len = 0
    if offset:
        unaligned_len = 4 - offset
        for i in range(unaligned_len):
            buf[i] = dev.pcie_mem_req(start_addr + i, size=1)

    aligned_len = ((len(buf) - unaligned_len) // 4) * 4
    for i in range(0, aligned_len, 4):
        buf_idx = unaligned_len + i
        struct.pack_into('<I', buf, buf_idx, dev.pcie_mem_req(start_addr + buf_idx, size=4))

    remaining = len(buf) - unaligned_len - aligned_len
    for i in range(remaining):
        buf_idx = unaligned_len + aligned_len + i
        buf[buf_idx] = dev.pcie_mem_req(start_addr + buf_idx, size=1)

    end_ns = time.perf_counter_ns()
    elapsed = end_ns - start_ns
    print("Read {} bytes in {:.6f} seconds ({} bytes per second).".format(
        len(buf), elapsed/1e9, int(len(buf)*1e9) // elapsed))

    dump = open(args.pcie_mem_dump_file, 'wb')
    dump.write(buf)
    dump.close()

    print("Wrote memory dump to \"{}\".".format(args.pcie_mem_dump_file))

    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="auto", help="The path to the ASM2x6x SCSI/SG_IO device. Default: auto")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands.")

    parser_dump = subparsers.add_parser("dump")
    parser_dump.add_argument("dump_file", help="The file to write the memory dump output to.")
    parser_dump.set_defaults(func=dump)

    parser_flash_dump = subparsers.add_parser("flash_dump")
    parser_flash_dump.add_argument("-l", "--length", type=int, default=512*1024, help="The total number of bytes to read from flash. Default: 524288")
    parser_flash_dump.add_argument("flash_dump_file", help="The file to write the flash dump output to.")
    parser_flash_dump.set_defaults(func=flash_dump)

    parser_fw_write = subparsers.add_parser("fw_write")
    parser_fw_write.add_argument("-r", "--raw", action='store_true', default=False, help="Write raw flash data, not a parsed firmware image. Default: False")
    parser_fw_write.add_argument("fw_file", help="The firmware file to write to flash.")
    parser_fw_write.set_defaults(func=fw_write)

    parser_info = subparsers.add_parser("info")
    parser_info.set_defaults(func=info)

    parser_read = subparsers.add_parser("read")
    parser_read.add_argument("-s", "--stride", type=int, default=255, help="The number of bytes to read with each SCSI command. Min: 1, Max: 255, Default: 255")
    parser_read.add_argument("-l", "--length", type=int, default=1, help="The total number of bytes to read. Default: 1")
    parser_read.add_argument("address", type=str, help="The address to start the read from, in hexadecimal.")
    parser_read.set_defaults(func=read)

    parser_write = subparsers.add_parser("write")
    parser_write.add_argument("-b", "--read-before", action='store_true', default=False, help="Perform a read before writing. Default: False")
    parser_write.add_argument("-a", "--read-after", action='store_true', default=False, help="Perform a read after writing. Default: False")
    parser_write.add_argument("address", type=str, help="The address to start the write to, in hexadecimal.")
    parser_write.add_argument("data", type=str, nargs="+", help="The data bytes to be written, in hexadecimal.")
    parser_write.set_defaults(func=write)

    parser_reload = subparsers.add_parser("reload")
    parser_reload.set_defaults(func=reload)

    parser_memtest = subparsers.add_parser("memtest")
    parser_memtest.add_argument("-s", "--stride", type=int, default=1, help="The number of bytes to read with each SCSI command. Min: 1, Max: 255, Default: 1")
    parser_memtest.add_argument("address", type=str, help="The address to start the test from, in hexadecimal.")
    parser_memtest.add_argument("length", type=int, default=1, help="The total number of bytes to test. Default: 1")
    parser_memtest.set_defaults(func=memtest)

    parser_flash_read = subparsers.add_parser("flash_read")
    parser_flash_read.add_argument("-s", "--stride", type=int, default=128, help="The number of bytes to read from flash with each internal flash command. Min: 1, Max: 4096, Default: 128")
    parser_flash_read.add_argument("-l", "--length", type=int, default=1, help="The total number of bytes to read from flash. Default: 1")
    parser_flash_read.add_argument("address", type=str, help="The flash address to start the read from, in hexadecimal.")
    parser_flash_read.set_defaults(func=flash_read)

    parser_pcie_cfg_dump = subparsers.add_parser("pcie_cfg_dump")
    parser_pcie_cfg_dump.add_argument("-s", "--bdf", type=str, default="00:00.0", help="The PCI address to dump the config space of. Default: 00:00.0")
    parser_pcie_cfg_dump.add_argument("-o", "--output", type=str, default="", help="The file to write the PCI config space dump output to. Default: standard output")
    parser_pcie_cfg_dump.set_defaults(func=pcie_cfg_dump)

    parser_pcie_mem_dump = subparsers.add_parser("pcie_mem_dump")
    parser_pcie_mem_dump.add_argument("address", type=str, help="The address to start the read from, in hexadecimal.")
    parser_pcie_mem_dump.add_argument("length", type=str, help="The number of bytes to read, in decimal (prefix with \"0x\" for hexadecimal).")
    parser_pcie_mem_dump.add_argument("pcie_mem_dump_file", help="The file to write the dump output to.")
    parser_pcie_mem_dump.set_defaults(func=pcie_mem_dump)

    parser_pcie = subparsers.add_parser("pcie")
    parser_pcie.add_argument("-s", "--bdf", type=str, default=None, help="The PCI address to send the Configuration Request to. Default: None (send Memory Request)")
    parser_pcie.add_argument("-v", "--value", type=str, default=None, help="The value to write. Default: None (Read)")
    parser_pcie.add_argument("address", type=str, help="The address to read from or write to, in hexadecimal. To specify the width of the data to read/write, append \".B\" (1 byte), \".W\" (2 bytes), and \".L\" (4 bytes) to the address. The default width is 4 and the specifiers are case-insensitive.")
    parser_pcie.set_defaults(func=pcie)

    args = parser.parse_args()

    dev = get_asm2x6x_dev(args.device)
    if not dev:
        sys.stderr.write("Error: Failed to auto-detect an ASM2x6x device. Please specify it manually using the \"-d\" flag.\n")
        return 1
    
    # data = dev.pcie_cfg_req(4, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x7, size=2)
    # print(dev.pcie_cfg_req(34, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))
    # data = dev.pcie_cfg_req(70, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x20, size=2)
    import time
    time.sleep(2)
    data = dev.pcie_cfg_req(70, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x10, size=2)
    print("Header", dev.pcie_cfg_req(0x0e, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))
    print("Primary", dev.pcie_cfg_req(0x18, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))
    print("Secondary", dev.pcie_cfg_req(0x19, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))
    print("Subordinate", dev.pcie_cfg_req(0x1a, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))

    cap_ptr = dev.pcie_cfg_req(0x34, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1)
    caps = {}
    while True:
        cap_0 = dev.pcie_cfg_req(cap_ptr, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1)
        cap_nxt = dev.pcie_cfg_req(cap_ptr+1, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1)
        cap_len = dev.pcie_cfg_req(cap_ptr+2, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1)
        print(cap_0, cap_nxt, cap_len)
        caps[cap_0] = cap_ptr
        if cap_nxt == 0: break
        cap_ptr = cap_nxt
    
    print("PM cap off", caps[0x1])
    
    pm_state = dev.pcie_cfg_req(caps[0x1]+4, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1)
    print("PM cap now", pm_state)
    dev.pcie_cfg_req(caps[0x1]+4, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x0, size=1)
    print(dev.pcie_cfg_req(caps[0x1]+4, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))

    for i in range(6):
        print("BAR", i, hex(dev.pcie_cfg_req(0x10 + 4 * i, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=4)))

    dev.pcie_cfg_req(0x1c, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x10, size=1)
    dev.pcie_cfg_req(0x1d, bus=1, dev=0, fn=0, cfgreq_type=1, value=0xff, size=1)
    
    dev.pcie_cfg_req(0x20, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x1000, size=2)
    dev.pcie_cfg_req(0x22, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x4000, size=2)

    dev.pcie_cfg_req(0x24, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x4000, size=2)
    dev.pcie_cfg_req(0x26, bus=1, dev=0, fn=0, cfgreq_type=1, value=0xffff, size=2)

    dev.pcie_cfg_req(0x1a, bus=1, dev=0, fn=0, cfgreq_type=1, value=3, size=1)
    dev.pcie_cfg_req(0x19, bus=1, dev=0, fn=0, cfgreq_type=1, value=2, size=1)
    dev.pcie_cfg_req(0x18, bus=1, dev=0, fn=0, cfgreq_type=1, value=1, size=1)

    print("br ctrl", dev.pcie_cfg_req(0x3e, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))
    dev.pcie_cfg_req(0x3e, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x40, size=1)
    time.sleep(1)
    dev.pcie_cfg_req(0x3e, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x3, size=1)

    data = dev.pcie_cfg_req(4, bus=1, dev=0, fn=0, cfgreq_type=1, value=0x7, size=1)
    print(dev.pcie_cfg_req(4, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))

    # Now do the same for downstram port
    
    pm_state = dev.pcie_cfg_req(caps[0x1]+4, bus=2, dev=0, fn=0, cfgreq_type=1, value=None, size=1)
    print("PM cap now", pm_state)
    dev.pcie_cfg_req(caps[0x1]+4, bus=2, dev=0, fn=0, cfgreq_type=1, value=0x0, size=1)
    print(dev.pcie_cfg_req(caps[0x1]+4, bus=2, dev=0, fn=0, cfgreq_type=1, value=None, size=1))

    for i in range(6):
        print("BAR", i, hex(dev.pcie_cfg_req(0x10 + 4 * i, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=4)))

    dev.pcie_cfg_req(0x1c, bus=2, dev=0, fn=0, cfgreq_type=1, value=0x10, size=1)
    dev.pcie_cfg_req(0x1d, bus=2, dev=0, fn=0, cfgreq_type=1, value=0xff, size=1)

    dev.pcie_cfg_req(0x20, bus=2, dev=0, fn=0, cfgreq_type=1, value=0x1000, size=2)
    dev.pcie_cfg_req(0x22, bus=2, dev=0, fn=0, cfgreq_type=1, value=0x4000, size=2)

    dev.pcie_cfg_req(0x24, bus=2, dev=0, fn=0, cfgreq_type=1, value=0x4000, size=2)
    dev.pcie_cfg_req(0x26, bus=2, dev=0, fn=0, cfgreq_type=1, value=0xffff, size=2)
    
    dev.pcie_cfg_req(0x1a, bus=2, dev=0, fn=0, cfgreq_type=1, value=3, size=1)
    dev.pcie_cfg_req(0x19, bus=2, dev=0, fn=0, cfgreq_type=1, value=3, size=1)
    dev.pcie_cfg_req(0x18, bus=2, dev=0, fn=0, cfgreq_type=1, value=2, size=1)

    print("br ctrl", dev.pcie_cfg_req(0x3e, bus=1, dev=0, fn=0, cfgreq_type=1, value=None, size=1))
    dev.pcie_cfg_req(0x3e, bus=2, dev=0, fn=0, cfgreq_type=1, value=0x40, size=1)
    time.sleep(1)
    dev.pcie_cfg_req(0x3e, bus=2, dev=0, fn=0, cfgreq_type=1, value=0x3, size=1)

    data = dev.pcie_cfg_req(4, bus=2, dev=0, fn=0, cfgreq_type=1, value=0x7, size=1)
    print(dev.pcie_cfg_req(4, bus=2, dev=0, fn=0, cfgreq_type=1, value=None, size=1))

    for bus in range(1, 32):
        for fn in range(0, 8):
            try:
                print(bus, fn, hex(dev.pcie_cfg_req(0, bus=bus, dev=0, fn=fn, cfgreq_type=1, value=None, size=2)))
            except: pass

    data = dev.pcie_cfg_req(4, bus=3, dev=0, fn=0, cfgreq_type=1, value=0x7, size=1)

    def round_up(num:int, amt:int) -> int: return (num+amt-1)//amt * amt
    bars = {}
    mmio_start = 0x1000
    bar_start = 0x40000000
    next_64 = 0
    bar = 0
    for i in range(12):
        if next_64:
            dev.pcie_cfg_req(0x10 + 4 * i, bus=3, dev=0, fn=0, cfgreq_type=1, value=0x0, size=4)
            next_64 = False
            continue

        x = dev.pcie_cfg_req(0x10 + 4 * i, bus=3, dev=0, fn=0, cfgreq_type=1, value=None, size=4)
        addr = x & 0xFFFFFFF0
        flags = x & 0xf

        if flags & 1 == 0:
            dev.pcie_cfg_req(0x10 + 4 * i, bus=3, dev=0, fn=0, cfgreq_type=1, value=0xffffffff, size=4)

            bs = dev.pcie_cfg_req(0x10 + 4 * i, bus=3, dev=0, fn=0, cfgreq_type=1, value=None, size=4)
            bar_size = 0xffffffff - (bs & 0xFFFFFFF0) + 1
            
            if bar_size < 0x100000000:
                bar_start += round_up(bar_size, 2 << 20)
                print(bar, hex(bar_size), hex(bar_start))
                
                dev.pcie_cfg_req(0x10 + 4 * i, bus=3, dev=0, fn=0, cfgreq_type=1, value=bar_start, size=4)
                x = dev.pcie_cfg_req(0x10 + 4 * i, bus=3, dev=0, fn=0, cfgreq_type=1, value=None, size=4)
                # print(bar, bar_size, x)

                addr = x & 0xFFFFFFF0
                print(bar, addr, flags)

                bars[bar] = (addr, bar_size)
        else:
            # dev.pcie_cfg_req(0x10 + 4 * i, bus=3, dev=0, fn=0, cfgreq_type=1, value=0xffffffff, size=4)

            # bs = dev.pcie_cfg_req(0x10 + 4 * i, bus=3, dev=0, fn=0, cfgreq_type=1, value=None, size=4)
            # bar_size = 0xffffffff - (bs & 0xFFFFFFF0) + 1

            # print("bar", bar, flags, "IO??", hex(bar_size))
            pass


        if flags & 0x8 == 0x8: 
            next_64 = True
        bar += 1

        if bar == 6: break

    
    dev.pcie_cfg_req(0x20, bus=0, dev=0, fn=0, cfgreq_type=0, value=0x1000, size=2)
    dev.pcie_cfg_req(0x22, bus=0, dev=0, fn=0, cfgreq_type=0, value=0x4000, size=2)

    dev.pcie_cfg_req(0x24, bus=0, dev=0, fn=0, cfgreq_type=0, value=0x4000, size=2)
    dev.pcie_cfg_req(0x26, bus=0, dev=0, fn=0, cfgreq_type=0, value=0xffff, size=2)

    print(bars)

    # dev.pcie_cfg_req(0x1a, bus=0, dev=0, fn=0, cfgreq_type=0, value=3, size=1)
    # dev.pcie_cfg_req(0x19, bus=0, dev=0, fn=0, cfgreq_type=0, value=3, size=1)
    # dev.pcie_cfg_req(0x18, bus=0, dev=0, fn=0, cfgreq_type=0, value=0, size=1)

    # print(dev.pcie_cfg_req(0x19, bus=0, dev=0, fn=0, cfgreq_type=0, value=None, size=1))
    # print(dev.pcie_cfg_req(0x22, bus=0, dev=0, fn=0, cfgreq_type=0, value=None, size=2))

    aa, size = bars[0]
    print(hex(dev.pcie_mem_req(aa, None, size=4)))

    x = dev.pcie_mem_req(aa, 0x0, size=4)
    print(x)

    # pci_read_config_word(dev, PCI_BRIDGE_CONTROL, &ctrl);
	# ctrl |= PCI_BRIDGE_CTL_BUS_RESET;
	# pci_write_config_word(dev, PCI_BRIDGE_CONTROL, ctrl);

	# /*
	#  * PCI spec v3.0 7.6.4.2 requires minimum Trst of 1ms.  Double
	#  * this to 2ms to ensure that we meet the minimum requirement.
	#  */
	# msleep(2);

	# ctrl &= ~PCI_BRIDGE_CTL_BUS_RESET;
	# pci_write_config_word(dev, PCI_BRIDGE_CONTROL, ctrl);



    return args.func(args, dev)


if __name__ == "__main__":
    sys.exit(main())
