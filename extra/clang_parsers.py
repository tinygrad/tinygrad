import sys
if sys.platform == "darwin":
    from macholib.MachO import MachO
from io import BytesIO
from struct import unpack

class Reader:
    def __init__(self, data, big_endian=False) -> None:
        self.cursor = 0
        self.big_endian = big_endian
        self.data = data

    def _read(self, type, size):
        if self.big_endian:
            res = unpack(f">{type}", self.data[self.cursor:self.cursor + size])[0]
        else:
            res = unpack(f"<{type}", self.data[self.cursor:self.cursor + size])[0]
        self.cursor += size
        return res
    
    def next_uint8(self): return self._read("B", 1)
    def next_uint16(self): return self._read("H", 2)
    def next_uint32(self): return self._read("I", 4)
    def next_uint64(self): return self._read("Q", 8)

def read_null_terminated_string(data):
    buffer = b""
    i = 0
    while True:
        ch = data[i:i+1]
        if ch[0] == 0: break
        buffer += ch
        i += 1
    return str(buffer, encoding="ascii")

if sys.platform == "darwin":
    class MyMachO(MachO):
        def __init__(self, data):
            self.data = data
            self.allow_unknown_load_commands = False
            self.fat = None
            self.headers = []
            self.filename = ""
            self.load(BytesIO(data))

        def extract_segment64(self):
            for header in self.headers:
                for command in header.commands[0]:
                    if str(type(command)) == "<class 'macholib.mach_o.segment_command_64'>":
                        return command.fileoff

        def extract_offset_and_symbols(self):
            segment_offset = self.extract_segment64()
            st = self.headers[0].getSymbolTableCommand()
            st_offset = st.symoff
            st_str_offset = st.stroff

            symbols = {}

            for i in range(st.nsyms):
                r = Reader(self.data[st_offset + (16 * i):])
                str_index = r.next_uint32()
                t = r.next_uint8()
                #if t != 15: continue
                r.next_uint8()
                r.next_uint16()
                name = read_null_terminated_string(self.data[st_str_offset + str_index:])
                addr = r.next_uint64()
                symbols[name] = addr

            return segment_offset, symbols

class ELFParser:
    def parse_section_headers(self):
        self.section_headers = {}
        offset = self.sh
        for i in range(0, self.sh_num * self.sh_entry_size, self.sh_entry_size):
            name_offset = unpack("<I", self.data[offset+i:offset+i+4])[0]
            name = read_null_terminated_string(self.data[self.string_table_offset + name_offset:])
            
            size = unpack("<Q", self.data[offset+i+0x20:offset+i+0x20+8])[0]
            file_offset = unpack("<Q", self.data[offset+i+0x18:offset+i+0x18+8])[0]
            self.section_headers[name] = (file_offset, size)

    def extract_symbol_table(self):
        file_offset, size = self.section_headers[".symtab"]
        string_table_offset = self.section_headers[".strtab"][0] 
        num_symbols = size // 0x18  # Each symbol entry is 24 bytes (0x18 in hex)

        self.symbol_table = {}
        for i in range(num_symbols):
            offset = file_offset + (i * 0x18)

            st_name = unpack("<I", self.data[offset:offset + 4])[0]  # Name offset
            st_info = unpack("<B", self.data[offset + 4:offset + 5])[0]  # Info byte
            st_other = unpack("<B", self.data[offset + 5:offset + 6])[0]  # Other byte
            st_shndx = unpack("<H", self.data[offset + 6:offset + 8])[0]  # Section index
            st_value = unpack("<Q", self.data[offset + 8:offset + 16])[0]  # Value
            st_size = unpack("<Q", self.data[offset + 16:offset + 24])[0]  # Size

            symbol_name = read_null_terminated_string(self.data[string_table_offset + st_name:])

            self.symbol_table[symbol_name] = {
                'info': st_info,
                'other': st_other,
                'shndx': st_shndx,
                'value': st_value,
                'size': st_size
            }

    def extract_string_table_offset(self):
        index = self.sh_name_index
        offset = self.sh + (self.sh_entry_size * index)
        size = unpack("<Q", self.data[offset+0x20:offset+0x20+8])[0]
        self.string_table_offset = unpack("<Q", self.data[offset+0x18:offset+0x18+8])[0]

    def __init__(self, data):
        self.data = data
        assert data[:4] == bytes([0x7f, 0x45, 0x4c, 0x46]), "Invalid magic"
        assert data[4] == 2, "Only 64 bit is supported"
        self.ph = unpack("<Q", data[0x20:0x20+8])[0]
        self.sh = unpack("<Q", data[0x28:0x28+8])[0]
        self.ph_entry_size = unpack("<H", data[0x36:0x36+2])[0]
        self.ph_num = unpack("<H", data[0x38:0x38+2])[0]
        self.sh_entry_size = unpack("<H", data[0x3A:0x3A+2])[0]
        self.sh_num = unpack("<H", data[0x3C:0x3C+2])[0]
        self.sh_name_index = unpack("<H", data[0x3E:0x3E+2])[0]
        self.extract_string_table_offset()
        self.parse_section_headers()
        self.extract_symbol_table()


