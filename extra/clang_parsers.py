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
