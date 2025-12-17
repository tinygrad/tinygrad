import msgpack, struct, pathlib
from tinygrad.runtime.support.elf import elf_loader

with open(pathlib.Path(__file__).parent/"lib", "rb") as f: lib = f.read()
_, sections, __ = elf_loader(lib)
data = next((s for s in sections if s.name.startswith(".note"))).content
namesz, descsz, typ = struct.unpack_from(hdr:="<III", data, 0)
offset = (struct.calcsize(hdr)+namesz+3) & -4
notes = msgpack.unpackb(data[offset:offset+descsz])
print(notes)
